"""Dual-stream YOLOv8m RGBT detector: CPCA + DSAM-Bneck96 + FSPR-Dense with the joint prior bridge (v8m_bneck96_dense_s2)."""
from contextlib import nullcontext
from copy import deepcopy
import torch
from torch import nn
import torch.nn.functional as F
from ultralytics.nn.tasks import DetectionModel
from tricue_ops.operators import FSPR
from tricue_ops.hfp import HFP
from reference.cpca import CPCA

VERSION = 'rgbt-det80-v1'
TAIL_BATCH_THRESHOLD = 16

SCALE_CONFIGS = {
    'm': {'yaml': 'yolov8m.yaml', 'channels': (192, 384, 576)},
}

# (scale, g1_kind, dsam_cfg, fspr_dense, step2_joint_bridge)
VARIANT_SPECS = {
    'v8m_add': ('m', None, None, False, False),
    'v8m_bneck96_dense_s2': ('m', 'cpca', {'w': 96, 'sep': False, 'bneck': True}, True, True),
}

EXPECTED = {
    'v8m_add': 37713913,
    'v8m_bneck96_dense_s2': 51980833,
}


class Residual(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        self.gamma = nn.Parameter(torch.zeros(()))


def _conv3x3(in_c, out_c, padding=1, dilation=1, bias=True, sep=False):
    if sep:
        return nn.Sequential(
            nn.Conv2d(in_c, in_c, 3, padding=padding, dilation=dilation, groups=in_c, bias=False),
            nn.Conv2d(in_c, out_c, 1, bias=bias),
        )
    return nn.Conv2d(in_c, out_c, 3, stride=1, padding=padding, dilation=dilation, bias=bias)


class PredLayer(nn.Module):
    def __init__(self, in_c, width, sep=False, bneck=False):
        super().__init__()
        if bneck:
            self.enlayer = nn.Sequential(
                nn.Conv2d(in_c, width, 1, bias=False),
                nn.BatchNorm2d(width),
                nn.ReLU(inplace=True),
                _conv3x3(width, width, padding=1, dilation=1, bias=True, sep=sep),
                nn.BatchNorm2d(width),
                nn.ReLU(inplace=True),
            )
        else:
            self.enlayer = nn.Sequential(
                _conv3x3(in_c, width, padding=1, dilation=1, bias=True, sep=sep),
                nn.BatchNorm2d(width),
                nn.ReLU(inplace=True),
            )
        self.outlayer = nn.Sequential(nn.Conv2d(width, 1, kernel_size=1, stride=1, padding=0))

    def forward(self, x):
        feat = self.enlayer(x)
        pred = self.outlayer(feat)
        return feat, pred


class ASPPBlock(nn.Module):
    def __init__(self, in_c, width, sep=False):
        super().__init__()
        self.aspp1 = nn.Sequential(
            nn.Conv2d(in_c, width, 1, 1),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )
        self.aspp2 = nn.Sequential(
            _conv3x3(in_c, width, padding=3, dilation=3, bias=True, sep=sep),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )
        self.aspp3 = nn.Sequential(
            _conv3x3(in_c, width, padding=5, dilation=5, bias=True, sep=sep),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )
        self.aspp4 = nn.Sequential(
            _conv3x3(in_c, width, padding=7, dilation=7, bias=True, sep=sep),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return torch.cat((self.aspp1(x), self.aspp2(x), self.aspp3(x), self.aspp4(x)), dim=1)


class _InnerDSAM(nn.Module):
    """Matches original DSAM attribute layout (ff_conv, bf_conv, rgbd_pred_layer) for audit clarity."""

    def __init__(self, in_c, width, sep=False, bneck=False):
        super().__init__()
        self.ff_conv = ASPPBlock(in_c, width, sep=sep)
        self.bf_conv = ASPPBlock(in_c, width, sep=sep)
        self.rgbd_pred_layer = PredLayer(width * 8, width, sep=sep, bneck=bneck)

    def forward(self, feat, pred):
        _, _, h, w = feat.size()
        prob = torch.sigmoid(F.interpolate(pred, size=(h, w), mode='bilinear', align_corners=True))
        ff_feat = self.ff_conv(feat * prob)
        bf_feat = self.bf_conv(feat * (1 - prob))
        enhanced_feat, new_pred = self.rgbd_pred_layer(torch.cat((ff_feat, bf_feat), 1))
        return enhanced_feat, prob, new_pred


def _spatial_norm_diff(rgb, thermal):
    def _norm(x):
        offset = x.mean((-2, -1), keepdim=True)
        scale = x.var((-2, -1), unbiased=False, keepdim=True).add(1e-6).sqrt()
        return (x - offset) / scale

    return (_norm(rgb) - _norm(thermal)).abs().mean(1, keepdim=True)


class DSAMFusion(nn.Module):
    def __init__(self, channels, width, sep=False, bneck=False, joint_bridge=False):
        super().__init__()
        self.width = int(width)
        self.sep = bool(sep)
        self.bneck = bool(bneck)
        self.joint_bridge = bool(joint_bridge)
        self.prior = PredLayer(channels, width, sep=sep, bneck=False)
        self.dsam = _InnerDSAM(channels, width, sep=sep, bneck=bneck)
        self.project = nn.Conv2d(width, channels, 1, bias=False)

    def forward(self, rgb, thermal):
        if not self.joint_bridge:
            _, p1 = self.prior(thermal)
            enhanced, _, _ = self.dsam(rgb, p1)
            return self.project(enhanced), None
        joint = rgb + thermal
        _, p1 = self.prior(joint)
        enhanced, p1_prob, p2 = self.dsam(joint, p1)
        h, w = joint.shape[-2:]
        p2_prob = torch.sigmoid(F.interpolate(p2, size=(h, w), mode='bilinear', align_corners=True))
        with torch.autocast(device_type=joint.device.type, enabled=False):
            diff = _spatial_norm_diff(rgb.float(), thermal.float())
            cond = torch.cat(
                [p1_prob.float(), p2_prob.float(), (p2_prob - p1_prob).abs().float(), diff],
                dim=1,
            )
        return self.project(enhanced), cond


def make_hfp(channels, dense=False):
    hfp = HFP(channels)
    if dense:
        hfp.out = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=True),
            nn.GroupNorm(32, channels),
        )
    return hfp


class AblationModel(DetectionModel):
    def __init__(self, variant='v8m_bneck96_dense_s2', pretrained=None):
        if variant not in VARIANT_SPECS:
            raise ValueError(f'Unknown variant: {variant}')
        scale, g1_kind, dsam_cfg, fspr_dense, step2_bridge = VARIANT_SPECS[variant]
        scale_cfg = SCALE_CONFIGS[scale]
        super().__init__(scale_cfg['yaml'], ch=3, nc=3, verbose=False)
        if pretrained is not None:
            self.load(pretrained)
        self.ir_backbone = nn.ModuleList(deepcopy(list(self.model[:10])))
        self.variant = variant
        self.experiment_version = VERSION
        self.scale_name = scale
        self.g1_kind = g1_kind
        self.dsam_cfg = deepcopy(dsam_cfg)
        self.use_fspr = bool(g1_kind or dsam_cfg)
        self.fspr_dense = bool(fspr_dense)
        self.step2_bridge = bool(step2_bridge)
        self.g1 = nn.ModuleDict()
        self.fusion = nn.ModuleDict()
        self.fspr = nn.ModuleDict()
        for level, channels in zip((4, 6, 9), scale_cfg['channels']):
            if g1_kind == 'cpca':
                for branch in ('rgb', 'ir'):
                    with torch.random.fork_rng(devices=[]):
                        torch.manual_seed(830000 + level * 100 + (branch == 'ir'))
                        module = CPCA(channels)
                    self.g1[f'{branch}_{level}'] = Residual(module)
            if dsam_cfg is not None:
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(820000 + level)
                    self.fusion[str(level)] = Residual(
                        DSAMFusion(
                            channels,
                            width=dsam_cfg['w'],
                            sep=dsam_cfg['sep'],
                            bneck=dsam_cfg['bneck'],
                            joint_bridge=step2_bridge,
                        )
                    )
            if self.use_fspr:
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(740000 + level)
                    self.fspr[str(level)] = Residual(FSPR(make_hfp(channels, dense=fspr_dense)))
        self.yaml.update(ch=6, ablation=dict(version=VERSION, variant=variant, scale=scale))
        actual = sum(p.numel() for p in self.parameters())
        assert actual == EXPECTED[variant], (variant, actual, EXPECTED[variant])

    def _predict_once(self, x, profile=False, visualize=False, embed=None):
        if not hasattr(self, 'ir_backbone'):
            return super()._predict_once(x, profile, visualize, embed)
        if x.ndim != 4 or x.shape[1] != 6:
            raise ValueError('Expected [B,6,H,W] RGB+IR')
        if profile or visualize or embed:
            raise ValueError('Use the isolated audit entrypoint')
        rgb, ir = x[:, :3], x[:, 3:]
        saved = []
        conditions = {}
        for index, (rgb_layer, ir_layer) in enumerate(zip(self.model[:10], self.ir_backbone)):
            rgb, ir = rgb_layer(rgb), ir_layer(ir)
            if index in (4, 6, 9):
                if self.g1:
                    rgb_op = self.g1[f'rgb_{index}']
                    ir_op = self.g1[f'ir_{index}']
                    rgb = rgb + rgb_op.gamma * rgb_op.module(rgb)
                    ir = ir + ir_op.gamma * ir_op.module(ir)
                fused = rgb + ir
                if self.fusion:
                    op = self.fusion[str(index)]
                    extra, cond = op.module(rgb, ir)
                    if cond is not None:
                        conditions[index] = cond
                    fused = fused + op.gamma * extra
            else:
                fused = None
            saved.append(fused if index in self.save else None)
        x = saved[9]
        for layer in self.model[10:]:
            if layer.f != -1:
                x = saved[layer.f] if isinstance(layer.f, int) else [x if j == -1 else saved[j] for j in layer.f]
            x = layer(x)
            if self.fspr and layer.i in (15, 18, 21):
                level = {15: 4, 18: 6, 21: 9}[layer.i]
                op = self.fspr[str(level)]
                x = x + op.gamma * op.module(x, conditions.get(level))
            saved.append(x if layer.i in self.save else None)
        return x

    def loss(self, batch, preds=None):
        tail_fp32 = (
            self.training
            and preds is None
            and isinstance(batch, dict)
            and 'img' in batch
            and torch.is_tensor(batch['img'])
            and batch['img'].shape[0] < TAIL_BATCH_THRESHOLD
            and batch['img'].is_cuda
        )
        ctx = torch.autocast('cuda', enabled=False) if tail_fp32 else nullcontext()
        with ctx:
            return super().loss(batch, preds=preds)

    def branch_report(self):
        rgb = dict(self.model[:10].named_parameters())
        ir = dict(self.ir_backbone.named_parameters())
        assert rgb.keys() == ir.keys()
        assert all(rgb[k].data_ptr() != ir[k].data_ptr() for k in rgb)
        g1_params = sum(p.numel() for p in self.g1.parameters())
        dsam_params = sum(p.numel() for p in self.fusion.parameters())
        fspr_params = sum(p.numel() for p in self.fspr.parameters())
        return dict(
            version=VERSION,
            variant=self.variant,
            scale=self.scale_name,
            g1_kind=self.g1_kind,
            dsam_cfg=self.dsam_cfg,
            use_fspr=self.use_fspr,
            fspr_dense=self.fspr_dense,
            step2_bridge=self.step2_bridge,
            parameters=sum(p.numel() for p in self.parameters()),
            independent=True,
            initially_equal=True,
            g1_parameters=g1_params,
            dsam_parameters=dsam_params,
            fspr_parameters=fspr_params,
            fspr_condition='joint_prior_bridge' if self.step2_bridge else 'four-channel zeros',
        )
