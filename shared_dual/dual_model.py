"""Dual independent YOLOv8 backbones; Add at P3/P4/P5 before the standard head.

Uses Ultralytics 8.3.150 (AGPL-3.0).
"""
from copy import deepcopy
import torch
from torch import nn
from ultralytics.nn.tasks import DetectionModel


class DualYOLO(DetectionModel):
    def __init__(self, pretrained=None):
        super().__init__("yolov8s.yaml", ch=3, nc=3, verbose=False)
        if pretrained is not None:
            self.load(pretrained)
        self.ir_backbone = nn.ModuleList(deepcopy(list(self.model[:10])))
        self.yaml["ch"] = 6

    def _predict_once(self, x, profile=False, visualize=False, embed=None):
        # DetectionModel builds strides with a 3-channel dummy before this branch exists.
        if not hasattr(self, "ir_backbone"):
            return super()._predict_once(x, profile, visualize, embed)
        if x.ndim != 4 or x.shape[1] != 6:
            raise ValueError("DualYOLO requires RGB+IR [B,6,H,W]")
        if profile or visualize or embed:
            raise ValueError("Profiling/embedding is not part of this training protocol")
        rgb, ir = x[:, :3], x[:, 3:]
        saved = []
        for i, (visible_layer, thermal_layer) in enumerate(zip(self.model[:10], self.ir_backbone)):
            assert visible_layer.f == thermal_layer.f == -1
            rgb, ir = visible_layer(rgb), thermal_layer(ir)
            fused = rgb + ir if i in (4, 6, 9) else None
            saved.append(fused if i in self.save else None)
        x = saved[9]
        for layer in self.model[10:]:
            if layer.f != -1:
                x = saved[layer.f] if isinstance(layer.f, int) else [x if j == -1 else saved[j] for j in layer.f]
            x = layer(x)
            saved.append(x if layer.i in self.save else None)
        return x

    def branch_report(self):
        rgb = dict(self.model[:10].named_parameters())
        ir = dict(self.ir_backbone.named_parameters())
        assert rgb.keys() == ir.keys()
        assert all(rgb[k].data_ptr() != ir[k].data_ptr() for k in rgb)
        return dict(independent=True, initially_equal=all(torch.equal(rgb[k], ir[k]) for k in rgb),
                    fusion_levels=["P3", "P4", "P5"], fusion="Add",
                    parameters=sum(p.numel() for p in self.parameters()),
                    rgb_parameters=sum(p.numel() for p in rgb.values()),
                    ir_parameters=sum(p.numel() for p in ir.values()))
