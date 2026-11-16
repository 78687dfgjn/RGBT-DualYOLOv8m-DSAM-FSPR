import gc
import subprocess
import sys
from pathlib import Path

import torch
from ultralytics import YOLO

from model import AblationModel, EXPECTED
from train import weight_file

VARIANT = "v8m_bneck96_dense_s2"


def test_parameter_count_and_zero_gamma_add_equivalence():
    torch.manual_seed(42)
    source = YOLO(str(weight_file("m"))).model
    torch.manual_seed(42)
    baseline = AblationModel("v8m_add", source).eval()
    torch.manual_seed(42)
    candidate = AblationModel(VARIANT, source).eval()
    params = candidate.branch_report()["parameters"]
    assert params == EXPECTED[VARIANT] == 51980833
    assert params < 55_000_000
    assert candidate.branch_report()["initially_equal"]
    for key, value in baseline.model.state_dict().items():
        torch.testing.assert_close(value, candidate.model.state_dict()[key], rtol=0, atol=0)
    images = torch.rand(2, 6, 64, 96)
    with torch.inference_mode():
        torch.testing.assert_close(candidate(images)[0], baseline(images)[0], rtol=0, atol=0)
    del source, baseline, candidate
    gc.collect()


def test_bneck96_dense_topology():
    model = AblationModel(VARIANT)
    module = model.fusion["4"].module
    assert module.dsam.ff_conv.aspp2[0].out_channels == 96
    assert module.dsam.ff_conv.aspp2[0].groups == 1
    assert module.dsam.rgbd_pred_layer.enlayer[0].kernel_size == (1, 1)
    assert module.dsam.rgbd_pred_layer.enlayer[0].in_channels == 768
    assert module.dsam.rgbd_pred_layer.enlayer[0].out_channels == 96
    assert isinstance(model.fspr["4"].module.hfp.out[0], torch.nn.Conv2d)
    assert model.fspr["4"].module.hfp.out[0].groups == 1


def test_zero_gamma_and_step2_condition_gradients():
    model = AblationModel(VARIANT).train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    images = torch.rand(2, 6, 64, 96)
    groups = (model.g1, model.fusion, model.fspr)
    for _ in range(3):
        optimizer.zero_grad()
        sum(output.square().mean() for output in model(images)).backward()
        for group in groups:
            for residual in group.values():
                assert residual.gamma.grad is not None and torch.isfinite(residual.gamma.grad)
        optimizer.step()
    optimizer.zero_grad()
    sum(output.square().mean() for output in model(images)).backward()
    for group in groups:
        for residual in group.values():
            assert any(
                parameter.grad is not None and torch.isfinite(parameter.grad).all()
                and parameter.grad.abs().max() > 0
                for parameter in residual.module.parameters()
            )
    for residual in model.fusion.values():
        assert residual.module.dsam.rgbd_pred_layer.outlayer[0].weight.grad is not None
    for residual in model.fspr.values():
        assert residual.module.gate[0].weight.grad is not None


def test_step2_passes_4ch_condition_to_fspr():
    model = AblationModel(VARIANT).train()
    report = model.branch_report()
    assert report["step2_bridge"] is True
    assert report["fspr_condition"] == "joint_prior_bridge"
    captured = []
    orig = model.fspr["4"].module.forward

    def hook(feat, cond=None):
        captured.append(cond.shape if cond is not None else None)
        return orig(feat, cond)

    model.fspr["4"].module.forward = hook
    with torch.no_grad():
        model(torch.rand(2, 6, 64, 96))
    assert captured == [(2, 4, 8, 12)]


def test_checkpoint_round_trip_in_fresh_process(tmp_path):
    checkpoint = tmp_path / f"{VARIANT}.pt"
    torch.save(dict(ema=AblationModel(VARIANT), train_args=dict(nbs=64, batch=16)), checkpoint)
    code = (
        "import sys,torch; c=torch.load(sys.argv[1],map_location='cpu',weights_only=False); "
        "assert c['ema'].variant==sys.argv[2]; "
        "assert c['train_args']['nbs']==64 and c['train_args']['batch']==16"
    )
    subprocess.run(
        [sys.executable, "-c", code, str(checkpoint), VARIANT],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )
