import json
import random
from pathlib import Path
import sys
import cv2
import numpy as np
import pytest
import torch
from ultralytics.cfg import get_cfg

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dual_model import DualYOLO
from paired_data import PairedYOLODataset, letterbox, HW


@pytest.fixture
def paired_root(tmp_path):
    image = np.random.default_rng(42).integers(0, 256, (81, 123, 3), dtype=np.uint8)
    for modal in ("visible", "thermal"):
        folder = tmp_path / "train" / modal
        folder.mkdir(parents=True)
        assert cv2.imwrite(str(folder / "a.png"), image)
    raw = dict(images=[dict(id=1, file_name="a.png", width=999, height=999)],
               annotations=[dict(image_id=1, category_id=0, bbox=[20, 20, 12, 15], area=180),
                            dict(image_id=1, category_id=3, bbox=[5, 5, 10, 10], area=100)])
    (tmp_path / "train_thermal.json").write_text(json.dumps(raw))
    return tmp_path


def test_letterbox_roundtrip():
    image = np.zeros((81, 123, 6), np.uint8)
    original = np.array([[20, 20, 32, 35]], np.float32)
    output, boxes, ((rx, ry), (left, top)) = letterbox(image, original)
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - left) / rx
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - top) / ry
    np.testing.assert_allclose(boxes, original, atol=1e-5)
    assert output.shape == (512, 640, 6)


def test_rectangular_paired_mosaic_and_close(paired_root):
    hyp = get_cfg(overrides=dict(hsv_h=0., hsv_s=0., hsv_v=0.))
    dataset = PairedYOLODataset(paired_root, "train", hyp, train=True)
    for seed in range(12):
        random.seed(seed)
        x = dataset[0]
        assert x["img"].shape == (6, 512, 640)
        assert torch.equal(x["img"][:3], x["img"][3:])
        assert torch.all((x["bboxes"] >= 0) & (x["bboxes"] <= 1))
        assert not torch.any(x["cls"] == 3)
    dataset.close_mosaic()
    assert dataset.mosaic is False
    assert dataset[0]["img"].shape == (6, *HW)


def test_hsv_changes_only_visible_branch(paired_root):
    hyp = get_cfg()
    dataset = PairedYOLODataset(paired_root, "train", hyp, train=True)
    random.seed(11); np.random.seed(11)
    colored = dataset[0]["img"]
    hyp.hsv_h = hyp.hsv_s = hyp.hsv_v = 0.
    control = PairedYOLODataset(paired_root, "train", hyp, train=True)
    random.seed(11); np.random.seed(11)
    plain = control[0]["img"]
    assert torch.equal(colored[3:], plain[3:])
    assert not torch.equal(colored[:3], plain[:3])


def test_backbones_add_and_gradients():
    torch.set_num_threads(4)
    model = DualYOLO()
    report = model.branch_report()
    assert report["independent"] and report["initially_equal"]
    features, handles = {}, []
    for i in (4, 6, 9):
        handles.append(model.model[i].register_forward_hook(lambda m, x, y, i=i: features.__setitem__(("rgb", i), y)))
        handles.append(model.ir_backbone[i].register_forward_hook(lambda m, x, y, i=i: features.__setitem__(("ir", i), y)))
    for head_index, level in ((11, 6), (14, 4), (20, 9)):
        def check(module, inputs, level=level):
            torch.testing.assert_close(inputs[0][1], features[("rgb", level)] + features[("ir", level)])
        handles.append(model.model[head_index].register_forward_pre_hook(check))
    model.train()
    result = model(torch.rand(2, 6, 64, 96))
    sum(x.square().mean() for x in result).backward()
    for branch in (model.model[:10], model.ir_backbone):
        assert any(p.grad is not None and p.grad.abs().max() > 0 for p in branch.parameters())
    for handle in handles:
        handle.remove()




@pytest.mark.parametrize("overflow", [False, True])
def test_optimizer_and_ema_match_native_even_on_amp_skip(overflow):
    from types import SimpleNamespace
    from ultralytics.engine.trainer import BaseTrainer
    from train_dual import DualTrainer
    class FakeScaler:
        def __init__(self): self.value = 1.
        def get_scale(self): return self.value
        def unscale_(self, optimizer): pass
        def step(self, optimizer):
            if not overflow: optimizer.step()
        def update(self):
            if overflow: self.value /= 2
    class EMA:
        def __init__(self): self.updates = 0
        def update(self, model): self.updates += 1
    def make():
        model = torch.nn.Linear(2, 1)
        for p in model.parameters():
            p.data.fill_(1.)
            p.grad = torch.ones_like(p)
        return SimpleNamespace(model=model, optimizer=torch.optim.SGD(model.parameters(), lr=.01),
                               scaler=FakeScaler(), ema=EMA(), successful_updates=1,
                               optimizer_attempts=0, amp_skips=0, samples_since_update=64)
    native, custom = make(), make()
    BaseTrainer.optimizer_step(native)
    # super() requires an actual subclass instance, without invoking GPU trainer initialization.
    wrapper = object.__new__(DualTrainer)
    wrapper.__dict__.update(custom.__dict__)
    DualTrainer.optimizer_step(wrapper)
    assert wrapper.ema.updates == native.ema.updates == 1
    assert wrapper.successful_updates == (1 if overflow else 2)
    for a, b in zip(wrapper.model.parameters(), native.model.parameters()):
        torch.testing.assert_close(a, b)


def test_shared_tiny_evaluator_against_reference(tmp_path):
    import importlib.util
    base = ROOT.parent
    sys.path.insert(0, str(base))
    spec = importlib.util.spec_from_file_location("shared_baseline_tests", base / "tests/evaluator_reference.py")
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    reference.test_evaluator_matches_original(tmp_path)
    reference.test_empty_predictions(tmp_path)
    reference.test_prediction_cap()
