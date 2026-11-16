import torch
from ultralytics.cfg import get_cfg
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.loss import v8DetectionLoss

from model import AblationModel, EXPECTED


def test_all_variants_use_native_ciou_bce_dfl_loss():
    for variant in EXPECTED:
        model = AblationModel(variant)
        model.args = get_cfg()
        criterion = model.init_criterion()
        assert type(criterion) is v8DetectionLoss
        assert (model.args.box, model.args.cls, model.args.dfl) == (7.5, 0.5, 1.5)


def test_native_loss_returns_finite_gradients():
    model = AblationModel('v8m_add')
    model.args = get_cfg()
    criterion = model.init_criterion()
    features = [
        torch.randn(2, 67, h, w, requires_grad=True)
        for h, w in ((8, 12), (4, 6), (2, 3))
    ]
    batch = dict(
        batch_idx=torch.tensor([0.0, 1.0]),
        cls=torch.tensor([[0.0], [1.0]]),
        bboxes=torch.tensor([[0.5, 0.5, 0.15, 0.2], [0.7, 0.6, 0.2, 0.2]]),
    )
    total, parts = criterion(features, batch)
    assert torch.isfinite(total).all() and torch.isfinite(parts).all()
    total.sum().backward()
    assert all(feature.grad is not None and torch.isfinite(feature.grad).all() for feature in features)


def test_tail_batch_fp32_island_triggers_only_for_cuda_batches_under_16(monkeypatch):
    from contextlib import nullcontext
    model = AblationModel('v8m_add').train()
    autocast_calls = []

    def fake_autocast(*args, **kwargs):
        autocast_calls.append((args, kwargs))
        return nullcontext()

    def fake_super_loss(self, batch, preds=None):
        return torch.tensor(1.0), torch.tensor([1.0, 0.0, 0.0])

    monkeypatch.setattr(torch, 'autocast', fake_autocast)
    monkeypatch.setattr(DetectionModel, 'loss', fake_super_loss)

    class FakeCudaTensor(torch.Tensor):
        @property
        def is_cuda(self):
            return True

    def make_batch(bs, cuda=True):
        img = torch.zeros(bs, 6, 32, 32)
        if cuda:
            img = img.as_subclass(FakeCudaTensor)
        return {'img': img}

    model.loss(make_batch(16, cuda=True))
    assert autocast_calls == []
    model.loss(make_batch(4, cuda=False))
    assert autocast_calls == []
    model.loss(make_batch(4, cuda=True), preds=['dummy'])
    assert autocast_calls == []
    model.eval()
    model.loss(make_batch(4, cuda=True))
    assert autocast_calls == []
    model.train()
    model.loss(make_batch(4, cuda=True))
    assert autocast_calls == [(('cuda',), {'enabled': False})]
