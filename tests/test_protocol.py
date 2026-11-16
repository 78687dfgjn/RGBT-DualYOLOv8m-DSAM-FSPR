import inspect
from types import SimpleNamespace

import pytest
from ultralytics.utils.torch_utils import EarlyStopping

from train import AblationTrainer, EPOCH_CAP, MOSAIC_CLOSE_BEFORE, cap_already_reached, config, run_name


def test_det80_training_schedule_and_disabled_early_stop():
    cfg = config()['train']
    assert EPOCH_CAP == 80 and MOSAIC_CLOSE_BEFORE == 71
    assert cfg['epochs'] == 250 and cfg['nbs'] == 64 and cfg['patience'] == 0
    stopper = EarlyStopping(patience=0)
    assert not any(stopper(epoch, 0.5 if epoch == 1 else 0.1) for epoch in range(1, 251))
    assert cfg['lr0'] == 0.01 and cfg['lrf'] == 0.01 and cfg['warmup_epochs'] == 3 and not cfg['cos_lr']
    assert cfg['close_mosaic'] == 0 and cfg['deterministic'] is True
    assert (cfg['box'], cfg['cls'], cfg['dfl']) == (7.5, 0.5, 1.5)


def test_mosaic_policy_closes_before_epoch_71_with_worker_reset():
    calls = []

    class FakeLoader:
        def __init__(self):
            self.dataset = SimpleNamespace(mosaic=True)

        def reset(self):
            calls.append('reset')

    def close_mosaic():
        calls.append('close')
        trainer.train_loader.dataset.mosaic = False

    holder = SimpleNamespace(mosaic_boundary=71, smoke=False)
    trainer = SimpleNamespace(
        epoch=69,
        train_loader=FakeLoader(),
        _close_dataloader_mosaic=close_mosaic,
        save_dir=None,
    )
    AblationTrainer.apply_mosaic_policy(holder, trainer)
    assert calls == [] and trainer.train_loader.dataset.mosaic is True

    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmp:
        trainer.save_dir = Path(tmp)
        trainer.epoch = 70  # epoch + 1 == 71
        AblationTrainer.apply_mosaic_policy(holder, trainer)
        assert calls == ['close', 'reset']
        assert trainer.train_loader.dataset.mosaic is False
        assert (trainer.save_dir / 'mosaic_closed.json').exists()


def test_run_name_format():
        assert run_name('v8m_bneck96_dense_s2', 42) == 'dual_v8m_bneck96_dense_s2_det80_cm71_seed42'


@pytest.mark.parametrize('completed', [1, 79, 80, 81])
def test_cap_stops_at_80_without_an_extra_resume_epoch(completed):
    holder = SimpleNamespace(epoch_cap=80)
    trainer = SimpleNamespace(epoch=completed - 1, stop=False)
    AblationTrainer.apply_stop_cap(holder, trainer)
    assert trainer.stop == (completed >= 80)
    assert cap_already_reached({'epoch': completed - 1}) == (completed >= 80)
    assert not cap_already_reached({'epoch': completed - 1}, smoke=True)


def test_cap_callback_runs_before_native_checkpoint_save():
    from ultralytics.engine.trainer import BaseTrainer
    source = inspect.getsource(BaseTrainer._do_train)
    boundary = source.index('self.run_callbacks("on_train_epoch_end")')
    assert boundary < source.index('self.save_model()', boundary)
