"""Deterministic 250-epoch LR trainer capped at epoch 80, Mosaic closed before epoch 71, tail-batch FP32 island."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parent
SHARED = ROOT / 'shared_dual'
os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / '.cache/ultralytics'))
os.environ['YOLO_AUTOINSTALL'] = 'false'
os.environ['WANDB_DISABLED'] = 'true'
sys.path.insert(0, str(SHARED))
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import yaml
import ultralytics
from ultralytics import YOLO
from ultralytics.data import build_dataloader
import ultralytics.engine.trainer as native_trainer
import train_dual
from train_dual import DualTrainer
from paired_data import PairedYOLODataset
from model import AblationModel, EXPECTED, SCALE_CONFIGS, VARIANT_SPECS, VERSION
from rgbt.runtime import json_write, RunLock, utcnow
from rgbt.results import write_results_csv

train_dual.CFG['data_root'] = os.environ.get('RGBT_DATA_ROOT') or str(Path(train_dual.CFG['data_root']).expanduser())
EPOCH_CAP = 80
MOSAIC_CLOSE_BEFORE = 71


def apply_determinism(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    assert os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8', (
        f"Expected CUBLAS_WORKSPACE_CONFIG=:4096:8, got {os.environ.get('CUBLAS_WORKSPACE_CONFIG')!r}"
    )
    assert os.environ.get('PYTHONHASHSEED') == str(seed), (
        f"Expected PYTHONHASHSEED={seed!r}, got {os.environ.get('PYTHONHASHSEED')!r}"
    )


def config(seed=42):
    cfg = yaml.safe_load((ROOT / 'config.yaml').read_text())
    train = cfg['train']
    train['seed'] = int(seed)
    assert train['nbs'] == 64 and train['patience'] == 0
    assert train['epochs'] == 250 and train['lr0'] == 0.01 and train['lrf'] == 0.01
    assert train['close_mosaic'] == 0 and train['deterministic'] is True
    assert (train['box'], train['cls'], train['dfl']) == (7.5, 0.5, 1.5)
    cfg['data_root'] = os.environ.get('RGBT_DATA_ROOT') or str(Path(cfg['data_root']).expanduser())
    return cfg


def variant_scale(variant):
    return VARIANT_SPECS[variant][0]


def scale_yaml(variant):
    return SCALE_CONFIGS[variant_scale(variant)]['yaml']


def weight_file(scale):
    records = json.loads((ROOT / 'weights.lock.json').read_text())
    record = records[scale]
    path = ROOT / record['path']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == record['sha256'], (scale, path)
    return path


def run_name(variant, seed=42):
    return f'dual_{variant}_det80_cm71_seed{int(seed)}'


def cap_already_reached(checkpoint, smoke=False, epoch_cap=EPOCH_CAP):
    return not smoke and checkpoint['epoch'] + 1 >= epoch_cap


class AblationTrainer(DualTrainer):
    def __init__(self, variant, seed=42, smoke=False, limit=None, val_limit=None, epoch_cap=EPOCH_CAP, **kwargs):
        self.variant = variant
        self.scale = variant_scale(variant)
        self.seed = int(seed)
        self.sample_limit = limit
        self.val_limit = val_limit
        self.epoch_cap = int(epoch_cap)
        self.schedule_epochs = 250
        self.mosaic_boundary = 2 if smoke else MOSAIC_CLOSE_BEFORE
        self.tail_fp32_batches = 0
        self.amp_b16_batches = 0
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        super().__init__(smoke=smoke, **kwargs)
        assert self.args.nbs == 64 and self.args.epochs == self.schedule_epochs and self.args.patience == 0
        assert self.args.batch == 16 and self.args.close_mosaic == 0 and self.args.deterministic is True
        self.csv = self.save_dir / 'native_results.csv'
        self.add_callback('on_train_start', self.verify_restored_state)
        self.add_callback('on_train_epoch_start', self.apply_mosaic_policy)
        if not smoke:
            self.add_callback('on_train_epoch_end', self.apply_stop_cap)
        self.add_callback('on_fit_epoch_end', lambda trainer: trainer.export_epoch())

    def apply_mosaic_policy(self, trainer):
        boundary = self.mosaic_boundary
        if boundary is not None and trainer.epoch + 1 >= boundary and trainer.train_loader.dataset.mosaic:
            trainer._close_dataloader_mosaic()
            trainer.train_loader.reset()
            json_write(
                trainer.save_dir / 'mosaic_closed.json',
                dict(
                    before_epoch=trainer.epoch + 1,
                    scheduled_epoch=boundary,
                    smoke=self.smoke,
                    updated_at=utcnow(),
                ),
            )

    def apply_stop_cap(self, trainer):
        if trainer.epoch + 1 >= self.epoch_cap:
            trainer.stop = True

    def preprocess_batch(self, batch):
        bsz = batch['img'].shape[0]
        if bsz < 16:
            self.tail_fp32_batches += 1
        else:
            self.amp_b16_batches += 1
        return super().preprocess_batch(batch)

    def save_model(self):
        self.ema.ema.tail_fp32_batches = self.tail_fp32_batches
        self.ema.ema.amp_b16_batches = self.amp_b16_batches
        super().save_model()

    def verify_restored_state(self, trainer):
        assert trainer.stopper.patience == float('inf')
        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False
        assert torch.are_deterministic_algorithms_enabled()
        assert torch.is_deterministic_algorithms_warn_only_enabled()
        if not trainer.args.resume:
            assert trainer.train_loader.dataset.mosaic is True
        elif trainer.start_epoch + 1 >= self.mosaic_boundary and (trainer.save_dir / 'mosaic_closed.json').exists():
            # Already past boundary in checkpoint; keep mosaic disabled before resumed epoch executes.
            if trainer.train_loader.dataset.mosaic:
                trainer._close_dataloader_mosaic()
                trainer.train_loader.reset()
        record = dict(
            ok=True,
            start_epoch=trainer.start_epoch + 1,
            epoch_cap=self.epoch_cap,
            schedule_epochs=self.schedule_epochs,
            mosaic_close_before_epoch=self.mosaic_boundary,
            batch=trainer.batch_size,
            nbs=64,
            seed=self.seed,
            patience=0,
            early_stopping_disabled=True,
            close_mosaic=0,
            mosaic=trainer.train_loader.dataset.mosaic,
            cudnn_deterministic=torch.backends.cudnn.deterministic,
            cudnn_benchmark=torch.backends.cudnn.benchmark,
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
            deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
            cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
            pythonhashseed=os.environ.get('PYTHONHASHSEED'),
            tail_fp32_policy='batch<16',
            successful_steps=trainer.successful_updates,
            ema_updates=trainer.ema.updates,
            optimizer_states=len(trainer.optimizer.state),
            resume=bool(trainer.args.resume),
            updated_at=utcnow(),
        )
        if trainer.args.resume:
            assert record['optimizer_states'] > 0 and record['ema_updates'] > 0
        json_write(
            self.save_dir / ('resume_setup_verified.json' if trainer.args.resume else 'initial_setup_verified.json'),
            record,
        )

    def setup_model(self):
        if self.args.resume:
            checkpoint = torch.load(self.args.model, map_location='cpu', weights_only=False)
            self.model = checkpoint['ema'].float()
            assert isinstance(self.model, AblationModel)
            assert self.model.variant == self.variant and self.model.experiment_version == VERSION
            assert checkpoint['train_args']['batch'] == 16 and checkpoint['train_args']['nbs'] == 64
            self.model.requires_grad_(True)
            if hasattr(self.model, 'criterion'):
                del self.model.criterion
            self.successful_updates = self.model.successful_steps
            self.tail_fp32_batches = getattr(self.model, 'tail_fp32_batches', 0)
            self.amp_b16_batches = getattr(self.model, 'amp_b16_batches', 0)
            return checkpoint
        torch.manual_seed(self.seed)
        self.model = AblationModel(self.variant, YOLO(str(weight_file(self.scale))).model)
        report = self.model.branch_report()
        assert report['initially_equal']
        json_write(self.save_dir / 'initialization.json', report)
        return None

    def get_dataloader(self, dataset_path, batch_size=16, rank=-1, mode='train'):
        if mode == 'train':
            limit = self.sample_limit if self.sample_limit is not None else (512 if self.smoke else None)
        else:
            limit = self.val_limit if self.val_limit is not None else (8 if self.smoke else None)
        dataset = PairedYOLODataset(
            dataset_path,
            'train' if mode == 'train' else 'val',
            self.args,
            train=mode == 'train',
            limit=limit,
        )
        return build_dataloader(
            dataset,
            batch_size if mode == 'train' else 4,
            self.args.workers,
            shuffle=mode == 'train',
            rank=-1,
        )

    def write_status(self, status='running'):
        super().write_status(status)
        path = self.save_dir / 'status.json'
        if path.exists():
            record = json.loads(path.read_text())
            record.update(
                variant=self.variant,
                scale=self.scale,
                version=VERSION,
                seed=self.seed,
                stop_epoch_cap=self.epoch_cap,
                schedule_epochs=self.schedule_epochs,
                early_stopping_disabled=True,
                close_mosaic=0,
                mosaic_close_before_epoch=self.mosaic_boundary,
                tail_fp32_policy='batch<16',
                tail_fp32_batches=self.tail_fp32_batches,
                amp_b16_batches=self.amp_b16_batches,
                cudnn_deterministic=torch.backends.cudnn.deterministic,
                cudnn_benchmark=torch.backends.cudnn.benchmark,
                loss_spec='CIoU + BCE + DFL; native YOLOv8 v8DetectionLoss',
                cuda_peak_reserved_mib=torch.cuda.max_memory_reserved() / 1024**2,
            )
            json_write(path, record)

    def export_epoch(self):
        path = self.save_dir / f'epoch_{self.epoch + 1:03d}' / 'metrics.json'
        if not path.exists():
            return
        result = json.loads(path.read_text())
        history = self.save_dir / 'history.jsonl'
        rows = (
            {row['epoch']: row for row in [json.loads(line) for line in history.read_text().splitlines() if line]}
            if history.exists()
            else {}
        )
        rows[self.epoch + 1] = dict(
            epoch=self.epoch + 1,
            metrics=result['metrics'],
            images=result['images'],
            successful_steps=self.successful_updates,
            ema_updates=self.ema.updates,
            tail_fp32_batches=self.tail_fp32_batches,
            amp_b16_batches=self.amp_b16_batches,
            updated_at=utcnow(),
            learning_rates=[group['lr'] for group in self.optimizer.param_groups],
            mosaic=self.train_loader.dataset.mosaic,
            train_loss_last_batch=self.loss_items.detach().float().cpu().tolist(),
            grad_norm_last_batch=self.last_grad_norm,
        )
        tmp = history.with_suffix('.tmp')
        tmp.write_text(''.join(json.dumps(rows[key], allow_nan=False) + '\n' for key in sorted(rows)))
        tmp.replace(history)
        write_results_csv(self.save_dir, list(rows.values()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', choices=list(EXPECTED), default='v8m_bneck96_dense_s2')
    parser.add_argument('--batch', type=int, choices=[16], default=16)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    assert ultralytics.__version__ == '8.3.150'
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    apply_determinism(args.seed)
    cfg = config(args.seed)
    parent = ROOT / 'smoke_runs' / f'batch{args.batch}' if args.smoke else ROOT / 'runs'
    run = parent / run_name(args.variant, args.seed)
    limit = None
    val_limit = None
    epoch_cap = EPOCH_CAP
    overrides = dict(
        cfg['train'],
        batch=16,
        seed=args.seed,
        model=scale_yaml(args.variant),
        data=str(SHARED / 'data.yaml'),
        project=str(run.parent),
        name=run.name,
        device='',
    )
    if args.resume:
        overrides['resume'] = str(run / 'weights/last.pt')
    native_trainer.check_amp = lambda model: True
    start = 0
    reached = False
    if args.resume:
        checkpoint = torch.load(run / 'weights/last.pt', map_location='cpu', weights_only=False)
        start = checkpoint['ema'].successful_steps
        reached = cap_already_reached(checkpoint, args.smoke, epoch_cap=epoch_cap)
        del checkpoint
    with RunLock(run / 'training.lock'):
        trainer = AblationTrainer(
            args.variant,
            seed=args.seed,
            smoke=args.smoke,
            limit=limit,
            val_limit=val_limit,
            epoch_cap=epoch_cap,
            overrides=overrides,
        )
        if not (run / 'experiment.json').exists():
            json_write(
                run / 'experiment.json',
                dict(
                    version=VERSION,
                    variant=args.variant,
                    scale=variant_scale(args.variant),
                    seed=args.seed,
                    parameters=EXPECTED[args.variant],
                    config=cfg,
                    actual_batch=16,
                    nbs=64,
                    stop_epoch_cap=epoch_cap,
                    schedule_epochs=cfg['train']['epochs'],
                    patience=0,
                    close_mosaic=0,
                    mosaic_close_before_epoch=2 if args.smoke else MOSAIC_CLOSE_BEFORE,
                    deterministic=True,
                    cudnn_benchmark=False,
                    deterministic_algorithms_warn_only=True,
                    tail_fp32_policy='batch<16',
                    loss='native CIoU+BCE+DFL',
                    initialization='independent official COCO',
                    created_at=utcnow(),
                ),
            )
        try:
            if reached:
                trainer._setup_train(world_size=1)
                trainer.epoch = trainer.start_epoch - 1
                trainer.apply_mosaic_policy(trainer)
                trainer.final_eval()
                trainer.write_status('completed')
            else:
                trainer.train()
        except BaseException as error:
            json_write(run / 'failure.json', dict(error=repr(error), updated_at=utcnow()))
            trainer.write_status('failed')
            raise
        finally:
            for name in ('train_loader', 'test_loader'):
                shutdown = getattr(getattr(getattr(trainer, name, None), 'iterator', None), '_shutdown_workers', None)
                if shutdown:
                    shutdown()
    if args.smoke:
        checkpoint = torch.load(run / 'weights/last.pt', map_location='cpu', weights_only=False)
        state = json.loads((run / 'status.json').read_text())
        assert checkpoint['optimizer'] is not None and checkpoint['train_args']['nbs'] == 64
        assert checkpoint['train_args']['batch'] == 16 and state['successful_steps'] >= start + 2
        assert checkpoint['ema'].successful_steps == state['successful_steps']
        assert checkpoint['updates'] == state['ema_updates'] and isinstance(checkpoint['ema'], AblationModel)
        if args.resume:
            assert state['mosaic'] is False and (run / 'mosaic_closed.json').exists()
        else:
            assert state['mosaic'] is True
        json_write(
            run / ('resume_verified.json' if args.resume else 'smoke_verified.json'),
            dict(
                ok=True,
                batch=16,
                nbs=64,
                seed=args.seed,
                initial_successful_steps=start,
                successful_steps=state['successful_steps'],
                ema_updates=state['ema_updates'],
                tail_fp32_batches=state['tail_fp32_batches'],
                amp_b16_batches=state['amp_b16_batches'],
                validation_images=8,
                mosaic=state['mosaic'],
                peak_reserved_mib=state['cuda_peak_reserved_mib'],
                completed_at=utcnow(),
            ),
        )


if __name__ == '__main__':
    main()
