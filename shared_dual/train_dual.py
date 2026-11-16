"""Use the pinned native YOLO trainer with paired data and the shared tiny evaluator."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".cache/ultralytics"))
os.environ["YOLO_AUTOINSTALL"] = "false"
os.environ["WANDB_DISABLED"] = "true"
import numpy as np
import torch
import yaml
import ultralytics
from ultralytics import settings, YOLO
from ultralytics.data import build_dataloader
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import callbacks
from ultralytics.utils.ops import non_max_suppression
import ultralytics.engine.trainer as native_trainer
from dual_model import DualYOLO
from paired_data import PairedYOLODataset, HW

sys.path.insert(0, str(BASE))
from rgbt.evaluation import evaluate, PROTOCOL
from rgbt.runtime import json_write, utcnow, RunLock

settings.update({"sync": False, "wandb": False, "mlflow": False, "tensorboard": False})
CFG = yaml.safe_load((ROOT / "config.yaml").read_text())


class TinyValidator:
    def __init__(self):
        self.metrics = SimpleNamespace(keys=["mAP25_all", "mAP50_all", "mAP75_all"])

    @torch.inference_mode()
    def __call__(self, trainer):
        model = trainer.ema.ema
        model.eval()
        predictions, ids = [], []
        for batch in trainer.test_loader:
            images = batch["img"].to(trainer.device).float() / 255
            assert images.shape[1:] == (6, *HW)
            with torch.autocast("cuda", enabled=trainer.amp):
                output = model(images)
            output = output[0] if isinstance(output, tuple) else output
            detections = non_max_suppression(output.float(), conf_thres=CFG["inference"]["conf"],
                                            iou_thres=CFG["inference"]["nms_iou"], nc=3,
                                            agnostic=False, max_det=100, max_time_img=30.0)
            assert len(detections) == len(batch["image_id"])
            for result, iid, shape, ratio_pad in zip(detections, batch["image_id"], batch["orig_shape"], batch["ratio_pad"]):
                (rx, ry), (left, top) = ratio_pad
                boxes = result[:, :4].clone()
                boxes[:, [0, 2]] = ((boxes[:, [0, 2]] - left) / rx).clamp(0, shape[1])
                boxes[:, [1, 3]] = ((boxes[:, [1, 3]] - top) / ry).clamp(0, shape[0])
                for box, row in zip(boxes.cpu().tolist(), result.cpu().tolist()):
                    x1, y1, x2, y2 = box
                    assert all(math.isfinite(x) for x in [*box, row[4]])
                    predictions.append(dict(image_id=iid, category_id=int(row[5]),
                                            bbox=[x1, y1, max(0, x2-x1), max(0, y2-y1)], score=row[4]))
                ids.append(iid)
        result = evaluate(trainer.test_loader.dataset.annotation_file, predictions, ids)
        assert result["protocol"] == PROTOCOL and result["images"] == len(trainer.test_loader.dataset)
        tag = getattr(trainer, "evaluation_tag", None)
        out = trainer.save_dir / (tag or ("epoch_%03d" % (trainer.epoch + 1)))
        json_write(out / "metrics.json", result)
        if tag is None:
            json_write(trainer.save_dir / "latest_metrics.json", dict(epoch=trainer.epoch + 1, **result))
        metrics = {key: 0.0 if value is None else value for key, value in result["metrics"].items()}
        return dict(metrics, fitness=metrics["mAP50_all"])


class DualTrainer(DetectionTrainer):
    def __init__(self, smoke=False, **kwargs):
        self.smoke = smoke
        self.samples_since_update = 0
        self.optimizer_attempts = 0
        self.successful_updates = 0
        self.amp_skips = 0
        self.last_grad_norm = None
        self.last_update_samples = 0
        self.started = utcnow()
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        super().__init__(**kwargs)
        assert os.environ.get("CUDA_VISIBLE_DEVICES") == visible, "Trainer changed GPU visibility"
        # Keep only local callbacks; no automatic integration/telemetry callbacks.
        self.callbacks = callbacks.get_default_callbacks()
        self.add_callback("on_train_batch_end", lambda t: t.write_status())
        self.add_callback("on_model_save", lambda t: t.write_status())
        self.add_callback("on_train_end", lambda t: t.write_status("smoke_completed" if t.smoke else "completed"))
        if smoke:
            self.add_callback("on_train_epoch_end", lambda t: setattr(t, "stop", True))

    def get_dataset(self):
        return dict(train=CFG["data_root"], val=CFG["data_root"], nc=3,
                    names={0: "person", 1: "rider", 2: "crowd"}, channels=6)

    def setup_model(self):
        if self.args.resume:
            checkpoint = torch.load(self.args.model, map_location="cpu", weights_only=False)
            self.model = checkpoint["ema"].float()
            self.model.requires_grad_(True)
            assert isinstance(self.model, DualYOLO)
            self.successful_updates = self.model.successful_steps
            return checkpoint
        weights = ROOT / "weights/yolov8s.pt"
        lock = json.loads((ROOT / "weights.lock.json").read_text())
        assert hashlib.sha256(weights.read_bytes()).hexdigest() == lock["sha256"]
        pretrained = YOLO(str(weights)).model
        self.model = DualYOLO(pretrained)
        report = self.model.branch_report()
        assert report["initially_equal"]
        json_write(self.save_dir / "initialization.json", report)
        return None

    def get_dataloader(self, dataset_path, batch_size=16, rank=-1, mode="train"):
        limit = (2048 if mode == "train" else 8) if self.smoke else None
        dataset = PairedYOLODataset(dataset_path, "train" if mode == "train" else "val",
                                   self.args, train=mode == "train", limit=limit)
        # Validation batching is independent of training batch/nbs.
        return build_dataloader(dataset, batch_size if mode == "train" else 4,
                                self.args.workers, shuffle=mode == "train", rank=-1)

    def get_validator(self):
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return TinyValidator()

    def preprocess_batch(self, batch):
        assert batch["img"].shape[1:] == (6, *HW)
        self.samples_since_update += batch["img"].shape[0]
        return super().preprocess_batch(batch)

    def optimizer_step(self):
        old_scale = self.scaler.get_scale()
        norms = [p.grad.detach().float().norm() / old_scale for p in self.model.parameters() if p.grad is not None]
        norm = torch.linalg.vector_norm(torch.stack(norms))
        self.last_grad_norm = float(norm) if torch.isfinite(norm) else None
        if self.successful_updates == 0 and torch.isfinite(norm):
            for branch in (self.model.model[:10], self.model.ir_backbone):
                assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().max() > 0
                           for p in branch.parameters()), "Branch has no finite nonzero gradient"
        # Preserve native unscale, clipping, scaler step/update, zero_grad AND unconditional EMA.
        super().optimizer_step()
        self.optimizer_attempts += 1
        if self.scaler.get_scale() >= old_scale:
            self.successful_updates += 1
        else:
            self.amp_skips += 1
        self.last_update_samples = self.samples_since_update
        self.samples_since_update = 0

    def write_status(self, status="running"):
        if self.ema is None or self.optimizer is None:
            return
        losses = self.loss_items.detach().float().cpu().tolist() if hasattr(self, "loss_items") else []
        if losses and not all(math.isfinite(x) for x in losses):
            raise FloatingPointError("Non-finite YOLO training loss")
        record = dict(status=status, pid=os.getpid(), started_at=self.started, updated_at=utcnow(),
                      epoch=self.epoch + 1, planned_epochs=self.epochs, patience=self.args.patience,
                      successful_steps=self.successful_updates, ema_updates=self.ema.updates,
                      optimizer_attempts_this_process=self.optimizer_attempts,
                      amp_skips_this_process=self.amp_skips, loss=losses, grad_norm=self.last_grad_norm,
                      batch=self.batch_size, nbs=self.args.nbs, accumulation=self.accumulate,
                      last_update_samples=self.last_update_samples, input_hw=list(HW),
                      lr=[g["lr"] for g in self.optimizer.param_groups],
                      momentum=[g.get("momentum") for g in self.optimizer.param_groups],
                      weight_decay=[g.get("weight_decay", 0) for g in self.optimizer.param_groups],
                      mosaic=self.train_loader.dataset.mosaic, amp=self.amp,
                      cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                      cuda_peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2)
        json_write(self.save_dir / "status.json", record)

    def final_eval(self):
        # Keep optimizer states in last.pt for actual recovery and the smoke resume test.
        if self.smoke:
            return
        if self.best.exists():
            checkpoint = torch.load(self.best, map_location="cpu", weights_only=False)
            self.ema.ema.load_state_dict(checkpoint["ema"].float().state_dict())
            self.evaluation_tag = "best_evaluation"
            result = self.validator(self)
            del self.evaluation_tag
            json_write(self.save_dir / "best_metrics.json", dict(epoch=checkpoint["epoch"] + 1, metrics=result))

    def save_model(self):
        # Plain metadata is serialized with the EMA model by the unmodified native saver.
        self.ema.ema.successful_steps = self.successful_updates
        super().save_model()

