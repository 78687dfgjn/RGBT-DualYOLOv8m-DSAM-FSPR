import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import numpy as np
import torch


def utcnow():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def json_write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        torch.backends.cuda.enable_cudnn_sdp(False)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])


class RunLock:
    def __init__(self, path):
        self.path = Path(path)

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+")
        import fcntl
        fcntl.flock(self.handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return self

    def __exit__(self, *args):
        self.handle.close()


def checkpoint_save(path, data):
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    torch.save(data, tmp)
    tmp.replace(path)


def config_hash(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()


def resume_micro_cursor(state, micro_batch, effective_batch):
    """Keep the same next sample when an OOM fallback changes micro-batch size."""
    consumed = state["next_micro"] * state["config"]["micro_batch"]
    if consumed % effective_batch or consumed % micro_batch:
        raise ValueError("Checkpoint is not at a complete effective-batch boundary")
    return consumed // micro_batch


def fp32_tree(value, device=None):
    """Cast/move loss inputs without detaching the model's AMP gradient graph."""
    if isinstance(value, torch.Tensor):
        result = value.float() if value.is_floating_point() else value
        return result.to(device) if device is not None else result
    if isinstance(value, dict):
        return {k: fp32_tree(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [fp32_tree(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(fp32_tree(v, device) for v in value)
    return value


def amp_dtype(cfg):
    name = cfg.get("amp_dtype", "float16")
    if name not in ("float16", "bfloat16"):
        raise ValueError(f"Unsupported AMP dtype: {name}")
    return getattr(torch, name)


class StepSchedule:
    """Epoch MultiStepLR plus a successful-optimizer-update warmup."""
    def __init__(self, optimizer, cfg):
        self.optimizer = optimizer
        self.warmup_steps = cfg["warmup_steps"]
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, cfg["milestones"], cfg["gamma"])

    def prepare_update(self, successful_steps):
        factor = min(1.0, (successful_steps + 1) / self.warmup_steps)
        for group, base in zip(self.optimizer.param_groups, self.scheduler.get_last_lr()):
            group["lr"] = base * factor

    def end_epoch(self):
        # Remove warmup scaling before the official epoch scheduler advances.
        for group, base in zip(self.optimizer.param_groups, self.scheduler.get_last_lr()):
            group["lr"] = base
        self.scheduler.step()

    def state_dict(self):
        return self.scheduler.state_dict()

    def load_state_dict(self, state):
        self.scheduler.load_state_dict(state)
