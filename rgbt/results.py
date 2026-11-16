"""Export epoch history as an Excel-readable CSV without training dependencies."""
import csv
import math
import os
from pathlib import Path
import tempfile

PRIMARY_METRICS = ("mAP50_all", "mAP50_tiny", "mAP50_small", "mAP50_tiny1",
                   "mAP50_tiny2", "mAP50_tiny3", "AP50_person", "AP50_rider",
                   "AP50_crowd", "AR50_all", "AR50_tiny", "AR50_small")
METADATA = ("epoch", "successful_steps", "updated_at", "images", "predictions",
            "train_loss_last_batch", "grad_norm_last_batch", "ema_updates")


def write_results_csv(run_dir, records):
    """One row per epoch, latest duplicate wins; AP/AR use explicit *_pct columns."""
    by_epoch = {record["epoch"]: record for record in records}
    ordered = [by_epoch[epoch] for epoch in sorted(by_epoch)]
    metrics = {key for record in ordered for key in record.get("metrics", {})}
    metric_keys = [key for key in PRIMARY_METRICS if key in metrics]
    metric_keys += sorted(metrics - set(metric_keys))
    lr_count = max((len(record.get("learning_rates", [])) for record in ordered), default=0)
    lr_keys = ["lr_group_" + str(index) for index in range(lr_count)]

    def column(key):
        return key + "_pct" if key.startswith(("mAP", "AP", "AR")) else key

    fieldnames = list(METADATA) + [column(key) for key in metric_keys] + lr_keys
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    output = run_dir / "results.csv"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8-sig", newline="",
                                         prefix="results.", suffix=".tmp", dir=run_dir,
                                         delete=False) as handle:
            temporary = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for record in ordered:
                row = {key: record.get(key) for key in METADATA}
                for key in metric_keys:
                    value = record.get("metrics", {}).get(key)
                    if value is not None and not math.isfinite(value):
                        value = None
                    if value is not None and column(key) != key:
                        value = round(value * 100, 6)
                    row[column(key)] = value
                row.update(zip(lr_keys, record.get("learning_rates", [])))
                writer.writerow(row)
        os.replace(temporary, output)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return output
