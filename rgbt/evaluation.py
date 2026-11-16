import copy
import importlib.util
from pathlib import Path
import numpy as np
from pycocotools.coco import COCO

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "vendor/COXNet/mmdet/datasets/evaluation/coco/cocoeval.py"
spec = importlib.util.spec_from_file_location("coxnet_original_cocoeval", PATH)
original = importlib.util.module_from_spec(spec)
spec.loader.exec_module(original)

PROTOCOL = dict(eval_standard="tiny", cat_ids=[0, 1, 2], iou_thresholds=[0.25, 0.5, 0.75],
                max_dets=[100, 300, 1000], prediction_topk=100, ignore_uncertain=True,
                use_iod_for_ignore=True, use_iod_for_crowd=False, use_ignore_attr=True,
                area_ranges=[[1, 1e10], [1, 400], [1, 64], [64, 144], [144, 400], [400, 1024], [1024, 1e10]])


def cap_predictions(predictions, topk=100):
    by_image = {}
    for prediction in predictions:
        if prediction["category_id"] not in (0, 1, 2):
            raise ValueError("prediction category outside baseline classes")
        by_image.setdefault(prediction["image_id"], []).append(prediction)
    return [p for group in by_image.values() for p in sorted(group, key=lambda p: -p["score"])[:topk]]


def evaluate(annotation_file, predictions, image_ids=None):
    gt = COCO(str(annotation_file))
    predictions = cap_predictions(predictions)
    if predictions:
        dt = gt.loadRes(predictions)
    else:
        dt = COCO()
        dt.dataset = {"images": copy.deepcopy(gt.dataset["images"]),
                      "categories": copy.deepcopy(gt.dataset["categories"]), "annotations": []}
        dt.createIndex()
    original.Params.EVAL_STRANDARD = "tiny"
    evaluator = original.COCOeval(gt, dt, "bbox", ignore_uncertain=True, use_iod_for_ignore=True)
    evaluator.params.catIds = [0, 1, 2]
    evaluator.params.imgIds = sorted(image_ids if image_ids is not None else gt.getImgIds())
    evaluator.params.maxDets = [100, 300, 1000]
    evaluator.params.iouThrs = np.array([0.25, 0.5, 0.75])
    assert evaluator.params.areaRng == PROTOCOL["area_ranges"]
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    areas = evaluator.params.areaRngLbl
    metrics = {}
    for i, threshold in enumerate([25, 50, 75]):
        for j, area in enumerate(areas):
            value = float(evaluator.stats[i * len(areas) + j])
            metrics[f"mAP{threshold}_{area}"] = value if value >= 0 else None
    precision = evaluator.eval["precision"]
    for k, name in enumerate(["person", "rider", "crowd"]):
        values = precision[1, :, k, 0, -1]
        metrics[f"AP50_{name}"] = float(values[values >= 0].mean()) if (values >= 0).any() else None
    return dict(metrics=metrics, protocol=PROTOCOL, images=len(evaluator.params.imgIds),
                predictions=len(predictions))
