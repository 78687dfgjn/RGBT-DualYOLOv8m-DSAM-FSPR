import copy, importlib.util, json
import numpy as np
import pytest
from rgbt.evaluation import evaluate, original, PROTOCOL, cap_predictions

def annotation_file(tmp_path):
    anns = []
    # Exercise every size boundary plus ignore/uncertain and absent categories.
    for i, side in enumerate([1, 8, 12, 20, 32, 40]):
        anns.append(dict(id=i + 1, image_id=1, category_id=0, bbox=[i * 45, 0, side, side],
                         area=side * side, ignore=0, iscrowd=0))
    anns += [dict(id=20, image_id=1, category_id=0, bbox=[350, 0, 25, 25], area=625, ignore=1, iscrowd=0),
             dict(id=21, image_id=1, category_id=0, bbox=[400, 0, 25, 25], area=625, ignore=0, iscrowd=0, uncertain=True)]
    data = dict(images=[dict(id=1, file_name="x.jpg", width=640, height=512)],
                categories=[dict(id=0, name="person"), dict(id=1, name="rider"), dict(id=2, name="crowd")], annotations=anns)
    path = tmp_path / "gt.json"
    path.write_text(json.dumps(data))
    return path, anns


def test_evaluator_matches_original(tmp_path):
    path, anns = annotation_file(tmp_path)
    predictions = [dict(image_id=1, category_id=0, bbox=a["bbox"], score=1 - i * .01) for i, a in enumerate(anns)]
    predictions += [dict(image_id=1, category_id=0, bbox=[351, 1, 2, 2], score=.99)]
    result = evaluate(path, predictions)
    # Load the vendor independently so this checks adapter configuration/indexing.
    spec = importlib.util.spec_from_file_location("coxnet_reference", original.__file__)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    from pycocotools.coco import COCO
    gt = COCO(str(path))
    ref = reference.COCOeval(gt, gt.loadRes(copy.deepcopy(predictions)), "bbox", ignore_uncertain=True, use_iod_for_ignore=True)
    ref.params.catIds, ref.params.imgIds = [0, 1, 2], [1]
    ref.params.maxDets = [100, 300, 1000]
    ref.params.iouThrs = np.array([.25, .5, .75])
    ref.evaluate(); ref.accumulate(); ref.summarize()
    assert result["metrics"]["mAP25_all"] == pytest.approx(ref.stats[0])
    assert result["metrics"]["mAP50_all"] == pytest.approx(ref.stats[7])
    for j, area in enumerate(ref.params.areaRngLbl):
        expected = ref.stats[7 + j]
        actual = result["metrics"]["mAP50_" + area]
        assert actual is None if expected < 0 else actual == pytest.approx(expected)
    assert result["metrics"]["AP50_rider"] is None
    assert result["metrics"]["mAP50_all"] == pytest.approx(1.0)


def test_empty_predictions(tmp_path):
    path, _ = annotation_file(tmp_path)
    result = evaluate(path, [])
    assert result["metrics"]["mAP50_all"] == 0


def test_prediction_cap():
    predictions = [dict(image_id=1, category_id=i % 3, bbox=[0, 0, 5, 5], score=i / 200) for i in range(200)]
    selected = cap_predictions(predictions)
    assert len(selected) == 100
    assert selected[-1]["score"] == .5
