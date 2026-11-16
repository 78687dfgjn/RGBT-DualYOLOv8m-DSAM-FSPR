"""Paired rectangular YOLO augmentation, using native affine and box filtering."""
import json
import random
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from ultralytics.data.augment import RandomPerspective, RandomHSV
from ultralytics.utils.instance import Instances

HW = (512, 640)


class PairedPerspective(RandomPerspective):
    def affine_transform(self, img, border):
        rgb, matrix, scale = super().affine_transform(img[:, :, :3], border)
        warp = cv2.warpPerspective if self.perspective else cv2.warpAffine
        ir = warp(img[:, :, 3:], matrix if self.perspective else matrix[:2],
                  dsize=self.size, borderValue=(114, 114, 114))
        return np.concatenate((rgb, ir), axis=2), matrix, scale


def letterbox(image, boxes, hw=HW):
    h, w = image.shape[:2]
    ratio = min(hw[0] / h, hw[1] / w)
    rh, rw = round(h * ratio), round(w * ratio)
    resized = cv2.resize(image, (rw, rh), interpolation=cv2.INTER_LINEAR)
    left, top = round((hw[1] - rw) / 2 - 0.1), round((hw[0] - rh) / 2 - 0.1)
    output = np.full((*hw, image.shape[2]), 114, dtype=np.uint8)
    output[top:top + rh, left:left + rw] = resized
    boxes = boxes.copy()
    boxes[:, [0, 2]] = boxes[:, [0, 2]] * (rw / w) + left
    boxes[:, [1, 3]] = boxes[:, [1, 3]] * (rh / h) + top
    return output, boxes, ((rw / w, rh / h), (left, top))


class PairedYOLODataset(Dataset):
    def __init__(self, root, split, hyp, train=False, limit=None):
        self.root, self.split, self.train = Path(root), split, train
        self.annotation_file = self.root / (split + "_thermal.json")
        raw = json.loads(self.annotation_file.read_text())
        self.images = raw["images"][:limit] if limit else raw["images"]
        self.annotations = {}
        for ann in raw["annotations"]:
            self.annotations.setdefault(ann["image_id"], []).append(ann)
        self.hyp, self.mosaic, self.rect = hyp, train, False
        self.perspective = PairedPerspective(degrees=hyp.degrees, translate=hyp.translate,
                                           scale=hyp.scale, shear=hyp.shear, perspective=hyp.perspective)
        self.hsv = RandomHSV(hyp.hsv_h, hyp.hsv_s, hyp.hsv_v)

    def __len__(self):
        return len(self.images)

    def close_mosaic(self, hyp=None):
        self.mosaic = False

    def raw(self, index):
        info = self.images[index]
        name = Path(info["file_name"]).name
        a = cv2.imread(str(self.root / self.split / "visible" / name))
        b = cv2.imread(str(self.root / self.split / "thermal" / name))
        if a is None or b is None or a.shape != b.shape:
            raise ValueError("Invalid RGB/IR pair: " + name)
        h, w = a.shape[:2]
        boxes, classes = [], []
        for ann in self.annotations.get(info["id"], []):
            if ann["category_id"] not in (0, 1, 2) or ann.get("ignore") or ann.get("iscrowd"):
                continue
            x, y, bw, bh = ann["bbox"]
            if ann["area"] <= 0 or bw < 1 or bh < 1:
                continue
            box = [max(0, x), max(0, y), min(w, x + bw), min(h, y + bh)]
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            boxes.append(box)
            classes.append(ann["category_id"])
        return np.concatenate((a, b), 2), np.array(boxes, np.float32).reshape(-1, 4), np.array(classes, np.float32).reshape(-1, 1)

    def mosaic_sample(self, index):
        h, w = HW
        yc, xc = int(random.uniform(h / 2, h * 1.5)), int(random.uniform(w / 2, w * 1.5))
        canvas = np.full((h * 2, w * 2, 6), 114, np.uint8)
        boxes_out, classes_out = [], []
        indices = [index] + random.choices(range(len(self)), k=3)
        for tile, j in enumerate(indices):
            image, boxes, classes = self.raw(j)
            image, boxes, _ = letterbox(image, boxes)
            if tile == 0:
                x1, y1, x2, y2 = max(xc-w, 0), max(yc-h, 0), xc, yc
                sx, sy = w-(x2-x1), h-(y2-y1)
            elif tile == 1:
                x1, y1, x2, y2 = xc, max(yc-h, 0), min(xc+w, 2*w), yc
                sx, sy = 0, h-(y2-y1)
            elif tile == 2:
                x1, y1, x2, y2 = max(xc-w, 0), yc, xc, min(yc+h, 2*h)
                sx, sy = w-(x2-x1), 0
            else:
                x1, y1, x2, y2 = xc, yc, min(xc+w, 2*w), min(yc+h, 2*h)
                sx, sy = 0, 0
            canvas[y1:y2, x1:x2] = image[sy:sy+y2-y1, sx:sx+x2-x1]
            boxes += np.array([x1-sx, y1-sy, x1-sx, y1-sy], np.float32)
            boxes_out.append(boxes)
            classes_out.append(classes)
        boxes = np.concatenate(boxes_out)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, 2*w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, 2*h)
        valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        return dict(img=canvas, cls=np.concatenate(classes_out)[valid],
                    instances=Instances(boxes[valid], segments=np.zeros((0, 1000, 2), np.float32), bbox_format="xyxy", normalized=False),
                    mosaic_border=(-h//2, -w//2))

    def __getitem__(self, index):
        image, boxes, classes = self.raw(index)
        orig_shape = image.shape[:2]
        ratio_pad = None
        if self.train and self.mosaic and random.random() < self.hyp.mosaic:
            labels = self.mosaic_sample(index)
        else:
            image, boxes, ratio_pad = letterbox(image, boxes)
            labels = dict(img=image, cls=classes, instances=Instances(boxes, segments=np.zeros((0, 1000, 2), np.float32), bbox_format="xyxy", normalized=False))
        if self.train:
            labels = self.perspective(labels)
            # Only RGB receives color augmentation, in OpenCV BGR space.
            visible = dict(img=np.ascontiguousarray(labels["img"][:, :, :3]))
            self.hsv(visible)
            labels["img"][:, :, :3] = visible["img"]
            if random.random() < self.hyp.fliplr:
                labels["img"] = np.fliplr(labels["img"])
                labels["instances"].fliplr(HW[1])
        image, boxes = labels["img"], labels["instances"].bboxes.copy()
        assert image.shape == (*HW, 6), image.shape
        rgb_ir = np.concatenate((image[:, :, :3][:, :, ::-1], image[:, :, 3:][:, :, ::-1]), 2)
        xywh = boxes.copy()
        xywh[:, :2] = (boxes[:, :2] + boxes[:, 2:]) / 2
        xywh[:, 2:] = boxes[:, 2:] - boxes[:, :2]
        xywh /= np.array([HW[1], HW[0], HW[1], HW[0]], np.float32)
        return dict(img=torch.from_numpy(np.ascontiguousarray(rgb_ir.transpose(2, 0, 1))),
                    cls=torch.from_numpy(labels["cls"]), bboxes=torch.from_numpy(xywh),
                    image_id=self.images[index]["id"], orig_shape=orig_shape, ratio_pad=ratio_pad)

    @staticmethod
    def collate_fn(batch):
        return dict(img=torch.stack([x["img"] for x in batch]),
                    cls=torch.cat([x["cls"] for x in batch]),
                    bboxes=torch.cat([x["bboxes"] for x in batch]),
                    batch_idx=torch.cat([torch.full((len(x["cls"]),), i) for i, x in enumerate(batch)]),
                    image_id=[x["image_id"] for x in batch], orig_shape=[x["orig_shape"] for x in batch],
                    ratio_pad=[x["ratio_pad"] for x in batch])
