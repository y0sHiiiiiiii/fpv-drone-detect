"""Detection metrics beyond Ultralytics' built-ins: COCO size breakdown and confusion at a
fixed operating point.

Boxes are pixel ``xyxy`` arrays. Ground truth per image is an ``(N, 5)`` array
``[class, x1, y1, x2, y2]``; predictions are ``(M, 6)`` ``[class, conf, x1, y1, x2, y2]``.
"""

import contextlib
import io
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from fpvdd.images import read_image
from fpvdd.labels import parse_label_file
from fpvdd.runinfo import label_path_for_image

# COCO area ranges in pixels² of the original image
AREA_RANGES: dict[str, tuple[float, float]] = {
    "all": (0.0, 1e10),
    "small": (0.0, 32.0**2),
    "medium": (32.0**2, 96.0**2),
    "large": (96.0**2, 1e10),
}


@dataclass(frozen=True)
class ImageGT:
    width: int
    height: int
    boxes: np.ndarray  # (N, 5) [class, x1, y1, x2, y2] in pixels


def load_ground_truth(images: Sequence[Path]) -> dict[Path, ImageGT]:
    """Read image sizes and YOLO labels; boxes are converted to pixel xyxy."""
    gt = {}
    for image in images:
        h, w = read_image(image).shape[:2]
        rows = [
            [
                b.class_id,
                (b.x_center - b.width / 2) * w,
                (b.y_center - b.height / 2) * h,
                (b.x_center + b.width / 2) * w,
                (b.y_center + b.height / 2) * h,
            ]
            for b in parse_label_file(label_path_for_image(image))
        ]
        gt[image] = ImageGT(w, h, np.array(rows, dtype=float).reshape(-1, 5))
    return gt


def size_bucket(area: float) -> str:
    """COCO size bucket (small / medium / large) of a box area in pixels²."""
    if area < AREA_RANGES["small"][1]:
        return "small"
    if area < AREA_RANGES["medium"][1]:
        return "medium"
    return "large"


def gt_size_counts(gt: Mapping[Path, ImageGT], names: Sequence[str]) -> dict[str, dict[str, int]]:
    """Number of ground-truth boxes per class and size bucket."""
    counts = {name: dict.fromkeys(("small", "medium", "large"), 0) for name in names}
    for item in gt.values():
        for cls, x1, y1, x2, y2 in item.boxes:
            counts[names[int(cls)]][size_bucket((x2 - x1) * (y2 - y1))] += 1
    return counts


def to_coco(
    gt: Mapping[Path, ImageGT], preds: Mapping[Path, np.ndarray], names: Sequence[str]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """COCO ground-truth dict and detection list. Category ids are class ids + 1."""
    images, annotations, detections = [], [], []
    for image_id, (path, item) in enumerate(sorted(gt.items()), start=1):
        images.append(
            {"id": image_id, "file_name": path.name, "width": item.width, "height": item.height}
        )
        # plain floats: numpy scalars (float32 from the model) are not JSON serialisable
        for cls, x1, y1, x2, y2 in item.boxes.tolist():
            w, h = x2 - x1, y2 - y1
            annotations.append(
                {
                    "id": len(annotations) + 1,
                    "image_id": image_id,
                    "category_id": int(cls) + 1,
                    "bbox": [x1, y1, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                }
            )
        for cls, conf, x1, y1, x2, y2 in preds.get(path, np.zeros((0, 6))).tolist():
            detections.append(
                {
                    "image_id": image_id,
                    "category_id": int(cls) + 1,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": conf,
                }
            )
    categories = [{"id": i + 1, "name": n} for i, n in enumerate(names)]
    return {"images": images, "annotations": annotations, "categories": categories}, detections


def coco_metrics(
    coco_gt: dict[str, Any], detections: list[dict[str, Any]], names: Sequence[str]
) -> dict[str, Any]:
    """AP@0.5:0.95, AP@0.5 and AR@100 (keys ``ap``, ``ap50``, ``ar``), overall and per class,
    for each COCO area range. Values are None where there is no ground truth to evaluate.
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    with contextlib.redirect_stdout(io.StringIO()):  # pycocotools prints progress
        gt = COCO()
        gt.dataset = coco_gt
        gt.createIndex()
        dt = gt.loadRes(detections) if detections else COCO()
        if not detections:
            dt.dataset = {"images": coco_gt["images"], "annotations": [], "categories": []}
            dt.createIndex()
        ev = COCOeval(gt, dt, iouType="bbox")
        ev.params.areaRng = list(AREA_RANGES.values())
        ev.params.areaRngLbl = list(AREA_RANGES)
        ev.evaluate()
        ev.accumulate()

    precision = ev.eval["precision"]  # [T, R, K, A, M]
    recall = ev.eval["recall"]  # [T, K, A, M]
    iou_50 = int(np.argmin(np.abs(ev.params.iouThrs - 0.5)))

    def mean_valid(values: np.ndarray) -> float | None:
        valid = values[values > -1]
        return float(valid.mean()) if valid.size else None

    def stats(k: slice | int, a: int) -> dict[str, float | None]:
        return {
            "ap": mean_valid(precision[:, :, k, a, -1]),
            "ap50": mean_valid(precision[iou_50, :, k, a, -1]),
            "ar": mean_valid(recall[:, k, a, -1]),  # AR@100, averaged over IoU 0.5:0.95
        }

    result: dict[str, Any] = {}
    for a, area in enumerate(AREA_RANGES):
        result[area] = {"overall": stats(slice(None), a)}
        for k, name in enumerate(names):
            result[area][name] = stats(k, a)
    return result


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU matrix between ``(N, 4)`` and ``(M, 4)`` xyxy boxes."""
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(br - tl, 0, None), axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return inter / (area_a[:, None] + area_b[None, :] - inter)


def confusion_matrix(
    gt: Mapping[Path, ImageGT],
    preds: Mapping[Path, np.ndarray],
    num_classes: int,
    conf: float = 0.25,
    iou: float = 0.5,
) -> np.ndarray:
    """``(nc + 1, nc + 1)`` matrix indexed ``[predicted, true]``; index ``nc`` = background.

    Predictions with ``score >= conf`` are matched one-to-one to ground truth, class-agnostic,
    greedily by IoU (highest first, ``>= iou``). A match counts in ``[pred_cls, true_cls]``,
    unmatched predictions in ``[pred_cls, background]`` (false positives) and unmatched ground
    truth in ``[background, true_cls]`` (missed). Same orientation as Ultralytics.
    """
    bg = num_classes
    matrix = np.zeros((num_classes + 1, num_classes + 1), dtype=int)
    for path, item in gt.items():
        p = preds.get(path, np.zeros((0, 6)))
        p = p[p[:, 1] >= conf] if len(p) else p
        ious = box_iou(p[:, 2:6], item.boxes[:, 1:5]) if len(p) and len(item.boxes) else None
        matched_p: set[int] = set()
        matched_g: set[int] = set()
        if ious is not None:
            for pi, gi in zip(
                *np.unravel_index(np.argsort(-ious, axis=None), ious.shape), strict=True
            ):
                if ious[pi, gi] < iou:
                    break
                if pi in matched_p or gi in matched_g:
                    continue
                matched_p.add(pi)
                matched_g.add(gi)
                matrix[int(p[pi, 0]), int(item.boxes[gi, 0])] += 1
        for pi in range(len(p)):
            if pi not in matched_p:
                matrix[int(p[pi, 0]), bg] += 1
        for gi in range(len(item.boxes)):
            if gi not in matched_g:
                matrix[bg, int(item.boxes[gi, 0])] += 1
    return matrix


def precision_recall(matrix: np.ndarray, names: Sequence[str]) -> dict[str, dict[str, Any]]:
    """Per-class TP/FP/FN, precision and recall from a confusion matrix.

    A prediction of the right box with the wrong class counts as a false positive for the
    predicted class and as a miss for the true class.
    """
    out = {}
    for c, name in enumerate(names):
        tp = int(matrix[c, c])
        fp = int(matrix[c, :].sum() - tp)
        fn = int(matrix[:, c].sum() - tp)
        out[name] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
        }
    return out
