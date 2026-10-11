from pathlib import Path

import numpy as np
import pytest

from fpvdd.eval_report import present_classes, render_report
from fpvdd.evaluation import (
    ImageGT,
    box_iou,
    coco_metrics,
    confusion_matrix,
    gt_size_counts,
    load_ground_truth,
    precision_recall,
    size_bucket,
    to_coco,
)
from fpvdd.images import write_image
from fpvdd.labels import CLASS_NAMES, YoloBox, write_label_file

NAMES = list(CLASS_NAMES)
M, B = NAMES.index("multirotor"), NAMES.index("bird")
BG = len(NAMES)


def _gt(*boxes: list[float], w: int = 640, h: int = 512) -> ImageGT:
    return ImageGT(w, h, np.array(boxes, dtype=float).reshape(-1, 5))


def _pred(*boxes: list[float]) -> np.ndarray:
    return np.array(boxes, dtype=float).reshape(-1, 6)


def test_load_ground_truth_converts_to_pixels(tmp_path: Path) -> None:
    img = tmp_path / "images" / "v_f000000.jpg"
    img.parent.mkdir(parents=True)
    write_image(img, np.zeros((100, 200, 3), dtype=np.uint8))
    write_label_file(tmp_path / "labels" / "v_f000000.txt", [YoloBox(B, 0.5, 0.5, 0.1, 0.2)])
    gt = load_ground_truth([img])[img]
    assert (gt.width, gt.height) == (200, 100)
    np.testing.assert_allclose(gt.boxes, [[B, 90, 40, 110, 60]])


@pytest.mark.parametrize(
    ("area", "bucket"),
    [
        (0, "small"),
        (32**2 - 1, "small"),
        (32**2, "medium"),
        (96**2 - 1, "medium"),
        (96**2, "large"),
    ],
)
def test_size_bucket(area: float, bucket: str) -> None:
    assert size_bucket(area) == bucket


def test_gt_size_counts() -> None:
    gt = {Path("a"): _gt([M, 0, 0, 10, 10], [B, 0, 0, 50, 50], [B, 0, 0, 200, 200])}
    counts = gt_size_counts(gt, NAMES)
    assert counts["multirotor"] == {"small": 1, "medium": 0, "large": 0}
    assert counts["bird"] == {"small": 0, "medium": 1, "large": 1}


def test_box_iou() -> None:
    iou = box_iou(
        np.array([[0, 0, 10, 10]]), np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]])
    )
    np.testing.assert_allclose(iou, [[1.0, 1 / 3, 0.0]])


GT = {
    Path("a"): _gt([M, 10, 10, 30, 30], [B, 100, 100, 160, 160]),  # small drone, medium bird
    Path("b"): _gt([M, 200, 200, 400, 400]),  # large drone
    Path("c"): _gt(),  # background frame
}


def test_coco_perfect_predictions() -> None:
    preds = {
        p: np.column_stack([g.boxes[:, :1], np.full(len(g.boxes), 0.9), g.boxes[:, 1:]])
        for p, g in GT.items()
    }
    coco = coco_metrics(*to_coco(GT, preds, NAMES), NAMES)
    assert coco["all"]["overall"]["ap"] == pytest.approx(1.0)
    assert coco["small"]["multirotor"]["ap50"] == pytest.approx(1.0)
    assert coco["medium"]["bird"]["ap"] == pytest.approx(1.0)
    assert coco["small"]["bird"]["ap"] is None  # no small birds in the ground truth
    assert coco["all"]["vtol"]["ap"] is None


def test_coco_output_is_json_serialisable_with_float32_predictions() -> None:
    import json

    preds = {Path("a"): _pred([M, 0.9, 10, 10, 30, 30]).astype(np.float32)}  # like the model
    coco_gt, coco_dt = to_coco(GT, preds, NAMES)
    json.dumps(coco_gt)
    json.dumps(coco_dt)
    json.dumps(coco_metrics(coco_gt, coco_dt, NAMES))


def test_coco_without_predictions() -> None:
    coco = coco_metrics(*to_coco(GT, {}, NAMES), NAMES)
    assert coco["all"]["overall"]["ap"] == 0.0


def test_coco_wrong_class_scores_zero_for_both() -> None:
    preds = {Path("a"): _pred([B, 0.9, 10, 10, 30, 30])}  # drone predicted as bird
    gt = {Path("a"): _gt([M, 10, 10, 30, 30])}
    coco = coco_metrics(*to_coco(gt, preds, NAMES), NAMES)
    assert coco["all"]["multirotor"]["ap"] == 0.0


def test_confusion_matrix_counts() -> None:
    preds = {
        Path("a"): _pred(
            [M, 0.9, 10, 10, 30, 30],  # TP drone
            [M, 0.8, 100, 100, 160, 160],  # bird detected as multirotor
            [M, 0.1, 300, 300, 320, 320],  # below operating conf: ignored
        ),
        Path("c"): _pred([B, 0.7, 0, 0, 20, 20]),  # FP on a background frame
    }  # image b: large drone missed
    matrix = confusion_matrix(GT, preds, len(NAMES), conf=0.25, iou=0.5)
    assert matrix[M, M] == 1
    assert matrix[M, B] == 1
    assert matrix[B, BG] == 1
    assert matrix[BG, M] == 1
    assert matrix.sum() == 4

    pr = precision_recall(matrix, NAMES)
    assert pr["multirotor"] == {"tp": 1, "fp": 1, "fn": 1, "precision": 0.5, "recall": 0.5}
    assert pr["bird"]["recall"] == 0.0
    assert pr["vtol"]["precision"] is None


def test_confusion_matching_is_one_to_one() -> None:
    gt = {Path("a"): _gt([M, 0, 0, 10, 10])}
    preds = {Path("a"): _pred([M, 0.9, 0, 0, 10, 10], [M, 0.8, 0, 0, 10, 10])}  # duplicate
    matrix = confusion_matrix(gt, preds, len(NAMES))
    assert matrix[M, M] == 1
    assert matrix[M, BG] == 1


def test_report_mentions_bird_confusion_and_skips_absent_classes() -> None:
    preds = {Path("a"): _pred([M, 0.9, 100, 100, 160, 160])}
    matrix = confusion_matrix(GT, preds, len(NAMES))
    coco_gt, coco_dt = to_coco(GT, preds, NAMES)
    results = {
        "run": "r",
        "weights": "w.pt",
        "split": "test",
        "images": 3,
        "limit": None,
        "imgsz": 1280,
        "names": NAMES,
        "dataset": {"sha256": "abc" * 20},
        "dataset_matches_training": True,
        "leakage_check": "passed",
        "ultralytics": {"map50": 0.5, "map50_95": 0.3, "per_class": {}},
        "coco": coco_metrics(coco_gt, coco_dt, NAMES),
        "gt_counts": gt_size_counts(GT, NAMES),
        "operating_point": {
            "conf": 0.25,
            "iou": 0.5,
            "confusion": matrix.tolist(),
            "per_class": precision_recall(matrix, NAMES),
        },
    }
    assert present_classes(results) == ["multirotor", "bird"]
    report = render_report(results)
    assert "1 birds detected as multirotor" in report
    assert "vtol" not in report
    assert "(same as training)" in report
