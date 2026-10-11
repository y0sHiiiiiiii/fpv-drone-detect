"""Evaluate trained weights on a split: mAP, per-class P/R, confusion matrix, size breakdown.

Writes to runs/eval/<run>_<split>/: metrics.json, report.md, confusion.csv, COCO ground truth
and predictions (for error analysis), plus Ultralytics' own plots in ultralytics/.

Examples:
    python scripts/evaluate.py --weights runs/detect/baseline/weights/best.pt --split test
    python scripts/evaluate.py --weights runs/detect/baseline/weights/best.pt --limit 200
"""

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from fpvdd.eval_report import render_report
from fpvdd.evaluation import (
    coco_metrics,
    confusion_matrix,
    gt_size_counts,
    load_ground_truth,
    precision_recall,
    to_coco,
)
from fpvdd.labels import CLASS_NAMES
from fpvdd.log import setup_logging
from fpvdd.paths import CONFIGS_DIR
from fpvdd.runinfo import dataset_fingerprint, read_split_list
from fpvdd.split import LeakageError, check_no_leakage
from fpvdd.yolo_env import (
    RUNS_DIR,
    prepare_ultralytics,
    resolve_dataset,
    resolve_path,
    sample_split_n,
)

log = logging.getLogger("evaluate")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--weights", type=Path, required=True, help="trained .pt file")
    parser.add_argument("--split", choices=["test", "val", "train"], default="test")
    parser.add_argument("--data", type=Path, default=CONFIGS_DIR / "dataset.yaml")
    parser.add_argument("--imgsz", type=int, default=None, help="default: the training imgsz")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--conf", type=float, default=0.001, help="confidence floor for mAP")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    parser.add_argument(
        "--op-conf", type=float, default=0.25, help="operating point for confusion matrix and P/R"
    )
    parser.add_argument(
        "--match-iou", type=float, default=0.5, help="IoU for matching at the operating point"
    )
    parser.add_argument("--limit", type=int, default=0, help="evaluate a random subset of N images")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", default=None, help="output folder name below runs/eval/")
    parser.add_argument("--device", default=None, help="e.g. 0 or cpu (default: auto)")
    parser.add_argument(
        "--workers", type=int, default=2, help="dataloader workers (each ~1.3 GB RAM on Windows)"
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def ultralytics_summary(metrics: Any, names: list[str]) -> dict[str, Any]:
    """Overall and per-class numbers from Ultralytics' DetMetrics."""
    box = metrics.box
    per_class = {}
    for i, cls in enumerate(box.ap_class_index):
        per_class[names[int(cls)]] = {
            "precision": float(box.p[i]),
            "recall": float(box.r[i]),
            "ap50": float(box.ap50[i]),
            "ap50_95": float(box.ap[i]),
        }
    return {"map50": float(box.map50), "map50_95": float(box.map), "per_class": per_class}


def predict(model: Any, images: list[Path], args: argparse.Namespace, imgsz: int):
    """Predictions per image as ``(M, 6)`` arrays ``[class, conf, x1, y1, x2, y2]``."""
    preds: dict[Path, np.ndarray] = {}
    stream = model.predict(
        source=[str(p) for p in images],
        stream=True,
        conf=args.conf,
        iou=args.iou,
        imgsz=imgsz,
        batch=args.batch,
        max_det=300,
        device=args.device,
        verbose=False,
    )
    for result in tqdm(stream, total=len(images), desc="predict", unit="img"):
        b = result.boxes
        arr = np.column_stack([b.cls.cpu().numpy(), b.conf.cpu().numpy(), b.xyxy.cpu().numpy()])
        preds[Path(result.path)] = arr.reshape(-1, 6)
    return preds


def training_run_info(weights: Path) -> dict[str, Any] | None:
    info = weights.parent.parent / "fpvdd_run.json"
    return json.loads(info.read_text(encoding="utf-8")) if info.exists() else None


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(args.verbose)
    weights = resolve_path(args.weights)
    if not weights.exists():
        log.error("weights not found: %s", weights)
        return 1
    try:
        # own resolved copy, never the one a running training reads
        data_yaml, splits = resolve_dataset(
            resolve_path(args.data), required=(args.split,), suffix=".eval"
        )
    except (OSError, ValueError, KeyError) as exc:
        log.error("%s", exc)
        return 1

    full_fingerprint = dataset_fingerprint({s: p for s, p in splits.items() if p.exists()})
    try:
        check_no_leakage({s: read_split_list(p) for s, p in splits.items() if p.exists()})
        leakage = "passed"
    except LeakageError as exc:
        log.error("split leakage: %s", exc)
        return 1

    run_name = weights.parent.parent.name if weights.parent.name == "weights" else weights.stem
    name = args.name or f"{run_name}_{args.split}" + (f"_n{args.limit}" if args.limit else "")
    out_dir = RUNS_DIR / "eval" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.limit:
        subset = sample_split_n(
            splits[args.split], args.limit, args.seed, out_dir / f"{args.split}_subset.txt"
        )
        data_yaml, splits = resolve_dataset(
            resolve_path(args.data),
            required=(args.split,),
            replace={args.split: subset},
            suffix=f".eval_{name}",
        )
    images = read_split_list(splits[args.split])
    log.info("%d %s images -> %s", len(images), args.split, out_dir)

    with logging_redirect_tqdm():
        gt = load_ground_truth(tqdm(images, desc="labels", unit="img"))

    if not prepare_ultralytics():
        return 1
    from ultralytics import YOLO

    model = YOLO(str(weights))
    names = [model.names[i] for i in sorted(model.names)]
    if tuple(names) != CLASS_NAMES:
        log.warning("model classes %s differ from configs/dataset.yaml %s", names, CLASS_NAMES)
    imgsz = args.imgsz or int((model.ckpt or {}).get("train_args", {}).get("imgsz", 640))

    log.info("Ultralytics validation (imgsz %d)", imgsz)
    metrics = model.val(
        data=str(data_yaml),
        split=args.split,
        imgsz=imgsz,
        batch=args.batch,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        project=str(out_dir),
        name="ultralytics",
        exist_ok=True,
        plots=True,
        verbose=False,
        workers=args.workers,
    )
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # release validation buffers before the predict pass
    with logging_redirect_tqdm():
        preds = predict(model, images, args, imgsz)
    missing = set(gt) - set(preds)
    if missing:
        log.error("%d images got no prediction result, e.g. %s", len(missing), next(iter(missing)))
        return 1

    coco_gt, coco_dt = to_coco(gt, preds, names)
    matrix = confusion_matrix(gt, preds, len(names), conf=args.op_conf, iou=args.match_iou)
    train_info = training_run_info(weights)
    results: dict[str, Any] = {
        "run": run_name,
        "weights": str(weights),
        "split": args.split,
        "images": len(images),
        "limit": args.limit or None,
        "imgsz": imgsz,
        "names": names,
        "dataset": full_fingerprint,
        "dataset_matches_training": (
            None
            if train_info is None
            else train_info.get("dataset", {}).get("sha256") == full_fingerprint["sha256"]
        ),
        "training_finished": None if train_info is None else "finished" in train_info,
        "leakage_check": leakage,
        "ultralytics": ultralytics_summary(metrics, names),
        "coco": coco_metrics(coco_gt, coco_dt, names),
        "gt_counts": gt_size_counts(gt, names),
        "operating_point": {
            "conf": args.op_conf,
            "iou": args.match_iou,
            "confusion": matrix.tolist(),
            "per_class": precision_recall(matrix, names),
        },
    }

    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (out_dir / "coco_gt.json").write_text(json.dumps(coco_gt), encoding="utf-8")
    (out_dir / "coco_predictions.json").write_text(json.dumps(coco_dt), encoding="utf-8")
    with (out_dir / "confusion.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        labels = [*names, "background"]
        writer.writerow(["predicted \\ true", *labels])
        for label, row in zip(labels, matrix.tolist(), strict=True):
            writer.writerow([label, *row])
    report = render_report(results)
    (out_dir / "report.md").write_text(report, encoding="utf-8")

    if results["dataset_matches_training"] is False:
        log.warning("labels or splits changed since this model was trained")
    if results["training_finished"] is False:
        log.warning("the training run has not finished; these are intermediate weights")
    log.info(
        "mAP50 %.4f  mAP50-95 %.4f (Ultralytics) | report: %s",
        results["ultralytics"]["map50"],
        results["ultralytics"]["map50_95"],
        out_dir / "report.md",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
