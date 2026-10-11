"""Evaluate infer_video.py output against per-frame ground truth of the source videos.

Reads every <name>_detections.json in --pred, loads the ground truth of the matching video and
writes to --output (default <pred>/video_eval/): metrics.json, per_video.csv and report.md with
- precision / recall / F1 per frame, overall, per clip group and per class
- false positives by kind (wrong class, duplicate, localization, other object, background)
- MOTA, IDF1, ID switches, fragmentation, false tracks and tracks per object, for videos
  whose ground truth identifies objects

Datasets: halmstad (labels for every frame; object ids only where a clip has at most one
object of our classes per frame).

Examples:
    python scripts/evaluate_video.py --pred runs/infer/baseline_test
    python scripts/evaluate_video.py --pred runs/infer/baseline_test --conf 0.25
"""

import argparse
import csv
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from fpvdd.eval_report import render_video_report
from fpvdd.halmstad import DATASET, find_clips, load_labels, video_ground_truth
from fpvdd.labels import CLASS_NAMES
from fpvdd.log import setup_logging
from fpvdd.paths import CONFIGS_DIR, RAW_DIR, SPLITS_DIR
from fpvdd.sources import load_source_config
from fpvdd.video_eval import add_counts, evaluate_sequence, preds_from_records, summarize
from fpvdd.yolo_env import resolve_path

log = logging.getLogger("evaluate_video")

SUFFIX = "_detections.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--pred", type=Path, required=True, help="output folder of infer_video.py")
    parser.add_argument("--dataset", choices=[DATASET], default=DATASET)
    parser.add_argument("--raw", type=Path, default=RAW_DIR / DATASET, help="dataset videos")
    parser.add_argument("--config", type=Path, default=CONFIGS_DIR / "sources" / f"{DATASET}.yaml")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU for a match")
    parser.add_argument("--conf", type=float, default=0.0, help="ignore predictions below this")
    parser.add_argument("--manifest", type=Path, default=SPLITS_DIR / "split_manifest.csv")
    parser.add_argument("--output", type=Path, default=None, help="default: <pred>/video_eval")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def read_manifest(path: Path) -> dict[str, str]:
    """Video id -> split, from split_dataset.py's manifest (empty if missing)."""
    if not path.exists():
        log.warning("no split manifest at %s; cannot check for training videos", path)
        return {}
    with path.open(encoding="utf-8") as fh:
        return {row["video_id"]: row["split"] for row in csv.DictReader(fh)}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(args.verbose)
    pred_dir = resolve_path(args.pred)
    out_dir = resolve_path(args.output) if args.output else pred_dir / "video_eval"
    files = sorted(pred_dir.glob(f"*{SUFFIX}"))
    if not files:
        log.error("no *%s files in %s", SUFFIX, pred_dir)
        return 1
    source = load_source_config(resolve_path(args.config))
    clips = {c.video.stem: c for c in find_clips(resolve_path(args.raw), ("V", "IR"))}
    splits = read_manifest(resolve_path(args.manifest))

    videos: list[dict[str, Any]] = []
    with logging_redirect_tqdm():
        for path in tqdm(files, desc="videos", unit="video"):
            stem = path.name[: -len(SUFFIX)]
            clip = clips.get(stem)
            if clip is None:
                log.error("no %s video with labels named %s below %s", DATASET, stem, args.raw)
                return 1
            labels = load_labels(clip)
            records = json.loads(path.read_text(encoding="utf-8"))
            preds = preds_from_records(records, CLASS_NAMES, args.conf)
            if preds and max(preds) >= len(labels):
                log.error("%s: frame %d has no label (%d labelled)", stem, max(preds), len(labels))
                return 1
            if len(preds) < len(labels):
                log.info("%s: %d of %d frames have predictions", stem, len(preds), len(labels))
            gt, has_ids = video_ground_truth(labels, source.map_class)
            counts = evaluate_sequence(gt, preds, CLASS_NAMES, args.iou, identities=has_ids)
            videos.append(
                {
                    "video": stem,
                    "group": clip.source_class,
                    "split": splits.get(clip.video_id),
                    "counts": counts,
                    "metrics": summarize(add_counts([counts])),
                }
            )

    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for v in videos:
        by_group[v["group"]].append(v)

    def block(items: list[dict[str, Any]]) -> dict[str, Any]:
        total = add_counts(v["counts"] for v in items)
        return {"videos": len(items), "counts": total, "metrics": summarize(total)}

    run_info = pred_dir / "infer_run.json"
    inference = (
        json.loads(run_info.read_text(encoding="utf-8")).get("settings")
        if run_info.exists()
        else None
    )
    leaked = sorted(v["video"] for v in videos if v["split"] in ("train", "val"))
    results = {
        "settings": {
            "pred": str(pred_dir),
            "dataset": args.dataset,
            "iou": args.iou,
            "conf": args.conf,
        },
        "inference": inference,
        "leaked": leaked,
        "overall": block(videos),
        "groups": {g: block(items) for g, items in sorted(by_group.items())},
        "videos": videos,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    with (out_dir / "per_video.csv").open("w", newline="", encoding="utf-8") as fh:
        fields = ["video", "group", "split", "frames", "gt", "tp", "fp", "fn", "precision"]
        fields += ["recall", "mota", "idf1", "idsw", "frag", "tracks", "false_tracks"]
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for v in videos:
            m = v["metrics"]
            writer.writerow({**v, **m, **(m["identity"] or {})})
    (out_dir / "report.md").write_text(render_video_report(results), encoding="utf-8")

    if leaked:
        log.warning("%d videos come from the training or validation split: %s", len(leaked), leaked)
    m = results["overall"]["metrics"]
    ident = m["identity"] or {}
    log.info(
        "P %.3f  R %.3f  MOTA %s  IDF1 %s  ID switches %s | report: %s",
        m["precision"] or 0,
        m["recall"] or 0,
        f"{ident['mota']:.3f}" if ident.get("mota") is not None else "–",
        f"{ident['idf1']:.3f}" if ident.get("idf1") is not None else "–",
        ident.get("idsw", "–"),
        out_dir / "report.md",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
