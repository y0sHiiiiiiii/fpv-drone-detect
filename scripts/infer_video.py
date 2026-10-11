"""Detect and track objects in recorded videos; write an annotated video and detections JSON.

For each input video, writes to --output:
    <name>_annotated.mp4    boxes with class, confidence and track id
    <name>_detections.json  per frame: {frame, timestamp_s, detections: [{track_id, class,
                            confidence, bbox_xyxy}]}
    <name>_tracks.csv       one row per track: lifetime, majority class, mean confidence
plus infer_run.json with the settings and a summary per video.

Detection runs on full frames, or with --sahi on overlapping slices (SAHI) for small objects.
Either way the boxes go through the same Ultralytics tracker (ByteTrack or BoT-SORT).

Examples:
    python scripts/infer_video.py --weights runs/detect/baseline/weights/best.pt \\
        --source data/raw/clip.mp4 --tracker bytetrack.yaml --output out/
    python scripts/infer_video.py --weights best.pt --source videos/ --sahi --slice 640
    python scripts/infer_video.py --weights best.pt --source clip.mp4 --tracker none
"""

import argparse
import csv
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from fpvdd.draw import draw_detections
from fpvdd.inference import (
    detections_from_array,
    detections_from_tracks,
    frame_record,
    summarize_tracks,
    suppress_contained,
)
from fpvdd.labels import CLASS_NAMES
from fpvdd.log import setup_logging
from fpvdd.paths import VIDEO_EXTS
from fpvdd.runinfo import git_state
from fpvdd.video import iter_frames, open_writer, probe, transcode_h264
from fpvdd.yolo_env import RUNS_DIR, prepare_ultralytics, resolve_path

log = logging.getLogger("infer_video")

# Ultralytics' track mode default: low enough for ByteTrack's second association (0.1).
TRACK_CONF = 0.1
DETECT_CONF = 0.25


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--weights", type=Path, required=True, help="trained .pt file")
    parser.add_argument(
        "--source", type=Path, nargs="+", required=True, help="video files and/or folders"
    )
    parser.add_argument("--output", type=Path, default=RUNS_DIR / "infer", help="output folder")
    parser.add_argument(
        "--tracker",
        default="bytetrack.yaml",
        help="bytetrack.yaml, botsort.yaml, a tracker YAML path, or 'none' (detection only)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=None,
        help=f"detection confidence floor (default {TRACK_CONF} with tracker, else {DETECT_CONF})",
    )
    parser.add_argument(
        "--iou", type=float, default=0.5, help="NMS IoU (0.5 removes most duplicate boxes)"
    )
    parser.add_argument(
        "--ios",
        type=float,
        default=0.8,
        help="drop boxes whose area lies to this share inside a higher-confidence box (0: off)",
    )
    parser.add_argument("--imgsz", type=int, default=None, help="default: the training imgsz")
    parser.add_argument("--max-det", type=int, default=100)
    parser.add_argument("--device", default=None, help="e.g. 0 or cpu (default: auto)")
    parser.add_argument("--sahi", action="store_true", help="sliced inference with SAHI")
    parser.add_argument("--slice", type=int, default=640, help="SAHI slice size in pixels")
    parser.add_argument("--overlap", type=float, default=0.2, help="SAHI slice overlap ratio")
    parser.add_argument(
        "--slice-imgsz",
        type=int,
        default=None,
        help="model input size per slice (default: --imgsz, i.e. slices are upscaled like the "
        "training images)",
    )
    parser.add_argument(
        "--max-frames", type=int, default=0, help="stop after N frames per video (0: all)"
    )
    parser.add_argument("--no-video", action="store_true", help="skip the annotated video")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def find_videos(sources: list[Path]) -> list[Path]:
    """Video files as given, plus all videos below given folders (sorted per folder)."""
    videos = []
    for source in sources:
        if source.is_file():
            videos.append(source)
        else:
            videos += sorted(p for p in source.rglob("*") if p.suffix.lower() in VIDEO_EXTS)
    return videos


def load_tracker_config(tracker: str) -> Any:
    """Tracker settings from an Ultralytics tracker YAML (bundled name or path)."""
    from ultralytics.utils import YAML, IterableSimpleNamespace
    from ultralytics.utils.checks import check_yaml

    cfg = IterableSimpleNamespace(**YAML.load(check_yaml(tracker)))
    if getattr(cfg, "with_reid", False) and getattr(cfg, "model", "auto") == "auto":
        raise ValueError(
            "with_reid and model: auto take features from inside Ultralytics' predictor, which "
            "this script does not use; set with_reid: False or name a ReID model"
        )
    return cfg


def new_tracker(cfg: Any) -> Any:
    from ultralytics.trackers.track import TRACKER_MAP

    if cfg.tracker_type not in TRACKER_MAP:
        raise ValueError(
            f"unknown tracker_type {cfg.tracker_type!r}, use one of {list(TRACKER_MAP)}"
        )
    return TRACKER_MAP[cfg.tracker_type](args=cfg)


class Detector:
    """Full-frame YOLO or SAHI sliced detection; returns ``(N, 6)`` arrays in pixels."""

    def __init__(self, model: Any, args: argparse.Namespace, imgsz: int) -> None:
        self.model = model
        self.args = args
        self.imgsz = imgsz
        self.sahi_model = None
        if args.sahi:
            from sahi.models.ultralytics import UltralyticsDetectionModel

            device = args.device
            if device is not None and str(device).isdigit():
                device = f"cuda:{device}"  # SAHI wants torch device names
            self.sahi_model = UltralyticsDetectionModel(
                model=model,
                confidence_threshold=args.conf,
                device=device,
                image_size=args.slice_imgsz or imgsz,
            )

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        dets = self._detect(frame)
        return suppress_contained(dets, self.args.ios) if self.args.ios > 0 else dets

    def _detect(self, frame: np.ndarray) -> np.ndarray:
        if self.sahi_model is None:
            result = self.model.predict(
                frame,
                conf=self.args.conf,
                iou=self.args.iou,
                imgsz=self.imgsz,
                max_det=self.args.max_det,
                device=self.args.device,
                verbose=False,
            )[0]
            return result.boxes.data.cpu().numpy().reshape(-1, 6)

        from sahi.predict import get_sliced_prediction

        pred = get_sliced_prediction(
            np.ascontiguousarray(frame[:, :, ::-1]),  # SAHI expects RGB
            self.sahi_model,
            slice_height=self.args.slice,
            slice_width=self.args.slice,
            overlap_height_ratio=self.args.overlap,
            overlap_width_ratio=self.args.overlap,
            perform_standard_pred=True,  # full frame too, for objects cut by slice borders
            postprocess_type="NMS",
            postprocess_match_metric="IOU",
            postprocess_match_threshold=self.args.iou,
            verbose=0,
        )
        rows = [
            [*p.bbox.to_xyxy(), p.score.value, p.category.id] for p in pred.object_prediction_list
        ]
        dets = np.asarray(rows, dtype=np.float32).reshape(-1, 6)
        order = np.argsort(-dets[:, 4])[: self.args.max_det]
        return dets[order]


def process_video(
    video: Path,
    detector: Detector,
    tracker_cfg: Any | None,
    names: list[str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Run detection (+ tracking) on one video and write its outputs; return a summary."""
    from ultralytics.engine.results import Boxes

    info = probe(video)
    out_dir = args.output
    stem = video.stem
    tracker = new_tracker(tracker_cfg) if tracker_cfg is not None else None
    raw_video = out_dir / f"{stem}_annotated.mp4v.mp4"
    writer = None if args.no_video else open_writer(raw_video, info.fps, info.width, info.height)

    records: list[dict[str, Any]] = []
    total = info.frame_count or None
    if args.max_frames:
        total = min(total or args.max_frames, args.max_frames)
    start = time.perf_counter()
    try:
        with tqdm(total=total, desc=stem, unit="frame", leave=False) as bar:
            for idx, ts, frame in iter_frames(video):
                if args.max_frames and idx >= args.max_frames:
                    break
                dets = detector(frame)
                if tracker is not None:
                    tracks = tracker.update(Boxes(dets, frame.shape[:2]), frame)
                    detections = detections_from_tracks(tracks)
                else:
                    detections = detections_from_array(dets)
                records.append(frame_record(idx, ts, detections, names))
                if writer is not None:
                    caption = f"frame {idx}  {ts:.2f} s"
                    writer.write(draw_detections(frame, detections, names, caption))
                bar.update()
    finally:
        if writer is not None:
            writer.release()
    seconds = time.perf_counter() - start

    (out_dir / f"{stem}_detections.json").write_text(json.dumps(records), encoding="utf-8")
    tracks = summarize_tracks(records)
    with (out_dir / f"{stem}_tracks.csv").open("w", newline="", encoding="utf-8") as fh:
        fields = ["track_id", "class", "frames", "first_frame", "last_frame", "duration_s"]
        fields += ["mean_conf", "median_box_px", "class_votes"]
        csv_writer = csv.DictWriter(fh, fieldnames=fields)
        csv_writer.writeheader()
        for row in tracks:
            csv_writer.writerow({**row, "class_votes": json.dumps(row["class_votes"])})

    if writer is not None:
        final = out_dir / f"{stem}_annotated.mp4"
        if transcode_h264(raw_video, final):
            raw_video.unlink()
        else:
            raw_video.replace(final)
            log.info("FFmpeg not available: %s stays MPEG-4 Part 2 (plays in VLC)", final.name)

    n_dets = sum(len(r["detections"]) for r in records)
    summary = {
        "video": str(video),
        "frames": len(records),
        "fps": info.fps,
        "size": [info.width, info.height],
        "detections": n_dets,
        "frames_with_detections": sum(bool(r["detections"]) for r in records),
        "tracks": len(tracks),
        "tracks_by_class": dict(sorted(Counter(t["class"] for t in tracks).items())),
        "processing_fps": round(len(records) / seconds, 2) if seconds else None,
    }
    log.info(
        "%s: %d frames, %d detections, %d tracks %s, %.1f frames/s",
        stem,
        summary["frames"],
        n_dets,
        summary["tracks"],
        summary["tracks_by_class"],
        summary["processing_fps"] or 0,
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(args.verbose)
    # resolve before prepare_ultralytics() changes the working directory
    weights = resolve_path(args.weights)
    sources = [resolve_path(s) for s in args.source]
    args.output = resolve_path(args.output)
    if not weights.exists():
        log.error("weights not found: %s", weights)
        return 1
    missing = [s for s in sources if not s.exists()]
    if missing:
        log.error("source not found: %s", ", ".join(map(str, missing)))
        return 1
    videos = find_videos(sources)
    if not videos:
        log.error("no videos (%s) in %s", ", ".join(sorted(VIDEO_EXTS)), args.source)
        return 1
    clashes = sorted(s for s, n in Counter(v.stem for v in videos).items() if n > 1)
    if clashes:
        log.error("several videos named %s; their outputs would overwrite each other", clashes)
        return 1
    use_tracker = args.tracker.lower() != "none"
    if args.conf is None:
        args.conf = TRACK_CONF if use_tracker else DETECT_CONF
    args.output.mkdir(parents=True, exist_ok=True)

    if not prepare_ultralytics():
        return 1
    from ultralytics import YOLO

    try:
        tracker_cfg = load_tracker_config(args.tracker) if use_tracker else None
        if tracker_cfg is not None:
            new_tracker(tracker_cfg)  # fail on a bad config before the first video
    except (OSError, ValueError, FileNotFoundError) as exc:
        log.error("tracker: %s", exc)
        return 1

    model = YOLO(str(weights))
    names = [model.names[i] for i in sorted(model.names)]
    if tuple(names) != CLASS_NAMES:
        log.warning("model classes %s differ from configs/dataset.yaml %s", names, CLASS_NAMES)
    imgsz = args.imgsz or int((model.ckpt or {}).get("train_args", {}).get("imgsz", 640))
    detector = Detector(model, args, imgsz)
    mode = f"SAHI slices {args.slice}px @ imgsz {args.slice_imgsz or imgsz}" if args.sahi else ""
    log.info(
        "%d video(s), imgsz %d, conf %.2f, NMS IoU %.2f, IoS %.2f, tracker %s %s -> %s",
        len(videos),
        imgsz,
        args.conf,
        args.iou,
        args.ios,
        tracker_cfg.tracker_type if tracker_cfg is not None else "none",
        mode,
        args.output,
    )

    summaries = []
    with logging_redirect_tqdm():
        for video in tqdm(videos, desc="videos", unit="video", disable=len(videos) == 1):
            summaries.append(process_video(video, detector, tracker_cfg, names, args))

    run = {
        "weights": str(weights),
        "git": git_state(),
        "settings": {
            "imgsz": imgsz,
            "conf": args.conf,
            "iou": args.iou,
            "ios": args.ios or None,
            "max_det": args.max_det,
            "tracker": None if tracker_cfg is None else dict(vars(tracker_cfg)),
            "sahi": (
                {
                    "slice": args.slice,
                    "overlap": args.overlap,
                    "imgsz": args.slice_imgsz or imgsz,
                    "postprocess": "NMS/IOU",
                }
                if args.sahi
                else None
            ),
            "max_frames": args.max_frames or None,
        },
        "videos": summaries,
    }
    (args.output / "infer_run.json").write_text(
        json.dumps(run, indent=2, default=str), encoding="utf-8"
    )
    log.info("done: %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
