"""Convert the Halmstad drone detection dataset to YOLO frames + labels in data/labeled/.

Output: images/halmstad/<video_id>/<video_id>_fNNNNNN.jpg with mirrored labels/, where
video_id is ``halmstad_<original stem>`` (e.g. halmstad_V_DRONE_001). Background crops
(options.background_crops) are written next to them as <video_id>_fNNNNNN_bg.jpg.
"""

import argparse
import logging
import shutil
import sys
from collections import Counter
from pathlib import Path

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from fpvdd.halmstad import (
    DATASET,
    MODALITIES,
    HalmstadError,
    convert_clip,
    find_clips,
    load_labels,
)
from fpvdd.log import setup_logging
from fpvdd.matlab_gt import MatParseError
from fpvdd.paths import CONFIGS_DIR, LABELED_DIR, RAW_DIR
from fpvdd.sources import load_source_config

log = logging.getLogger("convert_halmstad")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=RAW_DIR / DATASET, help="unpacked dataset")
    parser.add_argument("--output", type=Path, default=LABELED_DIR, help="labeled dataset root")
    parser.add_argument("--config", type=Path, default=CONFIGS_DIR / "sources" / f"{DATASET}.yaml")
    parser.add_argument("--fps", type=float, default=5.0, help="sampling rate; 0 = every frame")
    parser.add_argument(
        "--negative-fps",
        type=float,
        default=None,
        help="sampling rate for clips whose class is dropped (default: options.negative_fps, "
        "else --fps); keeps the background share in the 10-20 %% range",
    )
    parser.add_argument(
        "--bg-every",
        type=int,
        default=None,
        help="one background crop every N sampled frames (default: "
        "options.background_crops.every; 0 = no crops)",
    )
    parser.add_argument(
        "--modality",
        choices=[*MODALITIES, "both"],
        default=None,
        help="override options.modalities from the config",
    )
    parser.add_argument("--quality", type=int, default=95, help="JPEG quality (1-100)")
    parser.add_argument("--limit", type=int, default=0, help="convert at most N clips (testing)")
    parser.add_argument(
        "--overwrite", action="store_true", help="re-convert clips that already have output"
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(args.verbose)
    if not args.input.is_dir():
        log.error("dataset folder not found: %s (download and unpack it manually)", args.input)
        return 1
    config = load_source_config(args.config)

    if args.modality == "both":
        names = list(MODALITIES)
    elif args.modality:
        names = [args.modality]
    else:
        names = list(config.options.get("modalities", ["visible"]))
    unknown = set(names) - set(MODALITIES)
    if unknown:
        log.error("unknown modalities %s, use %s", sorted(unknown), list(MODALITIES))
        return 1

    negative_fps = args.negative_fps
    if negative_fps is None:
        negative_fps = float(config.options.get("negative_fps", args.fps))
    if args.fps < 0 or negative_fps < 0:
        log.error("--fps and --negative-fps must be >= 0")
        return 1

    bg_options = dict(config.options.get("background_crops") or {})
    bg_every = args.bg_every if args.bg_every is not None else int(bg_options.get("every", 0))
    bg_sizes = [float(s) for s in bg_options.get("sizes", [0.7, 0.6, 0.5])]
    bg_margin = float(bg_options.get("margin", 16))
    if bg_every < 0:
        log.error("--bg-every must be >= 0")
        return 1

    clips = find_clips(args.input, [MODALITIES[m] for m in names])
    if args.limit:
        clips = clips[: args.limit]
    if not clips:
        log.error("no labeled %s videos found in %s", names, args.input)
        return 1
    log.info("%d clips (%s)", len(clips), ", ".join(names))

    images_root, labels_root = args.output / "images", args.output / "labels"
    stats: Counter[str] = Counter()
    failed: list[str] = []
    with logging_redirect_tqdm():
        for clip in tqdm(clips, desc="clips", unit="clip"):
            image_dir = images_root / DATASET / clip.video_id
            label_dir = labels_root / DATASET / clip.video_id
            if image_dir.is_dir() and any(image_dir.iterdir()):
                if not args.overwrite:
                    stats["clips_skipped"] += 1
                    continue
                remove_outputs(image_dir, label_dir)
            try:
                # clips of dropped classes only yield negatives: sample them more sparsely
                negative = config.map_class(clip.source_class) is None
                fps = (negative_fps if negative else args.fps) or None
                frames = load_labels(clip)
                clip_stats = convert_clip(
                    clip,
                    frames,
                    config.map_class,
                    images_root,
                    labels_root,
                    fps,
                    args.quality,
                    bg_every=bg_every,
                    bg_sizes=bg_sizes,
                    bg_margin=bg_margin,
                )
            except (HalmstadError, MatParseError, KeyError, OSError, ValueError) as exc:
                log.error("%s: %s", clip.video.name, exc)
                remove_outputs(image_dir, label_dir)  # no half-converted clips
                failed.append(clip.video.name)
                continue
            stats.update(clip_stats)
            stats["clips"] += 1
            log.debug("%s: %s", clip.video_id, dict(clip_stats))

    boxes = ", ".join(f"{k[4:]}={v}" for k, v in sorted(stats.items()) if k.startswith("box:"))
    log.info(
        "done: %d clips (%d skipped, %d failed), %d frames, %d background | boxes: %s | "
        "dropped boxes (unmapped classes): %d",
        stats["clips"],
        stats["clips_skipped"],
        len(failed),
        stats["frames"],
        stats["background"],
        boxes or "-",
        stats["dropped"],
    )
    if bg_every:
        images = stats["frames"] + stats["bg_crops"]
        log.info(
            "background crops: %d written, %d frames without room for one | background share "
            "%.1f %% of %d images",
            stats["bg_crops"],
            stats["bg_crop_failed"],
            100 * (stats["background"] + stats["bg_crops"]) / max(images, 1),
            images,
        )
    if failed:
        log.warning("failed clips: %s", ", ".join(failed))
    return 1 if failed else 0


def remove_outputs(*dirs: Path) -> None:
    for d in dirs:
        shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
