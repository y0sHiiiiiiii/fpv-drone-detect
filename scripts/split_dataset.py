"""Split a labeled YOLO dataset into train/val/test by video and write Ultralytics list files.

Expects ``<input>/images/**`` and a mirrored ``<input>/labels/**`` (YOLO .txt per image;
a missing or empty label file is a background frame). Writes ``train.txt``, ``val.txt``,
``test.txt`` with absolute image paths plus ``split_manifest.csv`` to ``--output``.
"""

import argparse
import csv
import logging
import sys
from collections import Counter
from pathlib import Path

from fpvdd.labels import CLASS_NAMES, LabelError, YoloBox, parse_label_file
from fpvdd.log import setup_logging
from fpvdd.paths import IMAGE_EXTS, LABELED_DIR, SPLITS_DIR
from fpvdd.split import (
    SPLIT_NAMES,
    LeakageError,
    assign_splits,
    assign_splits_stratified,
    background_ratio,
    check_no_leakage,
    group_by_video,
)

log = logging.getLogger("split_dataset")

BACKGROUND_RANGE = (0.10, 0.20)


def label_path_for(image: Path, images_dir: Path, labels_dir: Path) -> Path:
    """Mirror ``images/<rel>.jpg`` to ``labels/<rel>.txt``."""
    return (labels_dir / image.relative_to(images_dir)).with_suffix(".txt")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=LABELED_DIR, help="folder with images/ and labels/"
    )
    parser.add_argument("--output", type=Path, default=SPLITS_DIR, help="where list files go")
    parser.add_argument(
        "--by",
        choices=["video"],
        default="video",
        help="grouping unit; only 'video' is allowed so frames of one video never leak",
    )
    parser.add_argument(
        "--ratios",
        type=float,
        nargs=3,
        default=[0.7, 0.15, 0.15],
        metavar=("TRAIN", "VAL", "TEST"),
    )
    parser.add_argument(
        "--stratify",
        choices=["class", "none"],
        default="class",
        help="balance each video's dominant class (or background) across splits",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def video_stratum(boxes_per_frame: list[list[YoloBox]]) -> str:
    """Dominant class name over all boxes of a video, or 'background' if it has none."""
    counts = Counter(b.class_name for boxes in boxes_per_frame for b in boxes)
    return counts.most_common(1)[0][0] if counts else "background"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(args.verbose)
    images_dir, labels_dir = args.input / "images", args.input / "labels"
    if not images_dir.is_dir():
        log.error("missing folder: %s", images_dir)
        return 1

    images = sorted(p for p in images_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    if not images:
        log.error("no images in %s", images_dir)
        return 1

    labels = {img: label_path_for(img, images_dir, labels_dir) for img in images}
    errors, class_counts = [], {name: Counter() for name in SPLIT_NAMES}
    boxes_per_image = {}
    for img, lbl in labels.items():
        try:
            boxes_per_image[img] = parse_label_file(lbl)
        except LabelError as exc:
            errors.append(str(exc))
    if errors:
        for err in errors[:20]:
            log.error("%s", err)
        log.error("%d invalid label files, fix them first", len(errors))
        return 1

    try:
        groups = group_by_video(images)
    except ValueError as exc:
        log.error("%s", exc)
        log.error("all images must be named <video_id>_fNNNNNN.<ext> to split by video")
        return 1

    sizes = {vid: len(p) for vid, p in groups.items()}
    strata = {
        vid: video_stratum([boxes_per_image[p] for p in paths]) for vid, paths in groups.items()
    }
    try:
        if args.stratify == "class":
            log.info("strata (videos): %s", dict(sorted(Counter(strata.values()).items())))
            assignment = assign_splits_stratified(sizes, strata, args.ratios, args.seed)
        else:
            assignment = assign_splits(sizes, args.ratios, args.seed)
    except ValueError as exc:
        log.error("%s", exc)
        return 1

    split_images: dict[str, list[Path]] = {name: [] for name in SPLIT_NAMES}
    for vid, paths in groups.items():
        split_images[assignment[vid]].extend(paths)
    try:
        check_no_leakage(split_images)
    except LeakageError as exc:
        log.error("%s", exc)
        return 1

    args.output.mkdir(parents=True, exist_ok=True)
    for split, paths in split_images.items():
        text = "".join(f"{p.resolve()}\n" for p in sorted(paths))
        (args.output / f"{split}.txt").write_text(text, encoding="utf-8")
        for p in paths:
            class_counts[split].update(CLASS_NAMES[b.class_id] for b in boxes_per_image[p])

    manifest = args.output / "split_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["video_id", "split", "stratum", "n_images", "n_background"])
        for vid, paths in groups.items():
            n_bg = sum(not boxes_per_image[p] for p in paths)
            writer.writerow([vid, assignment[vid], strata[vid], len(paths), n_bg])

    total = len(images)
    for split, paths in split_images.items():
        n_videos = sum(1 for s in assignment.values() if s == split)
        bg = background_ratio([boxes_per_image[p] for p in paths])
        counts = ", ".join(f"{k}={v}" for k, v in sorted(class_counts[split].items())) or "-"
        log.info(
            "%-5s %3d videos %6d images (%5.1f %%) background %5.1f %% | %s",
            split,
            n_videos,
            len(paths),
            100 * len(paths) / total,
            100 * bg,
            counts,
        )

    train_bg = background_ratio([boxes_per_image[p] for p in split_images["train"]])
    low, high = BACKGROUND_RANGE
    if split_images["train"] and not low <= train_bg <= high:
        log.warning(
            "train background share is %.1f %%, target is %.0f-%.0f %%",
            100 * train_bg,
            100 * low,
            100 * high,
        )
    log.info("wrote %s and list files to %s", manifest.name, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
