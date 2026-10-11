"""Draw labels on a random sample of a labeled dataset for visual QA.

Writes one annotated image per sample plus contact sheets to ``--output``.
"""

import argparse
import logging
import random
import sys
from pathlib import Path

from fpvdd.draw import contact_sheet, draw_boxes
from fpvdd.images import read_image, write_image
from fpvdd.labels import CLASS_NAMES, LabelError, parse_label_file
from fpvdd.log import setup_logging
from fpvdd.paths import DATA_DIR, IMAGE_EXTS, LABELED_DIR

log = logging.getLogger("visualize_labels")

SHEET_SIZE = 16


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=LABELED_DIR, help="folder with images/ and labels/"
    )
    parser.add_argument(
        "--subdir", default="", help="only sample below images/<subdir> (e.g. a dataset name)"
    )
    parser.add_argument("--output", type=Path, default=DATA_DIR / "qa", help="output folder")
    parser.add_argument("-n", "--num", type=int, default=48, help="number of samples")
    parser.add_argument(
        "--class",
        dest="class_name",
        choices=[*CLASS_NAMES, "background"],
        help="only sample frames containing this class (or background frames)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(args.verbose)
    images_dir, labels_dir = args.input / "images", args.input / "labels"
    root = images_dir / args.subdir
    images = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    if not images:
        log.error("no images in %s", root)
        return 1

    def boxes_for(img: Path):
        return parse_label_file((labels_dir / img.relative_to(images_dir)).with_suffix(".txt"))

    rng = random.Random(args.seed)
    rng.shuffle(images)
    picked = []
    try:
        for img in images:
            boxes = boxes_for(img)
            if args.class_name == "background" and boxes:
                continue
            if args.class_name not in (None, "background") and not any(
                b.class_name == args.class_name for b in boxes
            ):
                continue
            picked.append((img, boxes))
            if len(picked) >= args.num:
                break
    except LabelError as exc:
        log.error("%s", exc)
        return 1
    if not picked:
        log.error("no matching samples")
        return 1

    args.output.mkdir(parents=True, exist_ok=True)
    drawn = []
    for img, boxes in picked:
        annotated = draw_boxes(read_image(img), boxes)
        write_image(args.output / img.name, annotated)
        drawn.append(annotated)
    for i in range(0, len(drawn), SHEET_SIZE):
        sheet = contact_sheet(drawn[i : i + SHEET_SIZE])
        write_image(args.output / f"_sheet_{i // SHEET_SIZE:02d}.jpg", sheet, jpeg_quality=85)
    log.info("wrote %d samples and contact sheets to %s", len(picked), args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
