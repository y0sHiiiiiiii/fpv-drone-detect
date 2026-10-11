"""Move near-duplicate consecutive frames out of the frames folder (non-destructive)."""

import argparse
import logging
import sys
from pathlib import Path

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from fpvdd.dedupe import dhash, find_duplicates, move_duplicates
from fpvdd.images import read_image
from fpvdd.log import setup_logging
from fpvdd.paths import FRAMES_DIR, IMAGE_EXTS

log = logging.getLogger("dedupe_frames")

DUPLICATES_DIRNAME = "_duplicates"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=FRAMES_DIR, help="frames root folder")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.95,
        help="dHash similarity in (0, 1] at or above which a frame counts as duplicate",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=None,
        help=f"where duplicates are moved (default: <input>/{DUPLICATES_DIRNAME})",
    )
    parser.add_argument("--dry-run", action="store_true", help="only report, move nothing")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(args.verbose)
    if not args.input.is_dir():
        log.error("input folder not found: %s", args.input)
        return 1
    dest = args.dest or args.input / DUPLICATES_DIRNAME

    frames = sorted(
        p
        for p in args.input.rglob("*")
        if p.suffix.lower() in IMAGE_EXTS and DUPLICATES_DIRNAME not in p.parts
    )
    if not frames:
        log.error("no images found in %s", args.input)
        return 1

    hashes = {}
    with logging_redirect_tqdm():
        for frame in tqdm(frames, desc="hashing", unit="img"):
            try:
                hashes[frame] = dhash(read_image(frame))
            except OSError as exc:
                log.warning("%s", exc)

    try:
        duplicates = find_duplicates(hashes, args.threshold)
    except ValueError as exc:
        log.error("%s", exc)
        return 1

    share = len(duplicates) / len(hashes) if hashes else 0.0
    log.info("%d of %d frames are duplicates (%.1f %%)", len(duplicates), len(hashes), 100 * share)
    if args.dry_run:
        for path in duplicates:
            log.debug("duplicate: %s", path)
        log.info("dry run: nothing moved")
        return 0
    moved = move_duplicates(duplicates, args.input, dest)
    log.info("moved %d frames to %s", moved, dest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
