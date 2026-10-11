"""Extract frames from videos into data/frames/<video_id>/<video_id>_fNNNNNN.jpg."""

import argparse
import logging
import sys
from pathlib import Path

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from fpvdd.log import setup_logging
from fpvdd.naming import sanitize_video_id
from fpvdd.paths import FRAMES_DIR, RAW_DIR, VIDEO_EXTS
from fpvdd.video import extract_video

log = logging.getLogger("extract_frames")


def parse_exts(spec: str) -> frozenset[str]:
    """Parse ``"mp4,.MKV"`` into ``{".mp4", ".mkv"}``."""
    parts = (e.strip().lower() for e in spec.split(","))
    return frozenset(e if e.startswith(".") else f".{e}" for e in parts if e)


def find_videos(source: Path, exts: frozenset[str]) -> list[Path]:
    """Return ``source`` itself if it is a file, else all videos below it (sorted)."""
    if source.is_file():
        return [source]
    return sorted(p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in exts)


def assign_video_ids(videos: list[Path]) -> dict[Path, str]:
    """Map each video to a sanitized id; raise ValueError on collisions."""
    ids: dict[str, Path] = {}
    for video in videos:
        vid = sanitize_video_id(video.stem)
        if vid in ids:
            raise ValueError(f"video id {vid!r} used by both {ids[vid]} and {video}; rename one")
        ids[vid] = video
    return {video: vid for vid, video in ids.items()}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=RAW_DIR, help="video file or folder")
    parser.add_argument("--output", type=Path, default=FRAMES_DIR, help="frames root folder")
    parser.add_argument(
        "--fps", type=float, default=5.0, help="target sampling rate; 0 keeps every frame"
    )
    parser.add_argument(
        "--ext",
        default=",".join(sorted(VIDEO_EXTS)),
        help="comma-separated video extensions to search for",
    )
    parser.add_argument("--quality", type=int, default=95, help="JPEG quality (1-100)")
    parser.add_argument(
        "--overwrite", action="store_true", help="re-extract videos that already have frames"
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(args.verbose)
    if not args.input.exists():
        log.error("input not found: %s", args.input)
        return 1
    if args.fps < 0:
        log.error("--fps must be >= 0")
        return 1

    exts = parse_exts(args.ext)
    videos = find_videos(args.input, exts)
    if not videos:
        log.error("no videos with extensions %s in %s", sorted(exts), args.input)
        return 1
    try:
        video_ids = assign_video_ids(videos)
    except ValueError as exc:
        log.error("%s", exc)
        return 1

    target_fps = args.fps or None
    total = done = skipped = failed = 0
    with logging_redirect_tqdm():
        for video in tqdm(videos, desc="videos", unit="video"):
            vid = video_ids[video]
            out_dir = args.output / vid
            existing = sorted(out_dir.glob(f"{vid}_f*.jpg")) if out_dir.is_dir() else []
            if existing and not args.overwrite:
                log.info("skip %s (frames exist, use --overwrite)", vid)
                skipped += 1
                continue
            for old in existing:  # --overwrite: avoid mixing frames from different --fps runs
                old.unlink()
            try:
                n = extract_video(video, out_dir, vid, target_fps, args.quality)
            except OSError as exc:
                log.error("%s: %s", video, exc)
                failed += 1
                continue
            log.info("%s: %d frames", vid, n)
            total += n
            done += 1
    log.info("done: %d frames from %d videos (%d skipped, %d failed)", total, done, skipped, failed)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
