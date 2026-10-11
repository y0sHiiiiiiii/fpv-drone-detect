"""Near-duplicate frame detection with a difference hash (dHash)."""

from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path

import cv2
import numpy as np

from fpvdd.naming import parse_frame_name

HASH_SIZE = 8
HASH_BITS = HASH_SIZE * HASH_SIZE


def dhash(image: np.ndarray, hash_size: int = HASH_SIZE) -> int:
    """64-bit difference hash: sign of horizontal gradients on a 9x8 grayscale thumbnail."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    small = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int.from_bytes(np.packbits(bits).tobytes(), "big")


def similarity(a: int, b: int, bits: int = HASH_BITS) -> float:
    """1.0 for identical hashes, 0.0 if every bit differs."""
    return 1.0 - (a ^ b).bit_count() / bits


def _video_key(path: Path) -> tuple[str, int]:
    """Group by encoded video id; fall back to the parent folder for non-conforming names."""
    try:
        return parse_frame_name(path)
    except ValueError:
        return f"dir:{path.parent}", -1


def find_duplicates(hashes: Mapping[Path, int], threshold: float) -> list[Path]:
    """Return frames whose hash is at least ``threshold``-similar to the last kept frame.

    Frames are compared only within the same video, in frame order, against the most
    recently kept frame, so a slowly drifting scene still keeps a frame every so often.
    """
    if not 0.0 < threshold <= 1.0:
        raise ValueError(f"threshold must be in (0, 1], got {threshold}")
    by_video: dict[str, list[tuple[int, Path]]] = defaultdict(list)
    for path in hashes:
        video_id, frame_idx = _video_key(path)
        by_video[video_id].append((frame_idx, path))

    duplicates = []
    for frames in by_video.values():
        frames.sort(key=lambda item: (item[0], item[1].name))
        kept_hash = None
        for _idx, path in frames:
            h = hashes[path]
            if kept_hash is not None and similarity(h, kept_hash) >= threshold:
                duplicates.append(path)
            else:
                kept_hash = h
    return duplicates


def move_duplicates(duplicates: Iterable[Path], root: Path, dest: Path) -> int:
    """Move each duplicate to ``dest`` mirroring its path below ``root``; return the count."""
    moved = 0
    for path in duplicates:
        target = dest / path.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        path.replace(target)
        moved += 1
    return moved
