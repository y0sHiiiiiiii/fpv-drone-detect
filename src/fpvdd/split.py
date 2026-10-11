"""Train/val/test splitting by video (never by frame)."""

import math
import random
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from fpvdd.labels import YoloBox
from fpvdd.naming import video_id_from_frame

SPLIT_NAMES: tuple[str, ...] = ("train", "val", "test")


class LeakageError(RuntimeError):
    """The same video appears in more than one split."""


def group_by_video(images: Iterable[Path]) -> dict[str, list[Path]]:
    """Group frame paths by the video id encoded in their file names."""
    groups: dict[str, list[Path]] = defaultdict(list)
    for image in images:
        groups[video_id_from_frame(image)].append(image)
    return {vid: sorted(paths) for vid, paths in sorted(groups.items())}


def _validate_ratios(ratios: Sequence[float]) -> None:
    if len(ratios) != len(SPLIT_NAMES):
        raise ValueError(f"expected {len(SPLIT_NAMES)} ratios, got {len(ratios)}")
    if any(r < 0 for r in ratios):
        raise ValueError(f"ratios must be >= 0, got {list(ratios)}")
    if not math.isclose(sum(ratios), 1.0, abs_tol=1e-6):
        raise ValueError(f"ratios must sum to 1, got {sum(ratios):.4f}")


def assign_splits(
    group_sizes: Mapping[str, int], ratios: Sequence[float], seed: int = 0
) -> dict[str, str]:
    """Assign whole videos to splits so frame counts approximate ``ratios``.

    Greedy: videos are visited largest first (ties broken by a seeded shuffle) and each
    goes to the split furthest below its target frame count. Every split with a non-zero
    ratio receives at least one video. Deterministic for a given ``seed``.
    """
    _validate_ratios(ratios)
    active = [name for name, r in zip(SPLIT_NAMES, ratios, strict=True) if r > 0]
    if len(group_sizes) < len(active):
        raise ValueError(
            f"need at least {len(active)} videos for splits {active}, got {len(group_sizes)}"
        )

    order = sorted(group_sizes)
    random.Random(seed).shuffle(order)
    order.sort(key=lambda vid: group_sizes[vid], reverse=True)  # stable: shuffle breaks ties

    total = sum(group_sizes.values())
    target = {name: r * total for name, r in zip(SPLIT_NAMES, ratios, strict=True)}
    current = dict.fromkeys(SPLIT_NAMES, 0)
    members: dict[str, list[str]] = {name: [] for name in SPLIT_NAMES}
    for vid in order:
        split = max(active, key=lambda s: (target[s] - current[s], -SPLIT_NAMES.index(s)))
        members[split].append(vid)
        current[split] += group_sizes[vid]

    # Guarantee no active split is empty: borrow the smallest video from the fullest donor.
    for split in active:
        if members[split]:
            continue
        donors = [s for s in active if len(members[s]) > 1]
        donor = max(donors, key=lambda s: current[s] - target[s])
        vid = min(members[donor], key=lambda v: group_sizes[v])
        members[donor].remove(vid)
        members[split].append(vid)
        current[donor] -= group_sizes[vid]
        current[split] += group_sizes[vid]

    return {vid: split for split, vids in members.items() for vid in vids}


def check_no_leakage(split_images: Mapping[str, Iterable[Path]]) -> None:
    """Raise LeakageError if any video id occurs in more than one split."""
    seen: dict[str, str] = {}
    for split, images in split_images.items():
        for image in images:
            vid = video_id_from_frame(image)
            other = seen.setdefault(vid, split)
            if other != split:
                raise LeakageError(f"video {vid!r} is in both {other!r} and {split!r}")


def background_ratio(boxes_per_frame: Sequence[Sequence[YoloBox]]) -> float:
    """Fraction of frames without any boxes (0.0 for an empty sequence)."""
    if not boxes_per_frame:
        return 0.0
    return sum(not boxes for boxes in boxes_per_frame) / len(boxes_per_frame)
