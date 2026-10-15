"""Converter helpers for the Halmstad multi-sensor drone detection dataset.

Svanström, Alonso-Fernandez, Englund: "A dataset for multi-sensor drone detection",
Data in Brief 39 (2021). CC0-1.0. https://github.com/DroneDetectionThesis/Drone-detection-dataset

Videos are named ``<IR|V>_<CLASS>_<NNN>.mp4`` with a MATLAB ``groundTruth`` label file
``<same>_LABELS.mat`` next to them. Labels are per frame and per label name (AIRPLANE, BIRD,
DRONE, HELICOPTER), with any number of ``[x, y, w, h]`` boxes in MATLAB pixel coordinates.
"""

import logging
import math
import random
import re
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

from fpvdd.images import write_image
from fpvdd.labels import YoloBox, write_label_file
from fpvdd.matlab_gt import FrameLabels, PixelBox, load_groundtruth
from fpvdd.naming import frame_filename
from fpvdd.video import iter_frames, probe
from fpvdd.video_eval import GTBox

log = logging.getLogger(__name__)

DATASET = "halmstad"
MODALITIES = {"visible": "V", "ir": "IR"}

_NAME_RE = re.compile(r"^(?P<modality>IR|V)_(?P<cls>[A-Z]+)_(?P<num>\d+)$")


class HalmstadError(ValueError):
    """Labels do not fit the video (wrong length, boxes outside the image, ...)."""


@dataclass(frozen=True)
class Clip:
    video: Path
    labels: Path
    modality: str  # "V" or "IR"
    source_class: str  # class in the file name, e.g. "DRONE"
    number: int

    @property
    def video_id(self) -> str:
        return f"{DATASET}_{self.video.stem}"


def parse_clip_name(video: Path) -> tuple[str, str, int]:
    """Return ``(modality, source_class, number)`` from a video file name."""
    match = _NAME_RE.match(video.stem)
    if match is None:
        raise ValueError(f"not a Halmstad video name: {video.name}")
    return match["modality"], match["cls"], int(match["num"])


def find_clips(root: Path, modalities: Sequence[str] = ("V",)) -> list[Clip]:
    """Find all videos of the given modalities below ``root`` that have a label file."""
    clips = []
    for video in sorted(root.rglob("*.mp4")):
        try:
            modality, cls, num = parse_clip_name(video)
        except ValueError:
            continue  # demo files and the like
        if modality not in modalities:
            continue
        labels = video.with_name(f"{video.stem}_LABELS.mat")
        if not labels.exists():
            log.warning("no label file for %s, skipping", video.name)
            continue
        clips.append(Clip(video, labels, modality, cls, num))
    return clips


def load_labels(clip: Clip) -> list[FrameLabels]:
    """Per-frame ``{label_name: [boxes]}`` for a clip."""
    return load_groundtruth(clip.labels)


def matlab_to_pixels(box: PixelBox) -> PixelBox:
    """MATLAB spatial coords put the first pixel's top-left corner at 0.5; shift to 0-based."""
    x, y, w, h = box
    return x - 0.5, y - 0.5, w, h


def validate_labels(
    frames: Sequence[FrameLabels], n_frames: int, width: int, height: int, tol: float = 2.0
) -> None:
    """Raise HalmstadError if labels and video disagree.

    Requires exactly one label entry per video frame, boxes of positive size, and boxes
    inside the image within ``tol`` pixels.
    """
    if len(frames) != n_frames:
        raise HalmstadError(f"{len(frames)} label frames but the video has {n_frames} frames")
    for idx, labels in enumerate(frames):
        for name, boxes in labels.items():
            for box in boxes:
                x, y, w, h = matlab_to_pixels(box)
                if w <= 0 or h <= 0:
                    raise HalmstadError(f"frame {idx} {name}: non-positive box size {box}")
                if x < -tol or y < -tol or x + w > width + tol or y + h > height + tol:
                    raise HalmstadError(f"frame {idx} {name}: box {box} outside {width}x{height}")


def frame_boxes(
    labels: FrameLabels, map_class: Callable[[str], int | None], img_w: int, img_h: int
) -> tuple[list[YoloBox], int]:
    """Map one frame's labels to YOLO boxes; return ``(boxes, number_of_dropped_boxes)``."""
    boxes, dropped = [], 0
    for name, pixel_boxes in labels.items():
        class_id = map_class(name)
        for pixel_box in pixel_boxes:
            box = None
            if class_id is not None:
                box = YoloBox.from_pixels(class_id, *matlab_to_pixels(pixel_box), img_w, img_h)
            if box is None:
                dropped += 1
            else:
                boxes.append(box)
    return boxes, dropped


def background_crop(
    obstacles: Sequence[PixelBox],
    img_w: int,
    img_h: int,
    sizes: Sequence[float],
    margin: float,
    rng: random.Random,
    tries: int = 50,
) -> tuple[int, int, int, int] | None:
    """Pick a crop ``(x, y, w, h)`` that stays ``margin`` px away from every obstacle box.

    ``obstacles`` are 0-based ``x, y, w, h`` pixel boxes. ``sizes`` are crop sides as a
    fraction of the image side, tried in order. Candidates per size are ``tries`` random
    positions plus the positions flush with the image border or an obstacle's margin, so a
    narrow free strip (e.g. beside a centred object) is still found. One valid candidate is
    picked at random. Returns None if no size fits.
    """

    def free(cx: int, cy: int, cw: int, ch: int) -> bool:
        return not any(
            cx < x + w + margin
            and cx + cw > x - margin
            and cy < y + h + margin
            and cy + ch > y - margin
            for x, y, w, h in obstacles
        )

    for size in sizes:
        cw, ch = round(img_w * size), round(img_h * size)
        if not (0 < cw <= img_w and 0 < ch <= img_h):
            raise ValueError(f"crop size {size} out of range (0, 1]")
        xs = {0, img_w - cw}
        ys = {0, img_h - ch}
        for x, y, w, h in obstacles:
            xs.update((math.ceil(x + w + margin), math.floor(x - margin) - cw))
            ys.update((math.ceil(y + h + margin), math.floor(y - margin) - ch))
        xs = {v for v in xs if 0 <= v <= img_w - cw}
        ys = {v for v in ys if 0 <= v <= img_h - ch}
        candidates = [(cx, cy) for cx in sorted(xs) for cy in sorted(ys)]
        candidates += [
            (rng.randint(0, img_w - cw), rng.randint(0, img_h - ch)) for _ in range(tries)
        ]
        valid = [(cx, cy) for cx, cy in candidates if free(cx, cy, cw, ch)]
        if valid:
            cx, cy = rng.choice(valid)
            return cx, cy, cw, ch
    return None


def video_ground_truth(
    frames: Sequence[FrameLabels], map_class: Callable[[str], int | None]
) -> tuple[dict[int, list[GTBox]], bool]:
    """Per-frame ground truth for video evaluation; returns ``(frames, has_identities)``.

    Halmstad has no object ids. If no frame contains more than one object of our classes, all
    such boxes belong to the same object (id 1) and identity metrics can be computed;
    otherwise the ids stay None.
    """
    mapped = [
        [
            (map_class(name), matlab_to_pixels(box))
            for name, boxes in labels.items()
            for box in boxes
        ]
        for labels in frames
    ]
    single = all(sum(cls is not None for cls, _ in f) <= 1 for f in mapped)
    out = {
        idx: [
            GTBox(cls, (x, y, x + w, y + h), 1 if single and cls is not None else None)
            for cls, (x, y, w, h) in f
        ]
        for idx, f in enumerate(mapped)
    }
    return out, single


def convert_clip(
    clip: Clip,
    frames: Sequence[FrameLabels],
    map_class: Callable[[str], int | None],
    images_root: Path,
    labels_root: Path,
    target_fps: float | None,
    jpeg_quality: int = 95,
    bg_every: int = 0,
    bg_sizes: Sequence[float] = (0.7, 0.6, 0.5),
    bg_margin: float = 16.0,
) -> Counter[str]:
    """Write sampled frames and YOLO labels for one clip; return counters.

    Boxes whose label maps to None are dropped, but their frames are still written (with the
    remaining boxes, possibly none), so objects outside our class schema become negatives.
    With ``bg_every`` > 0, every ``bg_every``-th sampled frame also yields one background crop
    away from all labeled boxes (see `background_crop`), saved as ``<frame name>_bg`` with an
    empty label file. Counters: ``frames``, ``background`` (frames without boxes), ``dropped``,
    ``bg_crops``, ``bg_crop_failed`` and ``box:<class name>``.
    """
    info = probe(clip.video)
    validate_labels(frames, info.frame_count, info.width, info.height)
    image_dir = images_root / DATASET / clip.video_id
    label_dir = labels_root / DATASET / clip.video_id
    image_dir.mkdir(parents=True, exist_ok=True)

    stats: Counter[str] = Counter()
    for sample, (idx, _ts, frame) in enumerate(
        tqdm(iter_frames(clip.video, target_fps), desc=clip.video_id, unit="frame", leave=False)
    ):
        if idx >= len(frames):
            raise HalmstadError(f"decoded more frames than the container reports ({idx + 1})")
        h, w = frame.shape[:2]
        boxes, dropped = frame_boxes(frames[idx], map_class, w, h)
        name = frame_filename(clip.video_id, idx)
        write_image(image_dir / name, frame, jpeg_quality)
        write_label_file((label_dir / name).with_suffix(".txt"), boxes)
        stats["frames"] += 1
        stats["background"] += not boxes
        stats["dropped"] += dropped
        stats.update(f"box:{b.class_name}" for b in boxes)
        if bg_every > 0 and sample % bg_every == bg_every // 2:
            obstacles = [
                matlab_to_pixels(b) for pixel_boxes in frames[idx].values() for b in pixel_boxes
            ]
            rng = random.Random(f"{clip.video_id}:{idx}")  # deterministic per frame
            crop = background_crop(obstacles, w, h, bg_sizes, bg_margin, rng)
            if crop is None:
                stats["bg_crop_failed"] += 1
                continue
            x, y, cw, ch = crop
            bg_name = frame_filename(clip.video_id, idx, tag="bg")
            write_image(image_dir / bg_name, frame[y : y + ch, x : x + cw], jpeg_quality)
            write_label_file((label_dir / bg_name).with_suffix(".txt"), [])
            stats["bg_crops"] += 1
    return stats
