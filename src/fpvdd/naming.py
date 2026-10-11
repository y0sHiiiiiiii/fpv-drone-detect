"""Frame file naming convention: ``<video_id>_f<frame:06d>.<ext>``.

The video id encoded in every frame name is what lets the dataset be split by
video. Any converter that produces frames must follow this convention.
"""

import re
from pathlib import Path

_FRAME_RE = re.compile(r"^(?P<video_id>.+)_f(?P<frame>\d{6,})$")
_UNSAFE_RE = re.compile(r"[^A-Za-z0-9_-]+")


def sanitize_video_id(name: str) -> str:
    """Turn an arbitrary video file stem into a filesystem-safe video id."""
    video_id = _UNSAFE_RE.sub("_", name).strip("_")
    if not video_id:
        raise ValueError(f"cannot derive a video id from {name!r}")
    return video_id


def frame_filename(video_id: str, frame_idx: int, ext: str = ".jpg") -> str:
    """Build the file name for frame ``frame_idx`` of ``video_id``."""
    if frame_idx < 0:
        raise ValueError(f"frame index must be >= 0, got {frame_idx}")
    return f"{video_id}_f{frame_idx:06d}{ext}"


def parse_frame_name(path: str | Path) -> tuple[str, int]:
    """Return ``(video_id, frame_idx)`` for a frame path; raise ValueError if it does not match."""
    stem = Path(path).stem
    match = _FRAME_RE.match(stem)
    if match is None:
        raise ValueError(f"not a frame name (<video_id>_f<NNNNNN>): {Path(path).name}")
    return match["video_id"], int(match["frame"])


def video_id_from_frame(path: str | Path) -> str:
    """Return the video id encoded in a frame path."""
    return parse_frame_name(path)[0]
