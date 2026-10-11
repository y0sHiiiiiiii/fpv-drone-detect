"""Reading frames from video files with OpenCV."""

import logging
import shutil
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from fpvdd.images import write_image
from fpvdd.naming import frame_filename

log = logging.getLogger(__name__)

_FALLBACK_FPS = 30.0


@dataclass(frozen=True)
class VideoInfo:
    fps: float
    frame_count: int
    width: int
    height: int


def _open(path: Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise OSError(f"cannot open video: {path}")
    return cap


def _fps(cap: cv2.VideoCapture, path: Path) -> float:
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        log.warning("%s reports no FPS, assuming %.0f", path.name, _FALLBACK_FPS)
        return _FALLBACK_FPS
    return fps


def probe(path: Path) -> VideoInfo:
    """Return basic stream properties (frame count is the container's estimate)."""
    cap = _open(path)
    try:
        return VideoInfo(
            fps=_fps(cap, path),
            frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
    finally:
        cap.release()


def iter_frames(
    path: Path, target_fps: float | None = None
) -> Iterator[tuple[int, float, np.ndarray]]:
    """Yield ``(frame_idx, timestamp_s, bgr_image)``, subsampled to ``target_fps``.

    With ``target_fps`` None or >= the source FPS, every frame is yielded. Otherwise
    frames are picked at evenly spaced times, so non-integer ratios (e.g. 30 -> 4 fps)
    still give the requested rate on average.
    """
    if target_fps is not None and target_fps <= 0:
        raise ValueError(f"target_fps must be > 0, got {target_fps}")
    cap = _open(path)
    try:
        src_fps = _fps(cap, path)
        step = 1.0 if target_fps is None or target_fps >= src_fps else src_fps / target_fps
        next_keep = 0.0
        idx = 0
        while cap.grab():
            if idx + 1e-9 >= next_keep:
                ok, frame = cap.retrieve()
                if not ok:
                    log.warning("%s: failed to decode frame %d, stopping", path.name, idx)
                    break
                yield idx, idx / src_fps, frame
                next_keep += step
            idx += 1
    finally:
        cap.release()


def extract_video(
    video: Path, out_dir: Path, video_id: str, target_fps: float | None, jpeg_quality: int = 95
) -> int:
    """Write sampled frames of ``video`` as ``out_dir/<video_id>_fNNNNNN.jpg``; return the count."""
    out_dir.mkdir(parents=True, exist_ok=True)
    info = probe(video)
    expected = info.frame_count
    if target_fps is not None and target_fps < info.fps:
        expected = round(info.frame_count * target_fps / info.fps)
    written = 0
    with tqdm(total=expected or None, desc=video_id, unit="frame", leave=False) as bar:
        for idx, _ts, frame in iter_frames(video, target_fps):
            write_image(out_dir / frame_filename(video_id, idx), frame, jpeg_quality)
            written += 1
            bar.update()
    return written


def open_writer(path: Path, fps: float, width: int, height: int) -> cv2.VideoWriter:
    """MPEG-4 Part 2 writer (``mp4v``); see :func:`transcode_h264` for wide playback support."""
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise OSError(f"cannot open video writer: {path}")
    return writer


def transcode_h264(src: Path, dst: Path, crf: int = 20) -> bool:
    """Re-encode ``src`` to H.264 with FFmpeg; False if FFmpeg is missing or fails.

    OpenCV's ``mp4v`` output plays in VLC but not in browsers and many other players.
    """
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return False
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-i", str(src), "-c:v", "libx264"]
    cmd += ["-crf", str(crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log.warning("ffmpeg failed for %s: %s", src.name, result.stderr.strip())
        return False
    return True
