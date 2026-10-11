from pathlib import Path

import cv2
import numpy as np
import pytest


@pytest.fixture
def make_video(tmp_path: Path):
    """Write a small synthetic MJPG video and return its path."""

    def _make(name: str = "clip.avi", n_frames: int = 30, fps: float = 30.0) -> Path:
        path = tmp_path / name
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (64, 48))
        assert writer.isOpened()
        for i in range(n_frames):
            frame = np.full((48, 64, 3), (i * 8) % 256, dtype=np.uint8)
            writer.write(frame)
        writer.release()
        return path

    return _make
