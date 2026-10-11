"""Image I/O that also works with non-ASCII paths on Windows."""

from pathlib import Path

import cv2
import numpy as np


def read_image(path: Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    """Read an image file; raise OSError if it cannot be decoded."""
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, flags)
    if image is None:
        raise OSError(f"cannot decode image: {path}")
    return image


def write_image(path: Path, image: np.ndarray, jpeg_quality: int = 95) -> None:
    """Encode ``image`` by the file extension of ``path`` and write it."""
    params = (
        [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality] if path.suffix.lower() in {".jpg", ".jpeg"} else []
    )
    ok, buf = cv2.imencode(path.suffix, image, params)
    if not ok:
        raise OSError(f"cannot encode image: {path}")
    buf.tofile(path)
