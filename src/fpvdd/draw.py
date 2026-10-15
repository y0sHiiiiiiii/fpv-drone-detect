"""Drawing YOLO boxes on images for visual label checks."""

from collections.abc import Sequence

import cv2
import numpy as np

from fpvdd.inference import Detection
from fpvdd.labels import YoloBox

# BGR, one colour per class id
CLASS_COLORS: tuple[tuple[int, int, int], ...] = (
    (60, 60, 230),  # multirotor
    (230, 140, 40),  # fixed_wing
    (40, 200, 230),  # vtol
    (80, 200, 80),  # bird
    (200, 80, 200),  # unknown_aerial
    (200, 200, 60),  # airplane
    (40, 120, 240),  # helicopter
)


def draw_boxes(image: np.ndarray, boxes: Sequence[YoloBox], thickness: int = 2) -> np.ndarray:
    """Return a copy of ``image`` with boxes and class names drawn on it."""
    out = image.copy()
    h, w = out.shape[:2]
    for box in boxes:
        x1 = round((box.x_center - box.width / 2) * w)
        y1 = round((box.y_center - box.height / 2) * h)
        x2 = round((box.x_center + box.width / 2) * w)
        y2 = round((box.y_center + box.height / 2) * h)
        color = CLASS_COLORS[box.class_id]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(
            out,
            box.class_name,
            (x1, max(12, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    return out


def _label(image: np.ndarray, text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
    """Text on a filled background, placed above ``(x, y)`` or below it near the top edge."""
    font, scale = cv2.FONT_HERSHEY_SIMPLEX, 0.45
    (tw, th), base = cv2.getTextSize(text, font, scale, 1)
    top = y - th - base - 2 if y - th - base - 2 >= 0 else y
    cv2.rectangle(image, (x, top), (x + tw + 4, top + th + base + 2), color, -1)
    cv2.putText(image, text, (x + 2, top + th + 1), font, scale, (0, 0, 0), 1, cv2.LINE_AA)


def draw_detections(
    image: np.ndarray,
    detections: Sequence[Detection],
    names: Sequence[str],
    caption: str | None = None,
    min_side: int = 8,
) -> np.ndarray:
    """Return a copy of ``image`` with boxes labelled ``class conf #track``.

    Boxes smaller than ``min_side`` pixels are drawn enlarged around their centre, so drones
    that are only a few pixels wide stay visible; the JSON keeps the real coordinates.
    """
    out = image.copy()
    for det in detections:
        x1, y1, x2, y2 = det.bbox_xyxy
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        hw, hh = max(x2 - x1, min_side) / 2, max(y2 - y1, min_side) / 2
        p1, p2 = (round(cx - hw), round(cy - hh)), (round(cx + hw), round(cy + hh))
        color = CLASS_COLORS[det.class_id % len(CLASS_COLORS)]
        cv2.rectangle(out, p1, p2, color, 2)
        text = f"{names[det.class_id]} {det.confidence:.2f}"
        if det.track_id is not None:
            text += f" #{det.track_id}"
        _label(out, text, p1[0], p1[1], color)
    if caption:
        _label(out, caption, 0, 0, (255, 255, 255))
    return out


def contact_sheet(images: Sequence[np.ndarray], cols: int = 4, tile_width: int = 480) -> np.ndarray:
    """Tile images (resized to ``tile_width``, aspect kept) into one grid image."""
    if not images:
        raise ValueError("no images to tile")
    tiles = []
    for img in images:
        scale = tile_width / img.shape[1]
        tiles.append(cv2.resize(img, (tile_width, max(1, round(img.shape[0] * scale)))))
    tile_h = max(t.shape[0] for t in tiles)
    rows = -(-len(tiles) // cols)
    sheet = np.zeros((rows * tile_h, cols * tile_width, 3), dtype=np.uint8)
    for i, tile in enumerate(tiles):
        r, c = divmod(i, cols)
        sheet[r * tile_h : r * tile_h + tile.shape[0], c * tile_width : (c + 1) * tile_width] = tile
    return sheet
