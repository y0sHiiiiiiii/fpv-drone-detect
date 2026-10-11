"""Drawing YOLO boxes on images for visual label checks."""

from collections.abc import Sequence

import cv2
import numpy as np

from fpvdd.labels import YoloBox

# BGR, one colour per class id
CLASS_COLORS: tuple[tuple[int, int, int], ...] = (
    (60, 60, 230),  # multirotor
    (230, 140, 40),  # fixed_wing
    (40, 200, 230),  # vtol
    (80, 200, 80),  # bird
    (200, 80, 200),  # unknown_aerial
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
