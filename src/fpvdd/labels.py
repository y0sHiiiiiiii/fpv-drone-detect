"""Parsing, validation and writing of YOLO detection label files."""

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

CLASS_NAMES: tuple[str, ...] = ("multirotor", "fixed_wing", "vtol", "bird", "unknown_aerial")
NUM_CLASSES = len(CLASS_NAMES)


class LabelError(ValueError):
    """A label file or line is malformed."""


@dataclass(frozen=True)
class YoloBox:
    """One box in normalized YOLO format (center x/y, width, height in [0, 1])."""

    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float

    def __post_init__(self) -> None:
        if not 0 <= self.class_id < NUM_CLASSES:
            raise LabelError(f"class id {self.class_id} out of range 0..{NUM_CLASSES - 1}")
        for name in ("x_center", "y_center"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise LabelError(f"{name}={getattr(self, name)} outside [0, 1]")
        for name in ("width", "height"):
            if not 0.0 < getattr(self, name) <= 1.0:
                raise LabelError(f"{name}={getattr(self, name)} outside (0, 1]")

    @property
    def class_name(self) -> str:
        return CLASS_NAMES[self.class_id]

    def to_line(self) -> str:
        return (
            f"{self.class_id} {self.x_center:.6f} {self.y_center:.6f} "
            f"{self.width:.6f} {self.height:.6f}"
        )


def parse_label_line(line: str) -> YoloBox:
    """Parse one ``class x y w h`` line."""
    parts = line.split()
    if len(parts) != 5:
        raise LabelError(f"expected 5 columns, got {len(parts)}: {line!r}")
    try:
        class_id = int(parts[0])
        x, y, w, h = (float(p) for p in parts[1:])
    except ValueError as exc:
        raise LabelError(f"non-numeric value in {line!r}") from exc
    return YoloBox(class_id, x, y, w, h)


def parse_label_file(path: Path) -> list[YoloBox]:
    """Parse a label file; a missing or empty file means a background frame (no boxes)."""
    if not path.exists():
        return []
    boxes = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            boxes.append(parse_label_line(line))
        except LabelError as exc:
            raise LabelError(f"{path}:{lineno}: {exc}") from None
    return boxes


def write_label_file(path: Path, boxes: Iterable[YoloBox]) -> None:
    """Write boxes in YOLO format; an empty iterable writes an empty (background) file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [box.to_line() for box in boxes]
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")


def is_background(path: Path) -> bool:
    """True if the label file is missing or contains no boxes."""
    return not parse_label_file(path)
