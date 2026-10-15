from pathlib import Path

import pytest
import yaml

from fpvdd.labels import (
    CLASS_NAMES,
    LabelError,
    YoloBox,
    is_background,
    parse_label_file,
    parse_label_line,
    write_label_file,
)
from fpvdd.paths import DATASET_YAML


def test_class_names_match_dataset_yaml() -> None:
    names = yaml.safe_load(DATASET_YAML.read_text(encoding="utf-8"))["names"]
    assert tuple(names[i] for i in sorted(names)) == CLASS_NAMES


def test_parse_valid_line() -> None:
    box = parse_label_line("3 0.5 0.25 0.1 0.2")
    assert box == YoloBox(3, 0.5, 0.25, 0.1, 0.2)
    assert box.class_name == "bird"


@pytest.mark.parametrize(
    "line",
    [
        "0 0.5 0.5 0.1",  # too few columns
        "0 0.5 0.5 0.1 0.1 0.9",  # too many (segmentation/confidence)
        "a 0.5 0.5 0.1 0.1",  # non-numeric class
        "7 0.5 0.5 0.1 0.1",  # class id out of range
        "-1 0.5 0.5 0.1 0.1",
        "0 1.2 0.5 0.1 0.1",  # center outside image
        "0 0.5 0.5 0.0 0.1",  # zero width
        "0 0.5 0.5 0.1 1.5",
    ],
)
def test_parse_invalid_line(line: str) -> None:
    with pytest.raises(LabelError):
        parse_label_line(line)


def test_missing_and_empty_files_are_background(tmp_path: Path) -> None:
    assert is_background(tmp_path / "missing.txt")
    empty = tmp_path / "empty.txt"
    empty.write_text("\n\n", encoding="utf-8")
    assert is_background(empty)


def test_roundtrip_file(tmp_path: Path) -> None:
    boxes = [YoloBox(0, 0.5, 0.5, 0.2, 0.1), YoloBox(4, 0.1, 0.9, 0.05, 0.05)]
    path = tmp_path / "sub" / "x.txt"
    write_label_file(path, boxes)
    assert parse_label_file(path) == boxes
    assert not is_background(path)


def test_error_reports_file_and_line(tmp_path: Path) -> None:
    path = tmp_path / "bad.txt"
    path.write_text("0 0.5 0.5 0.1 0.1\n9 0.5 0.5 0.1 0.1\n", encoding="utf-8")
    with pytest.raises(LabelError, match=r"bad\.txt:2"):
        parse_label_file(path)
