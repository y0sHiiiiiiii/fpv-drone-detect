from pathlib import Path

import pytest

from fpvdd.labels import CLASS_NAMES, YoloBox
from fpvdd.paths import CONFIGS_DIR
from fpvdd.sources import load_source_config, parse_class_map


def test_parse_class_map() -> None:
    mapping = parse_class_map({"Drone": "multirotor", "airplane": None})
    assert mapping == {"drone": 0, "airplane": None}


def test_unknown_target_class_rejected() -> None:
    with pytest.raises(ValueError, match="unknown target class"):
        parse_class_map({"drone": "quadcopter"})


def test_map_class_is_case_insensitive_and_strict(tmp_path: Path) -> None:
    cfg = tmp_path / "x.yaml"
    cfg.write_text("name: x\nclass_map:\n  bird: bird\n", encoding="utf-8")
    config = load_source_config(cfg)
    assert config.map_class("BIRD") == CLASS_NAMES.index("bird")
    with pytest.raises(KeyError):
        config.map_class("helicopter")


def test_missing_keys_rejected(tmp_path: Path) -> None:
    cfg = tmp_path / "x.yaml"
    cfg.write_text("name: x\n", encoding="utf-8")
    with pytest.raises(ValueError, match="class_map"):
        load_source_config(cfg)


@pytest.mark.parametrize("path", sorted((CONFIGS_DIR / "sources").glob("*.yaml")))
def test_shipped_source_configs_are_valid(path: Path) -> None:
    load_source_config(path)


def test_from_pixels() -> None:
    box = YoloBox.from_pixels(0, 10, 20, 30, 40, 100, 200)
    assert box == YoloBox(0, 0.25, 0.2, 0.3, 0.2)


def test_from_pixels_clips_to_image() -> None:
    box = YoloBox.from_pixels(0, -10, -10, 30, 30, 100, 100)
    assert box == pytest.approx(YoloBox(0, 0.1, 0.1, 0.2, 0.2))
    assert YoloBox.from_pixels(0, 120, 10, 5, 5, 100, 100) is None
