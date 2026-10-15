import random
from pathlib import Path

import pytest

from fpvdd.halmstad import (
    Clip,
    HalmstadError,
    background_crop,
    convert_clip,
    find_clips,
    frame_boxes,
    matlab_to_pixels,
    parse_clip_name,
    validate_labels,
)
from fpvdd.labels import CLASS_NAMES, parse_label_file
from fpvdd.naming import parse_frame_name
from fpvdd.paths import CONFIGS_DIR
from fpvdd.sources import load_source_config

MAP_CLASS = load_source_config(CONFIGS_DIR / "sources" / "halmstad.yaml").map_class


def test_parse_clip_name() -> None:
    assert parse_clip_name(Path("V_DRONE_001.mp4")) == ("V", "DRONE", 1)
    assert parse_clip_name(Path("IR_HELICOPTER_045.mp4")) == ("IR", "HELICOPTER", 45)
    with pytest.raises(ValueError):
        parse_clip_name(Path("Demo_V_DRONE_017.mp4"))


def test_find_clips_filters_modality_and_requires_labels(tmp_path: Path) -> None:
    for name in ["V_DRONE_001", "V_BIRD_002", "IR_DRONE_001", "V_AIRPLANE_003"]:
        (tmp_path / f"{name}.mp4").touch()
    for name in ["V_DRONE_001", "V_BIRD_002", "IR_DRONE_001"]:  # airplane has no labels
        (tmp_path / f"{name}_LABELS.mat").touch()
    (tmp_path / "Demo_V_DRONE_017.mp4").touch()

    clips = find_clips(tmp_path, ["V"])
    assert [c.video.stem for c in clips] == ["V_BIRD_002", "V_DRONE_001"]
    assert clips[1].video_id == "halmstad_V_DRONE_001"
    assert len(find_clips(tmp_path, ["V", "IR"])) == 3


def test_matlab_to_pixels() -> None:
    assert matlab_to_pixels((1.5, 10.5, 4.0, 3.0)) == (1.0, 10.0, 4.0, 3.0)


def test_validate_labels() -> None:
    frames = [{}, {"DRONE": [(10.0, 10.0, 5.0, 5.0)]}]
    validate_labels(frames, n_frames=2, width=64, height=48)
    with pytest.raises(HalmstadError, match="label frames"):
        validate_labels(frames, n_frames=3, width=64, height=48)
    with pytest.raises(HalmstadError, match="outside"):
        validate_labels([{"BIRD": [(60.0, 10.0, 20.0, 5.0)]}], n_frames=1, width=64, height=48)
    with pytest.raises(HalmstadError, match="non-positive"):
        validate_labels([{"BIRD": [(1.0, 1.0, 0.0, 5.0)]}], n_frames=1, width=64, height=48)


def test_frame_boxes_maps_per_box_and_drops_unmapped() -> None:
    labels = {
        "BIRD": [(1.5, 1.5, 4, 4), (11.5, 11.5, 4, 4)],
        "DRONE": [(21.5, 21.5, 8, 8)],
        "HELICOPTER": [(31.5, 1.5, 10, 10)],
    }
    boxes, dropped = frame_boxes(labels, MAP_CLASS, 64, 48)
    assert dropped == 0
    assert sorted(b.class_name for b in boxes) == ["bird", "bird", "helicopter", "multirotor"]

    no_heli = {"bird": 3, "drone": 0, "helicopter": None}
    boxes, dropped = frame_boxes(labels, lambda n: no_heli[n.lower()], 64, 48)
    assert dropped == 1
    assert sorted(b.class_name for b in boxes) == ["bird", "bird", "multirotor"]


def test_halmstad_maps_all_four_label_names() -> None:
    assert [CLASS_NAMES[MAP_CLASS(n)] for n in ("DRONE", "BIRD", "AIRPLANE", "HELICOPTER")] == [
        "multirotor",
        "bird",
        "airplane",
        "helicopter",
    ]


def test_frame_boxes_unknown_label_fails() -> None:
    with pytest.raises(KeyError):
        frame_boxes({"BALLOON": [(1, 1, 2, 2)]}, MAP_CLASS, 64, 48)


def _clip(video: Path) -> Clip:
    return Clip(video, video.with_name("x_LABELS.mat"), "V", "DRONE", 1)


def test_convert_clip_writes_frames_and_labels(make_video, tmp_path: Path) -> None:
    video = make_video("V_DRONE_001.avi", n_frames=30, fps=30.0)
    frames = [{"DRONE": [(10.5, 10.5, 8.0, 6.0)]} if i % 2 == 0 else {} for i in range(30)]
    images, labels = tmp_path / "out" / "images", tmp_path / "out" / "labels"

    stats = convert_clip(_clip(video), frames, MAP_CLASS, images, labels, target_fps=5.0)
    assert stats["frames"] == 5  # sampled frames 0, 6, 12, 18, 24 are all even
    assert stats["box:multirotor"] == 5
    assert stats["background"] == 0
    files = sorted((images / "halmstad" / "halmstad_V_DRONE_001").glob("*.jpg"))
    assert [parse_frame_name(f) for f in files] == [
        ("halmstad_V_DRONE_001", i) for i in range(0, 30, 6)
    ]
    label = labels / "halmstad" / "halmstad_V_DRONE_001" / f"{files[0].stem}.txt"
    (box,) = parse_label_file(label)
    assert box.class_id == CLASS_NAMES.index("multirotor")
    assert box.x_center == pytest.approx((10 + 4) / 64)
    assert box.width == pytest.approx(8 / 64)


def test_dropped_class_keeps_frames_as_background(make_video, tmp_path: Path) -> None:
    video = make_video("V_AIRPLANE_001.avi", n_frames=10, fps=10.0)
    frames = [{"AIRPLANE": [(10.0, 10.0, 8.0, 6.0)]}] * 10
    images, labels = tmp_path / "images", tmp_path / "labels"
    stats = convert_clip(_clip(video), frames, lambda _: None, images, labels, target_fps=None)
    assert (stats["frames"], stats["background"], stats["dropped"]) == (10, 10, 10)
    label_files = list((labels / "halmstad").rglob("*.txt"))
    assert len(label_files) == 10
    assert all(not parse_label_file(p) for p in label_files)


def test_background_crop_avoids_obstacles() -> None:
    rng = random.Random(0)
    obstacle = (280.0, 230.0, 20.0, 16.0)
    for _ in range(100):
        x, y, w, h = background_crop([obstacle], 640, 512, [0.7, 0.5], 16, rng)
        assert (w, h) in {(448, 358), (320, 256)}
        assert 0 <= x <= 640 - w and 0 <= y <= 512 - h
        ox, oy, ow, oh = obstacle
        assert x >= ox + ow + 16 or x + w <= ox - 16 or y >= oy + oh + 16 or y + h <= oy - 16


def test_background_crop_none_when_nothing_fits() -> None:
    full = [(0.0, 0.0, 640.0, 512.0)]
    assert background_crop(full, 640, 512, [0.5], 0, random.Random(0)) is None
    with pytest.raises(ValueError):
        background_crop([], 640, 512, [1.5], 0, random.Random(0))


def test_convert_clip_writes_background_crops(make_video, tmp_path: Path) -> None:
    video = make_video("V_DRONE_001.avi", n_frames=30, fps=30.0)
    frames = [{"DRONE": [(1.5, 1.5, 6.0, 6.0)]}] * 30  # top-left corner, room elsewhere
    images, labels = tmp_path / "images", tmp_path / "labels"
    stats = convert_clip(
        _clip(video),
        frames,
        MAP_CLASS,
        images,
        labels,
        5.0,
        bg_every=2,
        bg_sizes=[0.5],
        bg_margin=2,
    )
    assert (stats["frames"], stats["bg_crops"], stats["background"]) == (5, 2, 0)
    crops = sorted((images / "halmstad" / "halmstad_V_DRONE_001").glob("*_bg.jpg"))
    # samples 1 and 3 of frames 0, 6, 12, 18, 24 get a crop; the video id stays the clip's
    assert [parse_frame_name(c) for c in crops] == [
        ("halmstad_V_DRONE_001", 6),
        ("halmstad_V_DRONE_001", 18),
    ]
    for crop in crops:
        label = labels / "halmstad" / "halmstad_V_DRONE_001" / f"{crop.stem}.txt"
        assert label.exists() and not parse_label_file(label)
    again = convert_clip(
        _clip(video),
        frames,
        MAP_CLASS,
        tmp_path / "i2",
        tmp_path / "l2",
        5.0,
        bg_every=2,
        bg_sizes=[0.5],
        bg_margin=2,
    )
    assert again == stats


def test_convert_clip_rejects_mismatched_labels(make_video, tmp_path: Path) -> None:
    video = make_video("V_DRONE_002.avi", n_frames=30, fps=30.0)
    with pytest.raises(HalmstadError):
        convert_clip(_clip(video), [{}] * 5, MAP_CLASS, tmp_path / "i", tmp_path / "l", None)
