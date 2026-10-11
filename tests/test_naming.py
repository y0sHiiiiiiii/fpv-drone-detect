from pathlib import Path

import pytest

from fpvdd.naming import frame_filename, parse_frame_name, sanitize_video_id, video_id_from_frame


@pytest.mark.parametrize("video_id", ["clip", "my_clip_2024", "a-b_c_f12", "x"])
def test_roundtrip(video_id: str) -> None:
    name = frame_filename(video_id, 42)
    assert parse_frame_name(Path("some/dir") / name) == (video_id, 42)


def test_frame_index_is_zero_padded() -> None:
    assert frame_filename("v", 7) == "v_f000007.jpg"
    assert frame_filename("v", 1234567, ".png") == "v_f1234567.png"


def test_negative_index_rejected() -> None:
    with pytest.raises(ValueError):
        frame_filename("v", -1)


@pytest.mark.parametrize("name", ["frame_0001.jpg", "clip_f12.jpg", "_f000001.jpg", "clip.jpg"])
def test_non_conforming_names_rejected(name: str) -> None:
    with pytest.raises(ValueError):
        video_id_from_frame(name)


def test_sanitize_video_id() -> None:
    assert sanitize_video_id("My Clip (1).final") == "My_Clip_1_final"
    with pytest.raises(ValueError):
        sanitize_video_id("...")
