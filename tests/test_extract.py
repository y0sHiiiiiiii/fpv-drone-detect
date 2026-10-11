from pathlib import Path

import pytest

from fpvdd.naming import parse_frame_name
from fpvdd.video import extract_video, iter_frames, probe


def test_probe(make_video) -> None:
    info = probe(make_video(n_frames=30, fps=30.0))
    assert info.fps == pytest.approx(30.0)
    assert (info.width, info.height) == (64, 48)


@pytest.mark.parametrize(
    ("target_fps", "expected_indices"),
    [
        (None, list(range(30))),
        (60.0, list(range(30))),  # above source fps -> every frame
        (10.0, list(range(0, 30, 3))),
        (4.0, [0, 8, 15, 23]),  # non-integer step 7.5
    ],
)
def test_iter_frames_sampling(make_video, target_fps, expected_indices) -> None:
    frames = list(iter_frames(make_video(n_frames=30, fps=30.0), target_fps))
    assert [idx for idx, _, _ in frames] == expected_indices
    assert frames[1][1] == pytest.approx(expected_indices[1] / 30.0)
    assert frames[0][2].shape == (48, 64, 3)


def test_iter_frames_rejects_bad_fps(make_video) -> None:
    with pytest.raises(ValueError):
        list(iter_frames(make_video(), 0))


def test_open_missing_video(tmp_path: Path) -> None:
    with pytest.raises(OSError):
        probe(tmp_path / "nope.mp4")


def test_extract_video_writes_named_frames(make_video, tmp_path: Path) -> None:
    out = tmp_path / "frames" / "clip"
    n = extract_video(make_video(n_frames=30, fps=30.0), out, "clip", target_fps=5.0)
    files = sorted(out.glob("*.jpg"))
    assert n == len(files) == 5
    assert [parse_frame_name(f) for f in files] == [("clip", i) for i in range(0, 30, 6)]
