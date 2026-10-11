from pathlib import Path

import numpy as np
import pytest

from fpvdd.dedupe import dhash, find_duplicates, move_duplicates, similarity
from fpvdd.images import read_image, write_image
from fpvdd.naming import frame_filename


def _pattern(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 256, size=(8, 9, 3), dtype=np.uint8)
    return np.kron(small, np.ones((40, 40, 1), dtype=np.uint8))  # 320x360 blocky image


def test_identical_and_noisy_images_are_similar() -> None:
    img = _pattern(0)
    rng = np.random.default_rng(1)
    noisy = np.clip(img.astype(int) + rng.integers(-3, 4, img.shape), 0, 255).astype(np.uint8)
    assert similarity(dhash(img), dhash(img)) == 1.0
    assert similarity(dhash(img), dhash(noisy)) >= 0.95


def test_different_images_are_dissimilar() -> None:
    assert similarity(dhash(_pattern(0)), dhash(_pattern(2))) < 0.8


def test_grayscale_input() -> None:
    gray = _pattern(0)[:, :, 0]
    assert isinstance(dhash(gray), int)


def test_find_duplicates_compares_to_last_kept() -> None:
    a, b = 0b0, (1 << 64) - 1
    hashes = {
        Path(frame_filename("v", 0)): a,
        Path(frame_filename("v", 5)): a,  # duplicate of 0
        Path(frame_filename("v", 10)): b,  # new scene, kept
        Path(frame_filename("v", 15)): a,  # differs from last kept (b) -> kept
    }
    assert find_duplicates(hashes, 0.95) == [Path(frame_filename("v", 5))]


def test_find_duplicates_never_crosses_videos() -> None:
    hashes = {Path(frame_filename("v1", 0)): 0, Path(frame_filename("v2", 0)): 0}
    assert find_duplicates(hashes, 0.95) == []


def test_frame_order_not_dict_order() -> None:
    hashes = {Path(frame_filename("v", 9)): 0, Path(frame_filename("v", 1)): 0}
    assert find_duplicates(hashes, 0.95) == [Path(frame_filename("v", 9))]


@pytest.mark.parametrize("threshold", [0.0, 1.5])
def test_invalid_threshold(threshold: float) -> None:
    with pytest.raises(ValueError):
        find_duplicates({}, threshold)


def test_end_to_end_on_files(tmp_path: Path) -> None:
    root = tmp_path / "frames"
    (root / "v").mkdir(parents=True)
    paths = [root / "v" / frame_filename("v", i, ".png") for i in range(3)]
    for path, img in zip(paths, [_pattern(0), _pattern(0), _pattern(3)], strict=True):
        write_image(path, img)
    hashes = {p: dhash(read_image(p)) for p in paths}
    dupes = find_duplicates(hashes, 0.95)
    assert dupes == [paths[1]]

    dest = tmp_path / "dupes"
    assert move_duplicates(dupes, root, dest) == 1
    assert not paths[1].exists()
    assert (dest / "v" / paths[1].name).exists()
