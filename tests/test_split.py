import csv
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

from fpvdd.labels import YoloBox, write_label_file
from fpvdd.naming import frame_filename, video_id_from_frame
from fpvdd.paths import REPO_ROOT
from fpvdd.split import (
    LeakageError,
    assign_splits,
    assign_splits_stratified,
    background_ratio,
    check_no_leakage,
    group_by_video,
)

RATIOS = (0.7, 0.15, 0.15)


def _sizes(n_videos: int = 20) -> dict[str, int]:
    return {f"vid{i:02d}": 10 + (i * 37) % 90 for i in range(n_videos)}


def test_group_by_video() -> None:
    paths = [Path(frame_filename(v, i)) for v in ("b", "a_x") for i in range(3)]
    groups = group_by_video(paths)
    assert list(groups) == ["a_x", "b"]
    assert all(len(g) == 3 for g in groups.values())


def test_every_video_in_exactly_one_split() -> None:
    sizes = _sizes()
    assignment = assign_splits(sizes, RATIOS, seed=0)
    assert set(assignment) == set(sizes)
    assert set(assignment.values()) == {"train", "val", "test"}


def test_ratios_are_approximated() -> None:
    sizes = _sizes(40)
    assignment = assign_splits(sizes, RATIOS, seed=0)
    total = sum(sizes.values())
    for split, ratio in zip(("train", "val", "test"), RATIOS, strict=True):
        share = sum(n for v, n in sizes.items() if assignment[v] == split) / total
        assert share == pytest.approx(ratio, abs=0.05)


def test_deterministic_for_seed() -> None:
    sizes = {f"v{i}": 10 for i in range(12)}  # all ties, so the seed decides
    assert assign_splits(sizes, RATIOS, seed=1) == assign_splits(sizes, RATIOS, seed=1)
    assert assign_splits(sizes, RATIOS, seed=1) != assign_splits(sizes, RATIOS, seed=2)


def test_no_empty_split_with_one_huge_video() -> None:
    sizes = {"huge": 1000, "a": 1, "b": 1}
    assignment = assign_splits(sizes, RATIOS)
    assert sorted(assignment.values()) == ["test", "train", "val"]


def test_zero_ratio_split_stays_empty() -> None:
    assignment = assign_splits(_sizes(5), (0.8, 0.2, 0.0))
    assert "test" not in assignment.values()


@pytest.mark.parametrize("ratios", [(0.7, 0.2, 0.2), (0.5, 0.5), (1.2, -0.1, -0.1)])
def test_invalid_ratios(ratios: tuple[float, ...]) -> None:
    with pytest.raises(ValueError):
        assign_splits(_sizes(), ratios)


def test_too_few_videos() -> None:
    with pytest.raises(ValueError, match="at least 3 videos"):
        assign_splits({"a": 5, "b": 5}, RATIOS)


def test_stratified_balances_each_stratum() -> None:
    sizes = {f"d{i}": 50 for i in range(20)} | {f"b{i}": 50 for i in range(20)}
    strata = {vid: "drone" if vid.startswith("d") else "bird" for vid in sizes}
    assignment = assign_splits_stratified(sizes, strata, RATIOS, seed=0)
    for stratum in ("drone", "bird"):
        counts = Counter(s for v, s in assignment.items() if strata[v] == stratum)
        assert counts == {"train": 14, "val": 3, "test": 3}


def test_stratified_pools_small_strata() -> None:
    sizes = {f"d{i}": 10 for i in range(10)} | {"rare1": 10, "rare2": 10}
    strata = {v: "rare" if v.startswith("rare") else "drone" for v in sizes}
    assignment = assign_splits_stratified(sizes, strata, RATIOS, seed=0)
    assert set(assignment) == set(sizes)
    assert set(assignment.values()) == {"train", "val", "test"}


def test_stratified_too_few_videos() -> None:
    with pytest.raises(ValueError):
        assign_splits_stratified({"a": 1, "b": 1}, {"a": "x", "b": "y"}, RATIOS)


def test_check_no_leakage() -> None:
    ok = {"train": [Path("a_f000001.jpg")], "val": [Path("b_f000001.jpg")]}
    check_no_leakage(ok)
    bad = {"train": [Path("a_f000001.jpg")], "val": [Path("a_f000002.jpg")]}
    with pytest.raises(LeakageError):
        check_no_leakage(bad)


def test_background_ratio() -> None:
    box = YoloBox(0, 0.5, 0.5, 0.1, 0.1)
    assert background_ratio([[], [box], [], [box]]) == 0.5
    assert background_ratio([]) == 0.0


def test_cli_end_to_end(tmp_path: Path) -> None:
    labeled, out = tmp_path / "labeled", tmp_path / "splits"
    box = YoloBox(0, 0.5, 0.5, 0.1, 0.1)
    for v in range(6):
        for i in range(10):
            img = labeled / "images" / f"v{v}" / frame_filename(f"v{v}", i)
            img.parent.mkdir(parents=True, exist_ok=True)
            img.write_bytes(b"")  # content is irrelevant for splitting
            lbl = labeled / "labels" / f"v{v}" / f"{img.stem}.txt"
            write_label_file(lbl, [] if i == 0 else [box])

    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")}
    cmd = [sys.executable, str(REPO_ROOT / "scripts" / "split_dataset.py")]
    args = ["--input", str(labeled), "--output", str(out)]
    result = subprocess.run(cmd + args, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

    seen: dict[str, str] = {}
    for split in ("train", "val", "test"):
        lines = (out / f"{split}.txt").read_text(encoding="utf-8").splitlines()
        assert lines, split
        for line in lines:
            assert Path(line).is_absolute()
            assert seen.setdefault(video_id_from_frame(line), split) == split
    with (out / "split_manifest.csv").open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 6
    assert all(row["n_background"] == "1" for row in rows)

    bad = subprocess.run(cmd + args + ["--by", "frame"], env=env, capture_output=True, text=True)
    assert bad.returncode != 0
