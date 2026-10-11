import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from fpvdd.labels import YoloBox, write_label_file
from fpvdd.paths import REPO_ROOT
from fpvdd.runinfo import dataset_fingerprint, git_state, label_path_for_image

SCRIPT = REPO_ROOT / "scripts" / "train.py"


def _dataset(root: Path, n: int = 3) -> dict[str, Path]:
    """Tiny labeled dataset with split lists; returns the split list paths."""
    splits = {}
    for split in ("train", "val"):
        lines = []
        for i in range(n):
            img = root / "labeled" / "images" / "ds" / f"{split}{i}_f000000.jpg"
            img.parent.mkdir(parents=True, exist_ok=True)
            img.write_bytes(b"")
            write_label_file(
                root / "labeled" / "labels" / "ds" / f"{img.stem}.txt",
                [YoloBox(0, 0.5, 0.5, 0.1, 0.1)] if i else [],
            )
            lines.append(str(img))
        splits[split] = root / "splits" / f"{split}.txt"
        splits[split].parent.mkdir(parents=True, exist_ok=True)
        splits[split].write_text("\n".join(lines) + "\n", encoding="utf-8")
    return splits


def test_label_path_for_image_uses_last_images_component() -> None:
    img = Path("/x/images/data/labeled/images/ds/a_f000001.jpg")
    assert label_path_for_image(img) == Path("/x/images/data/labeled/labels/ds/a_f000001.txt")
    with pytest.raises(ValueError):
        label_path_for_image(Path("/x/frames/a.jpg"))


def test_fingerprint_changes_with_labels_not_location(tmp_path: Path) -> None:
    a = dataset_fingerprint(_dataset(tmp_path / "a"))
    b = dataset_fingerprint(_dataset(tmp_path / "b"))  # same content, other folder
    assert a == b
    assert a["images"] == {"train": 3, "val": 3}

    label = tmp_path / "b" / "labeled" / "labels" / "ds" / "train1_f000000.txt"
    write_label_file(label, [YoloBox(3, 0.5, 0.5, 0.1, 0.1)])
    assert dataset_fingerprint(_dataset_lists(tmp_path / "b"))["sha256"] != a["sha256"]


def _dataset_lists(root: Path) -> dict[str, Path]:
    return {s: root / "splits" / f"{s}.txt" for s in ("train", "val")}


def test_git_state_outside_repo(tmp_path: Path) -> None:
    assert git_state(tmp_path) == {"commit": None, "dirty": None}


def _run(*args: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")}
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args], env=env, capture_output=True, text=True
    )


def test_dry_run_resolves_config(tmp_path: Path) -> None:
    _dataset(tmp_path)
    data_yaml = tmp_path / "dataset.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(tmp_path),
                "train": "splits/train.txt",
                "val": "splits/val.txt",
                "names": {0: "multirotor"},
            }
        ),
        encoding="utf-8",
    )
    result = _run("--data", str(data_yaml), "--smoke", "--set", "imgsz=320", "--dry-run")
    assert result.returncode == 0, result.stderr
    assert "'epochs': 1" in result.stderr
    assert "'imgsz': 320" in result.stderr
    assert "'name': 'smoke'" in result.stderr
    resolved = yaml.safe_load((tmp_path / "splits" / "dataset.smoke.resolved.yaml").read_text())
    assert resolved["path"] == str(tmp_path)
    # smoke runs use random subsets, never Ultralytics' `fraction` (alphabetical prefix)
    assert Path(resolved["train"]).name == "smoke_train.txt"
    assert len(Path(resolved["train"]).read_text().splitlines()) == 1  # 5 % of 3, at least 1
    assert "fraction" not in result.stderr


def test_sample_split_is_random_and_deterministic(tmp_path: Path) -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location("train_script", SCRIPT)
    train = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train)

    src = tmp_path / "all.txt"
    src.write_text("".join(f"/d/images/c{i:03d}_f000000.jpg\n" for i in range(200)))
    a = train.sample_split(src, 0.1, 0, tmp_path / "a.txt").read_text().splitlines()
    b = train.sample_split(src, 0.1, 0, tmp_path / "b.txt").read_text().splitlines()
    assert a == b
    assert len(a) == 20
    assert a != [f"/d/images/c{i:03d}_f000000.jpg" for i in range(20)]  # not just a prefix


def test_missing_splits_fail_early(tmp_path: Path) -> None:
    data_yaml = tmp_path / "dataset.yaml"
    data_yaml.write_text(
        yaml.safe_dump({"path": str(tmp_path), "train": "nope.txt", "val": "nope.txt"}),
        encoding="utf-8",
    )
    result = _run("--data", str(data_yaml), "--dry-run")
    assert result.returncode == 1
    assert "split_dataset.py" in result.stderr


def test_bad_override_rejected() -> None:
    result = _run("--set", "epochs", "--dry-run")
    assert result.returncode == 1
    assert "key=value" in result.stderr


def test_run_info_is_json_serialisable() -> None:
    # environment() must not return anything json.dumps chokes on
    from fpvdd.runinfo import environment

    json.dumps(environment())
