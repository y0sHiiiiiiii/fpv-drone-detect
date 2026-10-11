"""Shared setup for scripts that drive Ultralytics: paths, dataset YAML, project-local settings."""

import json
import logging
import os
import random
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

import yaml

from fpvdd.paths import REPO_ROOT
from fpvdd.runinfo import read_split_list

log = logging.getLogger(__name__)

RUNS_DIR = REPO_ROOT / "runs"
WEIGHTS_DIR = RUNS_DIR / "weights"
# Project-local Ultralytics settings, so global settings (weights_dir, runs_dir of other
# projects) are neither used nor modified.
ULTRALYTICS_CONFIG_DIR = RUNS_DIR / ".ultralytics"


def resolve_path(value: str | Path, base: Path = REPO_ROOT) -> Path:
    """Absolute path; relative paths are taken relative to ``base`` (the repo root)."""
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def check_splits(splits: Mapping[str, Path], required: tuple[str, ...] = ("train", "val")) -> None:
    """Fail early with a helpful message if split lists are missing or empty."""
    for name in required:
        path = splits.get(name)
        if path is None or not path.exists():
            raise FileNotFoundError(f"{name} split list missing ({path}); run split_dataset.py")
        if not read_split_list(path):
            raise ValueError(f"{name} split list {path} is empty")


def sample_split(path: Path, share: float, seed: int, out: Path) -> Path:
    """Write a random ``share`` of the images listed in ``path`` to ``out``."""
    images = read_split_list(path)
    return sample_split_n(path, max(1, round(len(images) * share)), seed, out)


def sample_split_n(path: Path, n: int, seed: int, out: Path) -> Path:
    """Write ``n`` random images (or all, if fewer) listed in ``path`` to ``out``."""
    images = read_split_list(path)
    picked = sorted(random.Random(seed).sample(images, min(n, len(images))))
    out.write_text("".join(f"{p}\n" for p in picked), encoding="utf-8")
    return out


def resolve_dataset(
    dataset_yaml: Path,
    required: tuple[str, ...] = ("train", "val"),
    replace: Mapping[str, Path] | None = None,
    suffix: str = "",
) -> tuple[Path, dict[str, Path]]:
    """Write a copy of the dataset YAML with an absolute ``path``; return it and the split lists.

    The dataset ``path`` is resolved relative to the repository root, so Ultralytics works from
    any working directory. ``replace`` swaps split lists (e.g. random subsets). The copy is
    written next to the first required split list as ``<stem><suffix>.resolved.yaml``.
    """
    cfg = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8"))
    root = resolve_path(cfg.get("path", "."))
    cfg["path"] = str(root)
    splits = {s: resolve_path(cfg[s], root) for s in ("train", "val", "test") if cfg.get(s)}
    check_splits(splits, required)
    for split, path in (replace or {}).items():
        splits[split] = path
        cfg[split] = str(path)
    resolved = splits[required[0]].parent / f"{dataset_yaml.stem}{suffix}.resolved.yaml"
    resolved.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return resolved, splits


def ensure_ultralytics_settings() -> None:
    """Point Ultralytics at project-local settings with our weights/runs folders.

    Ultralytics only honours YOLO_CONFIG_DIR if the folder exists, and reads ``weights_dir``
    into a constant at import time. So the folder is created first and the settings are
    written by a separate process before this process imports ultralytics. The user's global
    Ultralytics settings are never touched.
    """
    os.environ["YOLO_CONFIG_DIR"] = str(ULTRALYTICS_CONFIG_DIR)
    wanted = {"weights_dir": str(WEIGHTS_DIR), "runs_dir": str(RUNS_DIR)}
    settings_file = ULTRALYTICS_CONFIG_DIR / "Ultralytics" / "settings.json"
    if settings_file.exists():
        current = json.loads(settings_file.read_text(encoding="utf-8"))
        if all(current.get(k) == v for k, v in wanted.items()):
            return
    ULTRALYTICS_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    code = (
        "import json, sys; from ultralytics import settings; "
        "settings.update(json.loads(sys.argv[1]))"
    )
    subprocess.run(
        [sys.executable, "-c", code, json.dumps(wanted)],
        check=True,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )


def prepare_ultralytics() -> bool:
    """Project-local settings and working directory; call before importing ultralytics."""
    # All paths are absolute; work from the weights folder so nothing lands in the repo.
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    os.chdir(WEIGHTS_DIR)
    try:
        ensure_ultralytics_settings()
    except subprocess.CalledProcessError as exc:
        log.error("could not initialise Ultralytics settings: %s", exc.stderr)
        return False
    return True
