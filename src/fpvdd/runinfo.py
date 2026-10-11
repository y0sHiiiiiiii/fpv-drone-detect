"""Provenance for training runs: dataset fingerprint, git state, environment."""

import hashlib
import platform
import subprocess
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from fpvdd.paths import REPO_ROOT


def label_path_for_image(image: Path) -> Path:
    """Ultralytics' rule: replace the last ``images`` path component with ``labels``."""
    parts = list(image.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            return Path(*parts).with_suffix(".txt")
    raise ValueError(f"image path has no 'images' component: {image}")


def read_split_list(path: Path) -> list[Path]:
    """Image paths listed in an Ultralytics split file (one per line)."""
    return [Path(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def dataset_fingerprint(split_lists: Mapping[str, Path]) -> dict[str, Any]:
    """Hash split membership and label contents, so any change to the dataset changes the hash.

    Image pixels are not hashed (too slow); images are identified by their path relative to
    the common ``images`` root, so moving the dataset folder does not change the hash.
    """
    digest = hashlib.sha256()
    counts: dict[str, int] = {}
    for split in sorted(split_lists):
        images = sorted(read_split_list(split_lists[split]))
        counts[split] = len(images)
        digest.update(f"[{split}]\n".encode())
        for image in images:
            label = label_path_for_image(image)
            digest.update(_relative_to_images(image).encode())
            digest.update(label.read_bytes() if label.exists() else b"<background>")
    return {"sha256": digest.hexdigest(), "images": counts}


def _relative_to_images(image: Path) -> str:
    parts = image.parts
    idx = len(parts) - 1 - parts[::-1].index("images")
    return "/".join(parts[idx + 1 :])


def git_state(repo: Path = REPO_ROOT) -> dict[str, Any]:
    """Current commit and whether the working tree has uncommitted changes."""

    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=repo, capture_output=True, text=True, check=True
        ).stdout.strip()

    try:
        return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def environment(packages: Iterable[str] = ("torch", "ultralytics", "numpy", "opencv-python")):
    """Python/platform/package versions and the GPU name if CUDA is available."""
    from importlib.metadata import PackageNotFoundError, version

    versions = {}
    for name in packages:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    env: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": versions,
        "gpu": None,
    }
    try:
        import torch

        if torch.cuda.is_available():
            env["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return env
