"""Train a YOLO detector with Ultralytics, using configs/train.yaml plus CLI overrides.

Writes <run dir>/fpvdd_run.json with git commit, dataset fingerprint, resolved config,
environment and the final validation metrics.

Examples:
    python scripts/train.py --name baseline
    python scripts/train.py --smoke
    python scripts/train.py --set epochs=50 imgsz=960 model=yolo11m.pt --name m960
"""

import argparse
import json
import logging
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from fpvdd.log import setup_logging
from fpvdd.paths import CONFIGS_DIR, REPO_ROOT
from fpvdd.runinfo import dataset_fingerprint, environment, git_state, read_split_list

log = logging.getLogger("train")

WEIGHTS_DIR = REPO_ROOT / "runs" / "weights"
# Project-local Ultralytics settings, so global settings (weights_dir, runs_dir of other
# projects) are neither used nor modified.
ULTRALYTICS_CONFIG_DIR = REPO_ROOT / "runs" / ".ultralytics"
SMOKE_OVERRIDES = {"epochs": 1, "plots": False}
# Ultralytics' `fraction` takes the alphabetically first images (whole clips of one class), so
# smoke runs use random subsets of the split lists instead.
SMOKE_SAMPLE = {"train": 0.05, "val": 0.2}


def parse_overrides(items: list[str]) -> dict[str, Any]:
    """Parse ``key=value`` strings; values are YAML-typed (``1`` -> int, ``true`` -> bool)."""
    overrides = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise ValueError(f"override must look like key=value, got {item!r}")
        overrides[key.strip()] = yaml.safe_load(value)
    return overrides


def resolve_path(value: str | Path, base: Path = REPO_ROOT) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def sample_split(path: Path, share: float, seed: int, out: Path) -> Path:
    """Write a random ``share`` of the images listed in ``path`` to ``out``."""
    images = read_split_list(path)
    k = max(1, round(len(images) * share))
    picked = sorted(random.Random(seed).sample(images, k))
    out.write_text("".join(f"{p}\n" for p in picked), encoding="utf-8")
    return out


def resolve_dataset(dataset_yaml: Path, smoke: bool = False) -> tuple[Path, dict[str, Path]]:
    """Write a copy of the dataset YAML with an absolute ``path``; return it and the split lists.

    The dataset ``path`` is resolved relative to the repository root, so training works from
    any working directory. The copy is written next to the train split list. With ``smoke``,
    train/val point to random subsets (see SMOKE_SAMPLE).
    """
    cfg = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8"))
    root = resolve_path(cfg.get("path", "."))
    cfg["path"] = str(root)
    splits = {s: resolve_path(cfg[s], root) for s in ("train", "val", "test") if cfg.get(s)}
    check_splits(splits)
    out_dir = splits["train"].parent
    if smoke:
        for split, share in SMOKE_SAMPLE.items():
            splits[split] = sample_split(splits[split], share, 0, out_dir / f"smoke_{split}.txt")
            cfg[split] = str(splits[split])
    suffix = ".smoke" if smoke else ""
    resolved = out_dir / f"{dataset_yaml.stem}{suffix}.resolved.yaml"
    resolved.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return resolved, splits


def check_splits(splits: dict[str, Path]) -> None:
    """Fail early with a helpful message if the split lists are missing or empty."""
    for name in ("train", "val"):
        path = splits.get(name)
        if path is None or not path.exists():
            raise FileNotFoundError(f"{name} split list missing ({path}); run split_dataset.py")
        if not read_split_list(path):
            raise ValueError(f"{name} split list {path} is empty")


def ensure_ultralytics_settings() -> None:
    """Point Ultralytics at project-local settings with our weights/runs folders.

    Ultralytics only honours YOLO_CONFIG_DIR if the folder exists, and reads ``weights_dir``
    into a constant at import time. So the folder is created first and the settings are
    written by a separate process before this process imports ultralytics. The user's global
    Ultralytics settings are never touched.
    """
    os.environ["YOLO_CONFIG_DIR"] = str(ULTRALYTICS_CONFIG_DIR)
    wanted = {"weights_dir": str(WEIGHTS_DIR), "runs_dir": str(REPO_ROOT / "runs")}
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


def resolve_model(model: str) -> str:
    """Keep pretrained weights in runs/weights/ instead of the working directory."""
    path = Path(model)
    if path.suffix == ".pt" and not path.is_absolute() and len(path.parts) == 1:
        if (REPO_ROOT / path).exists():
            return str(REPO_ROOT / path)
        from ultralytics.utils.downloads import attempt_download_asset

        WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        return str(attempt_download_asset(WEIGHTS_DIR / path.name))
    return str(resolve_path(model)) if path.suffix == ".pt" else model


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, default=CONFIGS_DIR / "train.yaml")
    parser.add_argument("--data", type=Path, default=None, help="dataset YAML (overrides config)")
    parser.add_argument("--name", default=None, help="run name below the project folder")
    parser.add_argument(
        "--set", nargs="*", default=[], metavar="KEY=VALUE", help="Ultralytics train overrides"
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="quick check: 1 epoch on a random 5 %% of train and 20 %% of val",
    )
    parser.add_argument("--dry-run", action="store_true", help="resolve and print config only")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> dict[str, Any]:
    cfg = yaml.safe_load(resolve_path(args.config).read_text(encoding="utf-8")) or {}
    if args.smoke:
        cfg.update(SMOKE_OVERRIDES)
    cfg.update(parse_overrides(args.set))
    if args.data:
        cfg["data"] = str(args.data)
    if args.name:
        cfg["name"] = args.name
    elif args.smoke:
        cfg["name"] = "smoke"
    cfg["project"] = str(resolve_path(cfg.get("project", "runs/detect")))
    return cfg


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(args.verbose)
    try:
        cfg = build_config(args)
        data_yaml, splits = resolve_dataset(
            resolve_path(cfg.get("data", "configs/dataset.yaml")), smoke=args.smoke
        )
    except (OSError, ValueError, KeyError) as exc:
        log.error("%s", exc)
        return 1
    cfg["data"] = str(data_yaml)

    fingerprint = dataset_fingerprint({s: p for s, p in splits.items() if p.exists()})
    log.info("dataset %s… %s", fingerprint["sha256"][:12], fingerprint["images"])
    log.info("config: %s", cfg)
    if args.dry_run:
        return 0

    # All paths above are absolute; work from the weights folder so nothing lands in the repo.
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    os.chdir(WEIGHTS_DIR)
    try:
        ensure_ultralytics_settings()
    except subprocess.CalledProcessError as exc:
        log.error("could not initialise Ultralytics settings: %s", exc.stderr)
        return 1

    from ultralytics import YOLO

    model_name = cfg.pop("model", "yolo11s.pt")
    model = YOLO(resolve_model(model_name))
    run_info: dict[str, Any] = {
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "argv": sys.argv[1:] if argv is None else argv,
        "model": model_name,
        "config": cfg,
        "dataset": fingerprint,
        "git": git_state(),
        "environment": environment(),
    }
    state: dict[str, Path] = {}

    def write_run_info(trainer) -> None:
        state["save_dir"] = Path(trainer.save_dir)
        (state["save_dir"] / "fpvdd_run.json").write_text(
            json.dumps(run_info, indent=2, default=str), encoding="utf-8"
        )

    model.add_callback("on_train_start", write_run_info)
    if run_info["git"]["dirty"]:
        log.warning("working tree has uncommitted changes; the run is not fully reproducible")

    results = model.train(**cfg)

    if "save_dir" in state:
        metrics = getattr(results, "results_dict", None) or {}
        run_info["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        run_info["final_val"] = {k: float(v) for k, v in metrics.items()}
        write_run_info(model.trainer)
        log.info("run dir: %s", state["save_dir"])
        for key in ("metrics/mAP50(B)", "metrics/mAP50-95(B)"):
            if key in run_info["final_val"]:
                log.info("%s = %.4f", key, run_info["final_val"][key])
    return 0


if __name__ == "__main__":
    sys.exit(main())
