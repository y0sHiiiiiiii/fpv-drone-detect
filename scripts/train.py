"""Train a YOLO detector with Ultralytics, using configs/train.yaml plus CLI overrides.

Writes <run dir>/fpvdd_run.json with git commit, dataset fingerprint, resolved config,
environment and the final validation metrics.

Examples:
    python scripts/train.py --name baseline
    python scripts/train.py --smoke
    python scripts/train.py --set epochs=50 imgsz=960 model=yolo11m.pt --name m960
    python scripts/train.py --set model=configs/models/yolo11s-p2.yaml pretrained=yolo11s.pt \
        epochs=25 --name p2
    python scripts/train.py --resume runs/detect/baseline
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from fpvdd.log import setup_logging
from fpvdd.paths import CONFIGS_DIR, REPO_ROOT
from fpvdd.runinfo import dataset_fingerprint, environment, git_state
from fpvdd.yolo_env import (
    WEIGHTS_DIR,
    prepare_ultralytics,
    resolve_dataset,
    resolve_path,
    sample_split,
)

log = logging.getLogger("train")

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


def resolve_training_dataset(dataset_yaml: Path, smoke: bool) -> tuple[Path, dict[str, Path]]:
    """Resolved dataset YAML; with ``smoke``, train/val are random subsets (SMOKE_SAMPLE)."""
    resolved, splits = resolve_dataset(dataset_yaml)
    if not smoke:
        return resolved, splits
    out_dir = splits["train"].parent
    replace = {
        split: sample_split(splits[split], share, 0, out_dir / f"smoke_{split}.txt")
        for split, share in SMOKE_SAMPLE.items()
    }
    return resolve_dataset(dataset_yaml, replace=replace, suffix=".smoke")


def resolve_model(model: str) -> str:
    """Keep pretrained weights in runs/weights/ instead of the working directory.

    A bare ``.pt`` name is downloaded to runs/weights/; other ``.pt`` paths and model YAMLs with
    a directory part resolve against the repo root. A bare YAML name (``yolo11s.yaml``) stays as
    is, so Ultralytics finds its built-in config. For our own YAMLs the path may name a scale
    that has no file of its own (``configs/models/yolo11s-p2.yaml`` loads ``yolo11-p2.yaml``).
    """
    path = Path(model)
    if path.suffix == ".pt" and not path.is_absolute() and len(path.parts) == 1:
        if (REPO_ROOT / path).exists():
            return str(REPO_ROOT / path)
        from ultralytics.utils.downloads import attempt_download_asset

        WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        return str(attempt_download_asset(WEIGHTS_DIR / path.name))
    if path.suffix == ".pt" or (path.suffix in (".yaml", ".yml") and len(path.parts) > 1):
        return str(resolve_path(model))
    return model


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
    parser.add_argument(
        "--resume",
        type=Path,
        metavar="RUN_DIR",
        help="continue an interrupted run (e.g. runs/detect/baseline) from its last.pt",
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
    if args.resume:
        return resume(resolve_path(args.resume))
    try:
        cfg = build_config(args)
        data_yaml, splits = resolve_training_dataset(
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

    if not prepare_ultralytics():
        return 1
    from ultralytics import YOLO

    model_name = cfg.pop("model", "yolo11s.pt")
    model = YOLO(resolve_model(model_name))
    if isinstance(cfg.get("pretrained"), str) and cfg["pretrained"].endswith(".pt"):
        # weights to transfer into a model built from YAML (matching layers only)
        cfg["pretrained"] = resolve_model(cfg["pretrained"])
    run_info: dict[str, Any] = {
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "argv": sys.argv[1:] if argv is None else argv,
        "model": model_name,
        "config": cfg,
        "dataset": fingerprint,
        "git": git_state(),
        "environment": environment(),
    }
    if run_info["git"]["dirty"]:
        log.warning("working tree has uncommitted changes; the run is not fully reproducible")
    return train_with_run_info(model, run_info, **cfg)


def train_with_run_info(model, run_info: dict[str, Any], **train_args: Any) -> int:
    """Train, writing fpvdd_run.json when training starts and again with the final metrics."""
    state: dict[str, Path] = {}

    def write_run_info(trainer) -> None:
        state["save_dir"] = Path(trainer.save_dir)
        (state["save_dir"] / "fpvdd_run.json").write_text(
            json.dumps(run_info, indent=2, default=str), encoding="utf-8"
        )

    model.add_callback("on_train_start", write_run_info)
    results = model.train(**train_args)

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


def resume(run_dir: Path) -> int:
    """Continue an interrupted run from ``run_dir/weights/last.pt``."""
    last = run_dir / "weights" / "last.pt"
    if not last.exists():
        log.error("nothing to resume: %s not found", last)
        return 1
    info_file = run_dir / "fpvdd_run.json"
    run_info = json.loads(info_file.read_text(encoding="utf-8")) if info_file.exists() else {}
    run_info.setdefault("resumed", []).append(
        {"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "git": git_state()}
    )
    if not prepare_ultralytics():
        return 1
    from ultralytics import YOLO

    log.info("resuming %s", last)
    return train_with_run_info(YOLO(str(last)), run_info, resume=True)


if __name__ == "__main__":
    sys.exit(main())
