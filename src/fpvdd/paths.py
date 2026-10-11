"""Default locations, all relative to the repository root."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
FRAMES_DIR = DATA_DIR / "frames"
LABELED_DIR = DATA_DIR / "labeled"
SPLITS_DIR = DATA_DIR / "splits"

CONFIGS_DIR = REPO_ROOT / "configs"
DATASET_YAML = CONFIGS_DIR / "dataset.yaml"

IMAGE_EXTS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})
VIDEO_EXTS = frozenset({".mp4", ".mkv", ".webm", ".avi", ".mov"})
