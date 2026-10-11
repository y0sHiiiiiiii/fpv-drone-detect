# fpv-drone-detect

Detect and classify drones in recorded FPV video with YOLO (Ultralytics). This is a learning and
research project in video object detection, and it does purely offline analysis:

video file → frame extraction → annotation → training → evaluation → inference on video →
annotated video + JSON output.

Out of scope: hardware integration, control/guidance/targeting, real-time deployment, identifying
specific military models or units, and geolocation. See `CLAUDE.md`.

## Classes

| ID | Name | Meaning |
|----|------|---------|
| 0 | `multirotor` | quad/hexa/octocopter, incl. FPV quads |
| 1 | `fixed_wing` | fixed-wing drones |
| 2 | `vtol` | hybrids with rotors and wings |
| 3 | `bird` | negative class against false positives |
| 4 | `unknown_aerial` | aerial object, not clearly assignable |

## Setup

Requires Python 3.11+ and FFmpeg.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows PowerShell: .venv\Scripts\Activate.ps1

# GPU: install a CUDA build of PyTorch first, or pip installs the CPU-only build
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

pip install -r requirements.txt -r requirements-dev.txt
```

If the repo lives in a synced folder (Nextcloud, Dropbox, …), exclude `.venv/`, `data/` and
`runs/` from syncing, or create the venv outside the repo.

## Data preparation

Put videos into `data/raw/` manually, or use public research datasets. There are no download tools.
Add every source to `sources/sources.csv` with its license.

```bash
# 1. sample frames -> data/frames/<video_id>/<video_id>_f000123.jpg
python scripts/extract_frames.py --input data/raw/ --output data/frames/ --fps 5

# 2. move near-duplicate consecutive frames to data/frames/_duplicates/ (try --dry-run first)
python scripts/dedupe_frames.py --input data/frames/ --threshold 0.95

# 3. annotate (CVAT / Label Studio), export YOLO to data/labeled/{images,labels}/

# 4. split by video -> data/splits/{train,val,test}.txt + split_manifest.csv
python scripts/split_dataset.py --by video --ratios 0.7 0.15 0.15
```

Frame names must follow `<video_id>_fNNNNNN.<ext>`, because the video id in the name is how
`split_dataset.py` keeps every video in a single split. By default it also stratifies by each
video's dominant class, so every split gets a similar class mix. It validates all labels and warns
if the training set's background share (frames without objects) falls outside 10–20 %.

### Public datasets

**Halmstad drone detection dataset** (CC0-1.0, [GitHub](https://github.com/DroneDetectionThesis/Drone-detection-dataset),
[Zenodo](https://doi.org/10.5281/zenodo.5500576)): download and unpack it to `data/raw/halmstad/`, then run

```bash
python scripts/convert_halmstad.py            # visible videos, 5 fps -> data/labeled/images/halmstad/...
python scripts/visualize_labels.py --subdir halmstad --class bird   # check boxes in data/qa/
python scripts/split_dataset.py
```

The class mapping lives in `configs/sources/halmstad.yaml`: drone → `multirotor`, bird → `bird`.
Airplane and helicopter boxes are dropped and their frames are kept as negatives, sampled at
`negative_fps` to keep the background share in range. The MATLAB `groundTruth` label files are
read by `fpvdd.matlab_gt`, a small MAT5 parser that handles any number of boxes per frame.
Result with the defaults: 285 clips → 10,610 frames, 6,098 multirotor and 3,652 bird boxes,
about 18 % background.

## Training and evaluation

```bash
yolo detect train data=configs/dataset.yaml model=yolo11s.pt imgsz=1280 epochs=100
# or with the defaults from configs/train.yaml:
yolo detect train cfg=configs/train.yaml
```

Run `yolo` commands from the repo root, because `configs/dataset.yaml` resolves `data/` from there.
`scripts/evaluate.py` and `scripts/infer_video.py` are planned (see roadmap).

## Repository layout

```
configs/        dataset.yaml, train.yaml
data/           raw/, frames/, labeled/, splits/  (gitignored, never commit)
sources/        sources.csv  (url, license, date_accessed, notes)
configs/sources/  per-dataset class mapping and options
scripts/        extract_frames.py, dedupe_frames.py, split_dataset.py,
                convert_halmstad.py, visualize_labels.py
src/fpvdd/      shared helpers (naming, labels, video, dedupe, split, sources,
                matlab_gt, halmstad, draw)
notebooks/      error analysis (planned)
tests/          pytest tests
```

## Checks

```bash
pytest
ruff check . && ruff format --check .
```

## Roadmap

Details, decisions and open questions are in [ROADMAP.md](ROADMAP.md).

1. ✅ Scaffolding and data prep (extract, dedupe, split by video) with tests
2. ✅ Public dataset ingestion (Halmstad) with class mapping and label QA visualisation
3. Baseline training (`scripts/train.py`, yolo11s @ 1280)
4. Evaluation: mAP@0.5 / 0.5:0.95, per-class P/R, confusion matrix, small/medium/large breakdown
5. Video inference with ByteTrack/BoT-SORT and optional SAHI → annotated MP4 + detections JSON
6. Error analysis notebook and iteration (hard negatives, background share, image size)

## Results

_No trained models yet._
