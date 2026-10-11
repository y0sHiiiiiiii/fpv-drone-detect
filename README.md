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
python scripts/train.py --smoke                 # 1 epoch on 5 % of the data, checks the setup
python scripts/train.py --name baseline         # configs/train.yaml: yolo11s, imgsz 1280, 100 epochs
python scripts/train.py --set epochs=50 model=yolo11m.pt --name m50   # any Ultralytics override
python scripts/train.py --resume runs/detect/baseline   # continue an interrupted run
```

On the RTX 4070 SUPER the baseline runs at ~40 images/s (batch 4–5 at `imgsz=1280`, the first
epoch is slower while Windows starts the dataloader workers): ~3.5 min per epoch, ~6 h for 100
epochs. Progress per epoch is in `runs/detect/<name>/results.csv`.

`train.py` wraps the Ultralytics trainer. It resolves all paths against the repo root, checks the
split lists first, and keeps downloaded weights in `runs/weights/`. It uses project-local
Ultralytics settings in `runs/.ultralytics/`, so your global Ultralytics settings (e.g. another
project's `weights_dir`) are neither used nor changed. `--smoke` trains on random subsets: Ultralytics'
`fraction` takes the alphabetically first images, which here are whole clips of one class. Each run directory gets an
`fpvdd_run.json` with the git commit, a dataset fingerprint (hash of split membership and label
contents), the resolved config, package versions, GPU and the final validation metrics.
W&B logging is Ultralytics' built-in integration: `pip install wandb` and `yolo settings wandb=True`.

Plain Ultralytics works too, run from the repo root:
`yolo detect train cfg=configs/train.yaml`.

```bash
python scripts/evaluate.py --weights runs/detect/baseline/weights/best.pt --split test
python scripts/evaluate.py --weights ... --limit 200     # quick check on a random subset
```

`evaluate.py` writes `runs/eval/<run>_<split>/report.md` and `metrics.json` with:
- mAP@0.5 and mAP@0.5:0.95 from Ultralytics and from pycocotools
- per-class AP, plus precision/recall both at Ultralytics' best-F1 confidence and at a fixed
  operating point (`--op-conf 0.25`, IoU ≥ 0.5)
- AP per class and object size (COCO: small < 32², medium < 96², large, in original pixels)
- a confusion matrix at the operating point, with multirotor ↔ bird spelled out. Ultralytics'
  own matrix uses conf 0.001 and is dominated by low-confidence boxes.
- checks that no video appears in two splits and that the dataset matches the training run
- COCO ground truth and predictions as JSON for error analysis

Don't evaluate on the GPU while a training run uses it: two models at `imgsz=1280` overflow
the 12 GB VRAM and Windows falls back to very slow shared memory. Use `--device cpu` meanwhile.
`scripts/infer_video.py` is planned (see roadmap).

## Repository layout

```
configs/        dataset.yaml, train.yaml
data/           raw/, frames/, labeled/, splits/  (gitignored, never commit)
sources/        sources.csv  (url, license, date_accessed, notes)
configs/sources/  per-dataset class mapping and options
scripts/        extract_frames.py, dedupe_frames.py, split_dataset.py,
                convert_halmstad.py, visualize_labels.py, train.py, evaluate.py
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

Test split of Halmstad (1,615 frames from 44 videos never seen in training; 916 multirotor and
518 bird boxes). Halmstad is a static ground camera filming against mostly sky backgrounds, so
these numbers say little about FPV footage yet.

| Model | Data | imgsz | mAP@0.5 | mAP@0.5:0.95 | multirotor AP50 | bird AP50 | P / R multirotor @0.25 | P / R bird @0.25 |
|---|---|---|---|---|---|---|---|---|
| yolo11s (`baseline`, epoch 29) | Halmstad visible | 1280 | 0.777 | 0.347 | 0.802 | 0.751 | 0.52 / 0.87 | 0.81 / 0.73 |
| same weights, NMS IoU 0.5 | | 1280 | 0.781 | 0.334 | 0.794 | 0.767 | 0.74 / 0.84 | 0.84 / 0.73 |

mAP and AP values are Ultralytics'. P/R are measured at conf 0.25 and IoU ≥ 0.5. Report:
`runs/eval/baseline_test/report.md`.

- Almost all objects are small (1,061 of 1,434 boxes < 32² px). Small objects score AP50 0.78,
  medium ones 0.71.
- multirotor ↔ bird confusion is rare: 3 birds were detected as multirotor, and no multirotor as
  bird.
- Most multirotor false positives are not phantom detections. In 709 of 723 cases the model drew a
  second box on a drone it had already found, or a box that was offset on the tiny object. With
  NMS IoU 0.5 instead of 0.7, multirotor precision at conf 0.25 rises from 0.52 to 0.74.
- Validation mAP peaked at epoch 29 (mAP50 0.894) and then declined while the training loss kept
  falling. That is overfitting on a small domain, and `best.pt` keeps epoch 29. Training took
  6.1 h for 100 epochs.
