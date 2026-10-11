# Roadmap

Working plan for fpv-drone-detect. Update the status, decisions and open questions here whenever a
phase changes. Scope rules from `CLAUDE.md` apply to every phase.

Status: ✅ done · 🚧 in progress · ⏳ planned

| Phase | Topic | Status |
|---|---|---|
| 1 | Scaffolding + data prep | ✅ |
| 2 | Public dataset ingestion | ✅ (Halmstad) |
| 3 | Baseline training | ✅ |
| 4 | Evaluation | ✅ |
| 5 | Video inference + tracking | ⏳ |
| 6 | Error analysis + iteration | ⏳ |

---

## Phase 1: Scaffolding + data prep ✅

Goal: reproducible data-prep pipeline with tests.

Delivered:
- standalone git repo, `.gitignore`, `pyproject.toml` (ruff + pytest config), `requirements*.txt`
- `configs/dataset.yaml` (5 classes, split list files), `configs/train.yaml`, `sources/sources.csv`
- `src/fpvdd/`: `paths`, `log`, `naming`, `images`, `labels`, `video`, `dedupe`, `split`
- `scripts/extract_frames.py`, `scripts/dedupe_frames.py`, `scripts/split_dataset.py`
- tests for naming, labels, dedupe, split (incl. CLI end-to-end), extraction

Key conventions:
- Frame names are `<video_id>_fNNNNNN.<ext>`. The video id in the name is how splitting by video
  works, so every converter must produce names in this form.
- Labeled data layout: `data/labeled/images/**` + mirrored `data/labeled/labels/**`.
- Splits are list files (`data/splits/{train,val,test}.txt`) with absolute paths; nothing is copied.
- Dedupe never deletes; it moves files to `_duplicates/`.

## Phase 2: Public dataset ingestion ✅ (Halmstad)

Goal: bring public research datasets into the unified `data/labeled/` layout with our 5 classes.

Delivered:
- `src/fpvdd/matlab_gt.py`: MAT5 parser for MATLAB `groundTruth` label files (any number of
  boxes per frame); `src/fpvdd/halmstad.py` + `scripts/convert_halmstad.py`
- `src/fpvdd/sources.py` + `configs/sources/halmstad.yaml`: class mapping per label name
- `scripts/visualize_labels.py` (+ `src/fpvdd/draw.py`): annotated samples + contact sheets
- `split_dataset.py --stratify class` (default): balances each video's dominant class
- `sources/sources.csv` row for Halmstad; tests for parser, converter, mapping, QA script

Halmstad result (visible only, 5 fps, negatives at 1.5 fps):
- 285 clips → 10,610 frames; 6,098 `multirotor` and 3,652 `bird` boxes; 1,972 airplane/helicopter
  boxes dropped; ~18 % background in every split
- all 285 label files match their video frame count exactly; no box outside the image
- validated on the real dataset: identical to `mcos-decoder` on 36,368 single-box frames, and
  visual QA sheets checked for drones and birds

Known limitations of the data:
- small domain: three multirotor models, mostly sky backgrounds, static ground camera, 640×512.
  This is far from FPV footage, so expect a large domain gap.
- objects are often only a few pixels wide: use `imgsz=1280` and consider SAHI
- Halmstad labels only the clip's own class; other objects in a clip (e.g. distant birds) can be
  unlabeled

Rules:
- Datasets are downloaded manually into `data/raw/<dataset>/`; no download tools.
- Image-only datasets without a sequence id need a grouping strategy before they can be split
  by video; document the strategy per dataset.
- No new classes without asking the user.

## Phase 3: Baseline training ✅

Delivered:
- `scripts/train.py`: wrapper around the Ultralytics trainer; `configs/train.yaml` +
  `--set key=value` overrides, `--smoke`, `--dry-run`; W&B via Ultralytics' built-in integration
- `src/fpvdd/runinfo.py`: `fpvdd_run.json` per run with git commit, dataset fingerprint (split
  membership + label contents), resolved config, versions, GPU, final val metrics
- project-local Ultralytics settings (`runs/.ultralytics/`), weights in `runs/weights/`

Environment: `.venv` in the repo (excluded from Nextcloud), torch 2.14.1+cu126,
ultralytics 8.4.175, RTX 4070 SUPER. Autobatch picks batch 4–5 at `imgsz=1280`.
`data/` and `runs/` are in the Nextcloud exclude list.

Baseline run: `python scripts/train.py --name baseline` (yolo11s, imgsz 1280, 100 epochs,
6.1 h, 12:41–18:50 on 2026-10-11).
- validation mAP50 peaked at epoch 29 (0.894, mAP50-95 0.428) and fell to 0.816 by epoch 100.
  Validation box loss rose from about epoch 20 on, while training loss kept falling: overfitting on
  a small, homogeneous domain. `best.pt` is epoch 29.
- consequence for later runs: about 30–40 epochs are enough on Halmstad alone. More varied data
  matters more than more epochs.

## Phase 4: Evaluation ✅

Delivered:
- `scripts/evaluate.py`: one Ultralytics `val` pass (standard mAP) plus one predict pass that
  feeds pycocotools and our own operating-point matching
- `src/fpvdd/evaluation.py`: YOLO labels → COCO, COCO AP/AR per class × size, class-agnostic
  one-to-one matching for the confusion matrix, P/R at a fixed confidence
- `src/fpvdd/eval_report.py`: `report.md` (overall, per class, by size, confusion matrix,
  multirotor ↔ bird)
- provenance checks: split leakage, dataset fingerprint vs. the training run, unfinished training
- verified: COCO and Ultralytics mAP agree (0.771 vs 0.767 mAP50 on a dev subset)

Baseline on `test` (1,615 frames; full table in the README):
- mAP@0.5 0.777, mAP@0.5:0.95 0.347; multirotor AP50 0.802, bird AP50 0.751
- 74 % of the boxes are small (< 32² px)
- multirotor ↔ bird: 3 birds detected as multirotor, 0 multirotors detected as bird
- 709 of 723 multirotor false positives at conf 0.25 in drone clips are second or offset boxes on an
  already-found drone. With NMS IoU 0.5, precision goes from 0.52 to 0.74 (recall 0.87 → 0.84).

Fixed: Ultralytics loads a list source into memory as one batch and ignores `batch`. On the full
test set that was 23.7 GiB, which ran out of memory. `evaluate.py` now predicts in chunks.

## Phase 5: Video inference + tracking ⏳

- `scripts/infer_video.py --weights … --source … --tracker bytetrack.yaml [--sahi] --output out/`
- Ultralytics `model.track()` (ByteTrack / BoT-SORT); with `--sahi`, sliced detections are fed into
  Ultralytics' tracker
- outputs `<name>_annotated.mp4` and `<name>_detections.json` (format in `CLAUDE.md`)

## Phase 6: Error analysis + iteration ⏳

- `notebooks/error_analysis.ipynb`: FP/FN galleries, per-size and per-source breakdowns
- iterate: hard negatives (birds), background share, `imgsz`, model size, SAHI settings
- later: own videos annotated in CVAT

---

## Decisions log

- 2026-10-11: standalone git repo in `TrenchView/` (was untracked inside the parent `dev/` repo).
- 2026-10-11: start with public research datasets; own footage later.
- 2026-10-11: dedupe uses a 64-bit dHash, compared within a video against the last kept frame.
- 2026-10-11: first dataset is Halmstad only (CC0); visible videos only, IR optional via
  `--modality`.
- 2026-10-11: classes outside our schema (airplane, helicopter): drop the box, keep the frame as
  a negative. No new classes.
- 2026-10-11: `mcos-decoder` 0.1.0 rejected. It identifies frames by byte size, so a 2-box cell
  looks like an empty one, and it skips 3+ box cells (12 bird clips misaligned, e.g. 1 of 302
  frames). It was replaced by our own MAT5 parser (needs only scipy).
- 2026-10-11: `negative_fps: 1.5` for airplane/helicopter clips, so background stays at 10–20 %.
- 2026-10-11: splits are stratified by dominant class per video (still strictly by video).
- 2026-10-11: no Ultralytics `fraction` for smoke runs. It takes the alphabetically first images,
  which here were only airplane clips (0 boxes), so smoke runs use random subsets instead.
- 2026-10-11: `workers: 2`. With the default 8, the train loader plus a 2× val loader started
  26 Python processes (~29 GB of 32 GB RAM) and the first baseline attempt was stopped after
  1.5 epochs. 2 workers measured equally fast (10 it/s) at ~12 GB peak.
- 2026-10-11: the confusion matrix and P/R come from our own matching at conf 0.25 / IoU 0.5.
  Ultralytics fills its matrix at the validation conf of 0.001, where low-confidence boxes swamp
  the false-positive column.
- 2026-10-11: no GPU evaluation during training. Both together overflowed the 12 GB of VRAM
  and predict stalled.
- 2026-10-11: Ultralytics settings are project-local (`YOLO_CONFIG_DIR=runs/.ultralytics`). The
  global ones pointed to another project's weights/runs folders.
- 2026-10-11: benchmark mAP keeps Ultralytics' default NMS IoU 0.7, so it stays comparable.
  Phase 5 video inference should default to NMS IoU 0.5, which removes most duplicate boxes on
  small drones.

## Open questions

- More datasets later? Candidates: Drone-vs-Bird (on request, birds unlabeled), Anti-UAV300
  (license unclear), MAV-VID / Roboflow via a generic YOLO import.
- Phase 3: domain gap to FPV footage. A small hand-labeled FPV test set would show how well
  Halmstad-only training transfers.
