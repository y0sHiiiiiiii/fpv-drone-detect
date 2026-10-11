# Roadmap

Working plan for fpv-drone-detect. Update the status, decisions and open questions here whenever a
phase changes. Scope rules from `CLAUDE.md` apply to every phase.

Status: ✅ done · 🚧 in progress · ⏳ planned

| Phase | Topic | Status |
|---|---|---|
| 1 | Scaffolding + data prep | ✅ |
| 2 | Public dataset ingestion | ✅ (Halmstad) |
| 3 | Baseline training | ⏳ next |
| 4 | Evaluation | ⏳ |
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

## Phase 3: Baseline training ⏳

- `scripts/train.py`: thin wrapper around the Ultralytics trainer, reads `configs/train.yaml`,
  CLI overrides, optional W&B/MLflow
- baseline: yolo11s, `imgsz=1280`, 100 epochs on the RTX 4070 SUPER (12 GB)
- record the run config + dataset manifest hash with each run

## Phase 4: Evaluation ⏳

- `scripts/evaluate.py --weights … --split test`
- mAP@0.5, mAP@0.5:0.95, per-class precision/recall, confusion matrix (watch multirotor ↔ bird)
- metrics by object size (COCO area ranges small/medium/large) via pycocotools
- results to `runs/`, summary table in README on request

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

## Open questions

- More datasets later? Candidates: Drone-vs-Bird (on request, birds unlabeled), Anti-UAV300
  (license unclear), MAV-VID / Roboflow via a generic YOLO import.
- Phase 3: domain gap to FPV footage. A small hand-labeled FPV test set would show how well
  Halmstad-only training transfers.
