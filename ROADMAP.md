# Roadmap

Working plan for fpv-drone-detect. Update the status, decisions and open questions here whenever a
phase changes. Scope rules from `CLAUDE.md` apply to every phase.

Status: ✅ done · 🚧 in progress · ⏳ planned

| Phase | Topic | Status |
|---|---|---|
| 1 | Scaffolding + data prep | ✅ |
| 2 | Public dataset ingestion | 🚧 |
| 3 | Baseline training | ⏳ |
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

## Phase 2: Public dataset ingestion 🚧

Goal: bring public research datasets into the unified `data/labeled/` layout with our 5 classes.

Deliverables:
- converter per source format → YOLO labels + frames named by the convention
- class mapping per dataset in a config file (source class → our class or `drop`)
- `sources/sources.csv` row per dataset (url, license, date accessed, notes)
- label QA script: draws boxes on random samples for visual checks
- tests with small synthetic fixtures for every converter

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

## Open questions

- Phase 2: which datasets exactly, and how their classes map to ours
  (e.g. airplane/helicopter → `unknown_aerial` or drop).
