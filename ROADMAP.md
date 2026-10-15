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
| 5 | Video inference + tracking | ✅ |
| 6 | Error analysis + iteration | 🚧 |

---

## Current state (2026-10-15)

Phases 1–5 are done. In phase 6, video evaluation (`evaluate_video.py`) is done and the main error
is identified: the model's **box size bias on tiny objects**. The centre is accurate to ~1 px, but
the predicted box area is 4.2× the labeled area below 15 px and 1.4× at 15–30 px. Details in
[docs/experiments/2026-10-11_video_tracking.md](docs/experiments/2026-10-11_video_tracking.md).

**7-class migration: done (2026-10-15).** `airplane` (5) and `helicopter` (6) are in
`CLASS_NAMES`, `configs/dataset.yaml`, the drawing colours, the Halmstad mapping and the tests.
Halmstad was reconverted and resplit:
- 285 clips → 14,984 frames + 2,029 background crops; 6,098 multirotor, 3,652 bird,
  3,121 airplane, 3,288 helicopter boxes
- background: only 13 sampled frames have no labeled object, so background images are now
  **crops from the frames** away from all labeled boxes (user decision, see the decisions log).
  Train 13.0 %, val 10.0 %, test 9.4 % background.
- splits (by video, stratified over 4 dominant classes): train 197 videos / 11,962 images,
  val 43 / 2,476, test 45 / 2,575. The test split differs from the 5-class one, so old and new
  test numbers are not directly comparable. The old split lists are kept in
  `data/splits_5class_2026-10-11/` for the old baseline.
- `negative_fps` no longer applies to Halmstad (no class maps to null); the option remains for
  other sources.
- `evaluate_video.py`: airplane/helicopter boxes are targets now; `other_object` only remains for
  labels that map to nothing.

**Running / next (experiment order decided by the user on 2026-10-11):**
1. 🚧 **new baseline** `baseline7`: yolo11s, imgsz 1280, 40 epochs (`configs/train.yaml` default
   is now 40). Then `evaluate.py` on test, `infer_video.py` + `evaluate_video.py` on the test
   videos, write-up, new README results table. Check whether the false tracks on aircraft are gone.
2. **resolution test with no training** on the old 5-class weights: test videos of the *old*
   split (`data/splits_5class_2026-10-11/split_manifest.csv`) at `--imgsz 1920` and with SAHI
   (`--sahi --slice 320`), then `evaluate_video.py` + the size-bias table. No GPU inference while
   training runs.
3. **P2 head**: our own `configs/models/yolo11-p2.yaml` (yolo11 backbone + stride-4 head as in
   Ultralytics' `yolov8-p2.yaml`) with `pretrained=yolo11s.pt`, same data and epochs as step 1.
   imgsz 1920 only if step 2 shows a clear gain.

**Planned experiments** (each: train, `evaluate.py` on test, `infer_video.py` + `evaluate_video.py`
on the 44 test videos, a write-up in `docs/experiments/`, a row in the README results table):
- common change: `epochs=40` (the baseline peaked at epoch 29 and then overfit), or `patience=15`
- (a) `imgsz=1920`: more pixels per tiny object. Check that autobatch still fits in 12 GB VRAM;
  run `--smoke` first.
- (b) P2 head (stride 4): Ultralytics 8.4.175 ships `yolov8-p2.yaml` and `yolo26-p2.yaml`, but no
  `yolo11-p2.yaml`. Write `configs/models/yolo11-p2.yaml` (yolo11 backbone + P2 head as in
  yolov8-p2) and train with `pretrained=yolo11s.pt`, so only the head differs from the baseline.
  Whether the weights load as expected (backbone and neck where shapes match) is not verified yet.
- (c) SAHI with upscaled tiles on the existing baseline weights (no training):
  `infer_video.py --sahi --slice 320` (tiles are fed at imgsz 1280, i.e. 4×). Only 8 frames/s.
- metrics to compare against: the baseline row in the README and the video tracking write-up
  (drones P/R 0.82 / 0.82, MOTA 0.64, IDF1 0.79 at IoU 0.5). Also report the size-bias table.

**Commands to reproduce the current numbers** (outputs in `runs/`, which is not committed):

```bash
# 44 test videos -> runs/infer/baseline_test (each video is <stem>.mp4 in Video_V/)
V=data/raw/halmstad/DroneDetectionThesis-Drone-detection-dataset-e7a6eaf/Data/Video_V
SRC=$(grep ",test," data/splits/split_manifest.csv | cut -d, -f1 | sed "s#halmstad_\(.*\)#$V/\1.mp4#")
python scripts/infer_video.py --weights runs/detect/baseline/weights/best.pt --source $SRC \
    --output runs/infer/baseline_test --no-video
python scripts/evaluate_video.py --pred runs/infer/baseline_test            # IoU 0.5
python scripts/evaluate_video.py --pred runs/infer/baseline_test --iou 0.3 \
    --output runs/infer/baseline_test/video_eval_iou03
```

Existing outputs: `runs/infer/baseline_test{,_ios0,_botsort}/` (each with `video_eval/`),
`runs/infer/dev*/` (annotated sample videos), `runs/eval/baseline_test{,_nms05}/`.

**Not in the repo yet** (ad-hoc analyses done in a scratch script, to be moved into
`notebooks/error_analysis.ipynb`):
- the size-bias measurement (predicted/labeled area by labeled box size, from
  `<pred>/*_detections.json` + `fpvdd.halmstad.video_ground_truth`, best-IoU prediction per frame)
- the containment suppression check on saved test predictions (`runs/eval/baseline_test_nms05/`
  `coco_predictions.json`, P/R at conf 0.25 for IoS 0.6–0.9)
- crops of predicted vs. labeled boxes for visual checks

Environment notes: `.venv` in the repo; `lap` was installed into it for the trackers (now in
`requirements.txt`). FFmpeg is on the PATH (needed for the H.264 output).

Afterwards (bigger topics, see open questions): more public datasets and a small FPV test set
labeled in CVAT, because all numbers so far come from Halmstad's ground camera.

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

Baseline on `test` (1,615 frames; table in the README, full write-up in
[docs/experiments/2026-10-11_baseline.md](docs/experiments/2026-10-11_baseline.md)):
- mAP@0.5 0.777, mAP@0.5:0.95 0.347; multirotor AP50 0.802, bird AP50 0.751
- 74 % of the boxes are small (< 32² px)
- multirotor ↔ bird: 3 birds detected as multirotor, 0 multirotors detected as bird
- 709 of 723 multirotor false positives at conf 0.25 in drone clips are second or offset boxes on an
  already-found drone. With NMS IoU 0.5, precision goes from 0.52 to 0.74 (recall 0.87 → 0.84).

Fixed: Ultralytics loads a list source into memory as one batch and ignores `batch`. On the full
test set that was 23.7 GiB, which ran out of memory. `evaluate.py` now predicts in chunks.

## Phase 5: Video inference + tracking ✅

Delivered:
- `scripts/infer_video.py`: its own frame loop, detection on the full frame or with SAHI slices
  (`--sahi`), then an Ultralytics tracker (ByteTrack / BoT-SORT from their YAML files, or
  `--tracker none`). Both detection modes use the same tracking path.
- outputs per video: `<name>_annotated.mp4` (H.264 via FFmpeg if available),
  `<name>_detections.json` (format in `CLAUDE.md`), `<name>_tracks.csv`; plus `infer_run.json`
- `src/fpvdd/inference.py`: conversions, JSON records, track summary, containment suppression;
  `draw.draw_detections`, `video.open_writer` / `transcode_h264`; tests incl. CLI end-to-end
- defaults: NMS IoU 0.5, containment suppression `--ios 0.8`, conf 0.1 with tracker
  (Ultralytics' track default), SAHI slices at the training `imgsz`

Result on the 44 test videos (full write-up in
[docs/experiments/2026-10-11_video_tracking.md](docs/experiments/2026-10-11_video_tracking.md)):
- 18 drone clips, one drone each: 31 tracks with ByteTrack + IoS 0.8 (45 without IoS);
  12 of 18 clips give exactly one track; per-frame P/R 0.82 / 0.82. BoT-SORT: same track
  count, P/R 0.83 / 0.83, slightly slower; ByteTrack stays the default
- bird clips: P/R 0.89 / 0.70; one airplane clip is tracked as `multirotor` for the whole clip
- ~55 frames/s at 640×512 without video output on the RTX 4070 SUPER

## Phase 6: Error analysis + iteration 🚧

Delivered so far:
- `scripts/evaluate_video.py` + `src/fpvdd/video_eval.py`: tracker output vs. per-frame labels;
  P/R/F1 per clip type and class, false positives by kind (wrong class, duplicate, localization,
  other object, background), MOTA, IDF1, ID switches, fragmentation, false tracks, tracks per
  object, split-leakage check. Identity metrics only where the labels identify objects
  (Halmstad: clips with ≤ 1 object of our classes per frame, 42 of 44 test clips).
- result on the test videos (ByteTrack, IoS 0.8; details in the video tracking write-up):
  - drones P/R 0.82 / 0.82, MOTA 0.64, IDF1 0.79 at IoU 0.5, but 0.97 / 0.98, MOTA 0.95,
    IDF1 0.94 at IoU 0.3. 915 of 990 drone FPs are boxes on the drone with IoU 0.1–0.5:
    localization of tiny objects is the main error, not detection.
  - bird recall stays at 0.78 even at IoU 0.3 (real misses)
  - false tracks on negative clips are real airplanes/helicopters, plus label gaps (an
    unlabeled speck) and one camera turn mid-clip

Next:
- `notebooks/error_analysis.ipynb`: FP/FN galleries, per-size and per-source breakdowns; include
  the FP-by-clip-type and duplicate-box analysis from the baseline write-up
- box size bias on tiny objects (checked: the model, not the labels). Predicted box area vs.
  labeled area is 4.2× below 15 px, 1.4× at 15–30 px and 0.8× above 30 px. Try a higher `imgsz`,
  SAHI with upscaled tiles, and a P2-head model (stride 4); compare with `evaluate_video.py`.
- aircraft false positives: decided, airplanes/helicopters get their own classes (see "Next
  session"). Check on the new baseline whether the false tracks on aircraft disappear.
- check why medium-sized birds score lower (AP50 0.62) than small ones
- iterate: hard negatives (birds), background share, `imgsz`, model size, SAHI settings
- training: ~40 epochs or `patience` ~15 on Halmstad alone (the baseline overfit after epoch 29)
- later: own videos annotated in CVAT

---

## Decisions log

- 2026-10-11: standalone git repo in `TrenchView/` (was untracked inside the parent `dev/` repo).
- 2026-10-11: start with public research datasets; own footage later.
- 2026-10-11: dedupe uses a 64-bit dHash, compared within a video against the last kept frame.
- 2026-10-11: first dataset is Halmstad only (CC0); visible videos only, IR optional via
  `--modality`.
- 2026-10-11: classes outside our schema (airplane, helicopter): drop the box, keep the frame as
  a negative. No new classes. (Superseded on 2026-10-11, see below.)
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
- 2026-10-11: `infer_video.py` runs its own frame loop and calls Ultralytics' trackers directly
  instead of `model.track()`. That way full-frame and SAHI detections go through the same
  tracking code, and the containment suppression can sit between detector and tracker.
- 2026-10-11: containment suppression (`--ios 0.8`, class-agnostic) is on by default in video
  inference, but not in `evaluate.py`. It raises precision at the operating point, but removes
  alternatives at low confidence, so COCO mAP drops (0.753 → 0.722 mAP50 on test), and the
  benchmark should stay comparable to Ultralytics.
- 2026-10-11: video metrics are implemented in the repo (`fpvdd.video_eval`) instead of using
  `motmetrics`, whose last release (1.4.0, 2022) predates numpy 2 / pandas 3. MOTA, IDF1, ID
  switches and fragmentation are small and have their own tests; HOTA is not implemented.
- 2026-10-11: `evaluate_video.py` matches at IoU 0.5 by default, like `evaluate.py`. The video
  write-up also shows IoU 0.3, because most errors on tiny objects are boxes slightly off.
- 2026-10-11 (user): new classes `airplane` (5) and `helicopter` (6). Halmstad airplanes and
  helicopters are labeled instead of dropped, so they can be reviewed and their treatment decided
  later. Reason: as background they became stable false `multirotor`/`bird` tracks. Existing ids
  0–4 stay unchanged.
- 2026-10-15 (user): background images for Halmstad come from crops of the frames. With all four
  label names mapped, only 13 of 14,984 sampled frames are empty (0.1 %). Every 6th sampled frame
  gives one crop (70/60/50 % of the frame side, ≥ 16 px from every labeled box, deterministic per
  frame); 448 frames had no room. The crops are named `<video_id>_fNNNNNN_bg.jpg` (frame names may
  carry a tag now), so they stay in their clip's split. Rejected: training without background, and
  waiting for another dataset. Known limit: crops may contain unlabeled objects, as full frames do.
- 2026-10-15: `configs/train.yaml` defaults to 40 epochs (the 100-epoch baseline peaked at 29).
- 2026-10-11 (user): experiment order: class migration + new baseline (40 epochs, imgsz 1280), a
  resolution test with no training on the old weights, then a P2 head (own `yolo11-p2` config);
  imgsz 1920 only if the resolution test shows a clear gain.

## Open questions

- More datasets later? Candidates: Drone-vs-Bird (on request, birds unlabeled), Anti-UAV300
  (license unclear), MAV-VID / Roboflow via a generic YOLO import.
- Phase 3: domain gap to FPV footage. A small hand-labeled FPV test set would show how well
  Halmstad-only training transfers.
