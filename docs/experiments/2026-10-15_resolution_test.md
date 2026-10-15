# Experiment: higher resolution and SAHI at inference only (old baseline weights)

Date: 2026-10-15 · Phase 6 · Weights: `runs/detect/baseline/weights/best.pt` (5 classes, trained at
imgsz 1280, see [the baseline write-up](2026-10-11_baseline.md))

The question: does the box size bias on tiny drones (see
[the video tracking write-up](2026-10-11_video_tracking.md)) shrink if the model simply sees more
pixels per object, with no retraining? If so, training at imgsz 1920 would be worth its cost.

## Setup

| | |
|---|---|
| Videos | the 44 test videos of the old 5-class split (`data/splits_5class_2026-10-11/split_manifest.csv`), every frame |
| Reference | `runs/infer/baseline_test` (imgsz 1280), re-evaluated with the current 7-class `evaluate_video.py` |
| Variant A | `--imgsz 1920`: the 640×512 frames are upscaled 3× instead of 2× |
| Variant B | `--sahi --slice 320`: 320 px slices, each upscaled to 1280 (4×), overlap 0.2 (the default) |
| Otherwise | as in the video tracking write-up: conf ≥ 0.1, NMS IoU 0.5, IoS 0.8, ByteTrack; matching at IoU ≥ 0.5 |
| Outputs (not committed) | `runs/infer/baseline_test_imgsz1920/`, `runs/infer/baseline_test_sahi320/`, `runs/infer/baseline_test/video_eval_7class/` |

```bash
M=data/splits_5class_2026-10-11/split_manifest.csv
python scripts/infer_video.py --weights runs/detect/baseline/weights/best.pt --source $SRC \
    --imgsz 1920 --output runs/infer/baseline_test_imgsz1920 --no-video
python scripts/infer_video.py --weights runs/detect/baseline/weights/best.pt --source $SRC \
    --sahi --slice 320 --output runs/infer/baseline_test_sahi320 --no-video
python scripts/evaluate_video.py --pred runs/infer/<run> --manifest $M
```

With the 7-class `evaluate_video.py`, airplane and helicopter boxes are targets that a 5-class model
can never hit, so only the drone and bird rows are meaningful. The reference drone and bird numbers
are identical to those in the video tracking write-up.

## Results

Drone clips (18, 5,462 labeled boxes):

| variant | P / R | MOTA | IDF1 | ID switches | tracks | false tracks | FP: localization / background | speed (median) |
|---|---|---|---|---|---|---|---|---|
| imgsz 1280 (reference) | **0.819 / 0.821** | **0.638** | **0.787** | 7 | 31 | 8 | 915 / 75 | 57 frames/s |
| imgsz 1920 | 0.747 / 0.821 | 0.541 | 0.750 | 8 | 43 | 19 | 878 / 642 | 39 frames/s |
| SAHI 320 | 0.722 / 0.763 | 0.450 | 0.653 | 108 | 53 | 23 | 1,156 / 387 | 9 frames/s |

Bird clips (8): P/R 0.893 / 0.704 (reference), 0.859 / 0.741 (1920), 0.778 / 0.645 (SAHI).

Box size bias on drones, as the median area ratio predicted / labeled by labeled box side
(best-overlapping multirotor box, IoU ≥ 0.1):

| variant | < 15 px (n ≈ 10) | 15–30 px | > 30 px | mean IoU |
|---|---|---|---|---|
| imgsz 1280 | 4.18 | 1.36 | 0.81 | 0.663 |
| imgsz 1920 | 2.41 | 1.05 | 0.73 | 0.675 |
| SAHI 320 | 1.61 | 0.91 | 0.61 | 0.632 |

## Findings

- **No gain, so no imgsz 1920 training for now.** Both variants are worse than the reference on
  drones: MOTA drops from 0.64 to 0.54 (1920) and 0.45 (SAHI). Bird recall rises slightly at 1920
  (0.70 → 0.74), but bird precision drops.
- **Upscaling shifts the bias instead of removing it.** All box sizes shrink by roughly the same
  factor, so tiny drones get boxes closer to the right size, but large ones get boxes that are too
  small (0.81 → 0.73 → 0.61). The mean IoU barely moves (0.663 → 0.675 at 1920). The model seems to
  predict sizes in input pixels around a typical size it learned at 1280; showing it bigger objects
  at test time just moves where the error lands.
- **New errors at the unfamiliar scale:**
  - at 1920, background false positives in the drone clips rise from 75 to 642, for example on
    empty sky next to the drone (`V_DRONE_029`) or at the tree line
  - large drones get split into two part-boxes (`V_DRONE_034`)
  - SAHI adds false positives at slice borders and on helicopter clips (399 background boxes), and
    the drone tracks break up (108 ID switches), because the same object is detected in overlapping
    slices with slightly different boxes
- SAHI does not pay off at 640×512, as expected, and costs 6× the time. It stays meant for 1080p
  and larger footage.

Caveat: this only tests scaling at inference. A model *trained* at 1920 sees the larger objects
during training, so these numbers do not rule out a gain from training at 1920. They only show no
gain for free.

## Next steps

1. The P2 head at imgsz 1280 (the ROADMAP said imgsz 1920 only if this test shows a clear gain; it
   does not).
2. Training at 1920 stays an option for later, mainly for footage with higher resolution than
   Halmstad.
