# Experiment: video inference and tracking (baseline weights, Halmstad test videos)

Date: 2026-10-11 · Phase 5 · Weights: `runs/detect/baseline/weights/best.pt` (epoch 29, see
[the baseline write-up](2026-10-11_baseline.md))

The question: does `infer_video.py` produce one clean track per object on video, and what do NMS
IoU, containment suppression and the tracker choice change?

## Setup

| | |
|---|---|
| Command | `python scripts/infer_video.py --weights runs/detect/baseline/weights/best.pt --source <44 test videos> --output runs/infer/baseline_test --no-video` |
| Videos | all 44 visible videos of the test split (never seen in training), every frame at the source frame rate, ~300 frames each (13,504 frames in total) |
| Detection | full frame, imgsz 1280, conf ≥ 0.1, NMS IoU 0.5 |
| Tracker | ByteTrack with Ultralytics' defaults (`bytetrack.yaml`) |
| Variants | `--ios 0` (no containment suppression), `--tracker botsort.yaml` |
| Outputs (not committed) | `runs/infer/baseline_test{,_ios0,_botsort}/` |

Ground truth: Halmstad labels every frame of a clip, not only the 5 fps samples used for training,
so the tracker output can be compared to the labels frame by frame. That comparison is an ad-hoc
script so far (greedy matching per frame by confidence, IoU ≥ 0.5, the class has to match;
airplane and helicopter labels count as "other", so any box on them is a false positive). A
track counts as "true" if at least half of its boxes match a labeled object.

## Results

Drone clips (18 clips, exactly one drone in every frame, 5,462 labeled boxes):

| variant | tracks | true tracks | clips with exactly 1 track | P / R per frame |
|---|---|---|---|---|
| ByteTrack, IoS 0.8 (default) | 31 | 23 | 12 | 0.819 / 0.821 |
| ByteTrack, no IoS | 45 | 29 | 7 | 0.797 / 0.827 |
| BoT-SORT, IoS 0.8 | 31 | 23 | 11 | 0.827 / 0.831 |

Tracks per drone clip with the default: 1, 4, 4, 1, 1, 2, 1, 1, 1, 2, 1, 5, 1, 1, 1, 1, 1, 2
(without IoS: 4, 6, 4, 1, 2, 3, 1, 2, 6, 2, 2, 5, 1, 1, 1, 1, 1, 2).

Other clips with the default settings:

| clip type | clips | labeled boxes of our classes | P / R per frame | tracks | false positive boxes |
|---|---|---|---|---|---|
| BIRD | 8 | 3,106 | 0.893 / 0.704 | 45 (37 true) | 261 |
| AIRPLANE | 6 | 0 (1,870 airplane) | – | 2 | 401 |
| HELICOPTER | 12 | 0 (3,704 helicopter) | – | 7 | 244 |

Speed: median 57 frames/s per video with ByteTrack, 50 with BoT-SORT (640×512, video decoding included, no video output) on the
RTX 4070 SUPER. SAHI with 320 px slices on a bird clip ran at 8 frames/s.

## Findings

- **Containment suppression helps the tracker.** Without it, the second, larger box around a
  drone (see the baseline FP analysis) often survives for long enough to become its own track.
  With `--ios 0.8`, the drone tracks drop from 45 to 31 and the clips with a single clean track
  rise from 7 to 12 of 18, at a cost of 0.6 points of recall. In the first test clip
  (`V_DRONE_001`) it turns 4 tracks into one track over the full 10 s.
- The same suppression on the test split's single images (conf 0.25, NMS 0.5): multirotor
  precision 0.738 → 0.800, recall 0.838 → 0.828; birds unchanged; the threshold makes no
  difference between 0.6 and 0.9. COCO mAP50 drops from 0.753 to 0.722 because the
  suppression also removes alternative boxes at low confidence. So it is a video-inference
  default and not part of `evaluate.py`.
- **BoT-SORT is on par with ByteTrack here**: the same 31 drone tracks, P/R about 1 point higher,
  ~13 % slower. With a static camera its camera motion compensation has nothing to correct.
  ByteTrack stays the default; BoT-SORT should be compared again on footage from a moving camera.
- Per-frame P/R of the tracker output (drones 0.82 / 0.82) is about what the detector reaches on
  single frames at conf 0.25. ByteTrack does not raise recall much here. Its benefit is the
  identity over time, plus dropping isolated one-frame boxes.
- Remaining false tracks:
  - `V_AIRPLANE_032`: the airplane is detected as `multirotor` in 284 of 316 frames and forms one
    long track. On single frames the airplane clips were barely a problem, but a consistent
    misclassification becomes a stable, confident track.
  - three helicopter clips produce short `multirotor` tracks (3–12 boxes), and two produce `bird`
    tracks
  - the drone clips with 4–5 tracks need a look at the video (ID switches vs. duplicate boxes)
- Track IDs are not consecutive (e.g. 1, 23, 27). Ultralytics hands out an ID when a
  candidate track starts, and candidates that are never confirmed still use up their ID.

## Limits

- Halmstad is a static ground camera with mostly sky backgrounds. FPV footage has a moving
  camera and motion blur, which matters for Kalman-filter tracking. BoT-SORT's camera motion
  compensation is likely to matter more there than here.
- The ground-truth comparison is ad hoc. Proper MOT metrics (HOTA, IDF1, ID switches) are a
  phase 6 task.
- SAHI was only checked for correctness (boxes land at the right place in the frame). At 640×512
  it has nothing to gain; it is meant for 1080p and larger footage.

## Next steps

1. Turn the ad-hoc comparison into `scripts/evaluate_video.py` with MOT metrics.
2. Look at the drone clips with several tracks and the airplane clip tracked as a drone.
3. Test on FPV-like footage, where SAHI and camera motion compensation should matter.
