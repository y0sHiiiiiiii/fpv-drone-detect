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
so the tracker output can be compared to the labels frame by frame. The tables under "Results"
come from a first ad-hoc script (greedy matching by confidence, IoU ≥ 0.5, the class has to match;
any box on an airplane or helicopter is a false positive). A track counts as "true" if at least
half of its boxes match a labeled object. `scripts/evaluate_video.py` replaced that script
afterwards (see "Tracking metrics" below); its P/R and track counts agree with the ad-hoc numbers.

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
  - `V_AIRPLANE_032`: the airplane is detected in 284 of 316 frames (252 times as `multirotor`,
    32 times as `bird`) and forms one long track. On single frames the airplane clips were barely
    a problem, but a consistent misclassification becomes a stable track.
  - three helicopter clips produce short `multirotor` tracks (3–12 boxes), and two produce `bird`
    tracks
  - the drone clips with 4–5 tracks: see "What the metrics show" below
- Track IDs are not consecutive (e.g. 1, 23, 27). Ultralytics hands out an ID when a
  candidate track starts, and candidates that are never confirmed still use up their ID.

## Tracking metrics (`evaluate_video.py`)

`python scripts/evaluate_video.py --pred runs/infer/<run>` matches boxes per frame, class-aware
and one-to-one (IoU ≥ 0.5). As in CLEAR MOT, an object keeps its previous track as long as that
track still overlaps it enough; the rest is assigned by maximum IoU. Halmstad has no object ids,
so identity metrics only cover clips with at most one object of our classes per frame: 42 of the
44 test clips. The flock clips `V_BIRD_017` (up to 7 birds) and `V_BIRD_048` (2) only count
towards P/R.

Drone clips (18):

| variant | P / R | MOTA | IDF1 | ID switches | fragmentations | tracks / object | false tracks |
|---|---|---|---|---|---|---|---|
| ByteTrack, IoS 0.8 (default) | 0.819 / 0.821 | 0.638 | 0.787 | 7 | 348 | 1.39 | 8 |
| ByteTrack, no IoS | 0.797 / 0.827 | 0.613 | 0.739 | 18 | 337 | 1.78 | 16 |
| BoT-SORT, IoS 0.8 | 0.827 / 0.831 | 0.655 | 0.789 | 7 | 336 | 1.39 | 8 |
| default, matched at IoU ≥ 0.3 | 0.974 / 0.976 | 0.949 | 0.938 | 7 | 56 | 1.39 | 6 |

Bird clips (8, identity metrics on 6): P/R 0.893 / 0.704, MOTA 0.738, IDF1 0.726 with the
default; P/R 0.987 / 0.778 at IoU ≥ 0.3.

False positives by kind (default, IoU ≥ 0.5):

| clip type | FP | wrong class | duplicate | localization (IoU 0.1–0.5) | other object | background |
|---|---|---|---|---|---|---|
| DRONE | 990 | 0 | 0 | 915 | 0 | 75 |
| BIRD | 261 | 16 | 0 | 239 | 0 | 6 |
| AIRPLANE | 401 | 0 | 0 | 0 | 389 | 12 |
| HELICOPTER | 244 | 0 | 0 | 0 | 93 | 151 |

Without IoS, the drone clips have 96 duplicate boxes; with IoS there are none.

### What the metrics show

- **The drones are found; the boxes are what's off.** 915 of the 990 drone false positives sit on
  the drone with IoU between 0.1 and 0.5. Every such frame counts once as a false positive and
  once as a miss, and it breaks the track's run of matches (348 fragmentations). At IoU ≥ 0.3,
  P/R rises to 0.97 / 0.98, MOTA to 0.95, and fragmentations drop to 56. The mean IoU of the
  matches is 0.71.
- **The cause is a size bias in the model, not the labels.** On the 910 drone localization
  errors, the box centre is accurate (median offset 1.0 / 0.8 px), but the predicted box is too
  large (median 1.43× the labeled width, 1.52× the height, labeled boxes median 23 × 17 px).
  Over all drone frames, the area ratio predicted / labeled depends on object size:

  | labeled box side | < 15 px | 15–30 px | > 30 px |
  |---|---|---|---|
  | median area ratio predicted / labeled (drones) | 4.18 | 1.36 | 0.81 |

  The model pulls box sizes towards a typical size: tiny objects get boxes about twice as wide
  and high, large ones slightly too small. The training labels match the MAT boxes exactly (checked
  on `V_DRONE_002`), and in a sample of 8 errors the labeled box was tight in 6. Two were label
  noise (`V_DRONE_025` f126, `V_DRONE_080` f12: labeled box far too large).
  Levers to try: more pixels per object (higher `imgsz`, SAHI tiles upscaled), a model with a P2
  head (stride 4) for very small objects, more training data with tiny objects.
- **Birds are really missed:** recall stays at 0.78 even at IoU ≥ 0.3.
- Looked at by eye:
  - `V_DRONE_005` (4 tracks): at frame ~252 the camera turns to a different view (Halmstad's
    pan/tilt platform), and afterwards the drone is far away in front of trees. A new track there
    is unavoidable with IoU-based tracking.
  - `V_DRONE_053` (5 tracks): one extra track sits on a tiny speck in the sky that is not labeled
    (probably a distant bird or insect, so a label gap). Another, at conf ~0.22, sits on a cloud
    at the image edge.
  - the false tracks on airplane and helicopter clips are real aircraft: a distant airplane only a
    few pixels wide, and helicopters whose rotors look like a multirotor. These are model errors
    on classes that our schema currently drops.

## Limits

- Halmstad is a ground camera on a pan/tilt platform with mostly sky backgrounds. FPV footage has
  a constantly moving camera and motion blur, which matters for Kalman-filter tracking.
  BoT-SORT's camera motion compensation is likely to matter more there than here.
- No HOTA yet, only MOTA and IDF1. Identity metrics are missing for multi-object clips, because
  Halmstad has no object ids.
- SAHI was only checked for correctness (boxes land at the right place in the frame). At 640×512
  it has nothing to gain; it is meant for 1080p and larger footage.

## Next steps

1. ✅ `scripts/evaluate_video.py` (see "Tracking metrics").
2. ✅ Box precision checked: a size bias of the model on tiny objects (see above). Next, try a
   higher `imgsz`, SAHI and a P2-head model, and compare them with `evaluate_video.py`.
3. Hard negatives against aircraft: more airplane/helicopter frames as background, or map them to
   `unknown_aerial` instead of dropping them (a class-schema decision, still open).
4. Test on FPV-like footage, where SAHI and camera motion compensation should matter.
