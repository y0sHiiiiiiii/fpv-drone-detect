<div align="center">

# fpv-drone-detect

**Every drone that reaches a trench was a dot in the sky first.**<br>
Detecting, classifying and tracking small aerial objects in recorded video.

[![CI](https://github.com/y0sHiiiiiiii/fpv-drone-detect/actions/workflows/ci.yml/badge.svg)](https://github.com/y0sHiiiiiiii/fpv-drone-detect/actions/workflows/ci.yml)
[![License: AGPL-3.0](https://img.shields.io/badge/license-AGPL--3.0-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Ultralytics YOLO11](https://img.shields.io/badge/Ultralytics-YOLO11-111F68.svg)](https://docs.ultralytics.com/)
![Status: research prototype](https://img.shields.io/badge/status-research%20prototype-orange.svg)

<img src="docs/media/showcase.gif" width="760" alt="Detections and track IDs on held-out test clips: two drones, a drone against clouds, birds, a helicopter">

<sub>Held-out test footage the model never saw during training ·
full 720p video: <a href="https://github.com/y0sHiiiiiiii/fpv-drone-detect/releases/download/v0.1.0/showcase.mp4">showcase.mp4</a></sub>

</div>

---

## Situation

Since 2022, the war in Ukraine has made the small FPV drone one of the defining weapons of
modern warfare. A few hundred dollars of plastic, motors and a camera now does work that used to
need artillery. Both sides build them by the hundreds of thousands, and the front line has
become a strip of land where anything that moves in the open is watched from above.

It is also one of the most filmed wars in history. Strikes, interceptions and near misses are
recorded from drone cameras, phones and trenches, and then uploaded. That adds up to thousands of
hours of footage. In almost every frame the most important object is tiny: a dark speck of 15 to
30 pixels against a grey sky, a tree line or a cloud, and a second later it is gone.

Anyone who wants to understand this footage has to answer one question frame by frame first:
**is that dot a drone?** That goes for researchers, journalists and analysts, and for the people
building tomorrow's airspace protection.

## Mission

`fpv-drone-detect` answers that question for recorded video:

1. **Find** every aerial object in every frame, down to a few pixels in size.
2. **Classify** it: drone or bird, airplane or helicopter.
3. **Follow** it through the clip with a stable track ID.
4. **Report** it as an annotated video and a per-frame JSON log for analysis.

Everything runs offline on recorded files. The pipeline never touches a live feed, a camera or a
drone.

## Order of battle

| Asset | Role |
|---|---|
| **YOLO11** (Ultralytics) | the detector, fine-tuned at 1280 px input so that a 20 px drone still has pixels to work with |
| **Stride-4 detection head** (`yolo11-p2`) | a custom head for the smallest targets in the sky, currently in training |
| **ByteTrack / BoT-SORT** | turns detections into tracks: one object, one ID, start to finish |
| **SAHI** | slices large frames into tiles for footage beyond 640 × 512 |
| **Containment suppression** | removes the second, oversized box that small objects tend to get |
| **7 airframe classes** | `multirotor`, `fixed_wing`, `vtol`, `bird`, `unknown_aerial`, `airplane`, `helicopter` |

Birds, airplanes and helicopters are classes of their own. The model has to learn what a drone is
*not*, or it reports every crow and every airliner as a threat.

## Intel: results so far

The model was trained on the public Halmstad drone detection dataset: 197 videos for training,
scored on 45 videos it never saw. The footage comes from static ground cameras, shows mostly sky,
and most drones in it are smaller than 32 × 32 px.

| | |
|---|---|
| detection, test split (mAP@0.5) | **0.847** |
| drones on video, precision / recall | **0.89 / 0.91** |
| drone tracking (MOTA / IDF1) | **0.77 / 0.83** |
| airplanes or helicopters tracked as a drone | **0** on all held-out aircraft clips |

What still goes wrong is documented just as carefully:
- On the smallest drones, the boxes come out too large.
- Airliners are sometimes taken for helicopters.
- Specks of dirt on a lens were reported as birds.
- Upscaling the input at inference time made things *worse*, and the write-up says so.

Every run, including the failures, is in [`docs/experiments/`](docs/experiments/).

**The honest caveat:** this is still a ground camera in Sweden, not footage from the front. Shaky
FPV video, motion blur, smoke and ground clutter are the next battlefield for this model, and it
has not fought there yet.

## Rules of engagement

This project watches. It does not act.

- **Offline only:** recorded video in, annotations out. No live feeds, no on-board or real-time
  deployment.
- **No hardware:** no cameras, flight controllers, gimbals, sensors or MAVLink.
- **No targeting:** no control, guidance, interception or any other actuation logic.
- **No attribution:** no identification of specific military models, manufacturers or units, and
  no geolocation of footage or people.
- **Data with care:** only public research datasets, no scrapers, and no footage showing
  identifiable people or casualties.

Pull requests that cross these lines will be closed.

## Field manual

[docs/GUIDE.md](docs/GUIDE.md) covers everything needed to run the pipeline:
- setup and data preparation
- training and evaluation
- video inference
- detailed results
- the repository layout

Plans and decisions are in [ROADMAP.md](ROADMAP.md). How to contribute is in
[CONTRIBUTING.md](CONTRIBUTING.md).

## Citation

If this project is useful for your work, please cite it (see [CITATION.cff](CITATION.cff)). Please
also cite the dataset it was trained on: F. Svanström, F. Alonso-Fernandez, C. Englund, "A dataset
for multi-sensor drone detection", *Data in Brief* 39 (2021), doi:10.1016/j.dib.2021.107521.

## License

The code is licensed under [AGPL-3.0](LICENSE), matching Ultralytics, on which it builds. The
Halmstad dataset is CC0-1.0, and the showcase GIF and video show frames from it.
