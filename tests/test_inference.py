import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from fpvdd.draw import draw_detections
from fpvdd.inference import (
    Detection,
    detections_from_array,
    detections_from_tracks,
    frame_record,
    summarize_tracks,
    suppress_contained,
)
from fpvdd.paths import REPO_ROOT
from fpvdd.video import open_writer, probe, transcode_h264

NAMES = [
    "multirotor",
    "fixed_wing",
    "vtol",
    "bird",
    "unknown_aerial",
    "airplane",
    "helicopter",
]
SCRIPT = REPO_ROOT / "scripts" / "infer_video.py"
LOCAL_WEIGHTS = REPO_ROOT / "runs" / "weights" / "yolo11s.pt"


def test_suppress_contained_drops_nested_box() -> None:
    dets = np.array(
        [
            [100, 100, 140, 130, 0.3, 0],  # larger box around the drone, IoU 0.4 with the tight one
            [110, 105, 135, 125, 0.5, 0],  # tight box, higher confidence
            [300, 300, 310, 310, 0.2, 3],  # separate object
        ]
    )
    kept = suppress_contained(dets, 0.8)
    assert kept[:, 4].tolist() == pytest.approx([0.5, 0.2])


def test_suppress_contained_keeps_partial_overlap_and_empty() -> None:
    dets = np.array([[0, 0, 10, 10, 0.9, 0], [5, 0, 15, 10, 0.8, 3]])  # half inside
    assert len(suppress_contained(dets, 0.8)) == 2
    assert suppress_contained(np.zeros((0, 6)), 0.8).shape == (0, 6)


def test_conversions() -> None:
    (det,) = detections_from_array(np.array([[1, 2, 3, 4, 0.5, 3]]))
    assert det == Detection(None, 3, 0.5, (1.0, 2.0, 3.0, 4.0))
    (trk,) = detections_from_tracks(np.array([[1, 2, 3, 4, 7, 0.6, 0, 0]]))
    assert trk == Detection(7, 0, 0.6, (1.0, 2.0, 3.0, 4.0))


def test_frame_record_format() -> None:
    rec = frame_record(12, 0.4000001, [Detection(None, 3, 0.123456, (1.04, 2, 3, 4.26))], NAMES)
    assert rec == {
        "frame": 12,
        "timestamp_s": 0.4,
        "detections": [
            {"track_id": None, "class": "bird", "confidence": 0.1235, "bbox_xyxy": [1.0, 2, 3, 4.3]}
        ],
    }
    json.dumps(rec)  # plain Python types only


def test_summarize_tracks_majority_class() -> None:
    def rec(frame: int, *dets: tuple[int | None, int]) -> dict:
        boxes = [Detection(tid, cls, 0.5, (0, 0, 4, 9)) for tid, cls in dets]
        return frame_record(frame, frame / 10, boxes, NAMES)

    records = [rec(0, (1, 0)), rec(1, (1, 3), (None, 0)), rec(2, (1, 0), (2, 3)), rec(3)]
    rows = summarize_tracks(records)
    assert [r["track_id"] for r in rows] == [1, 2]
    one = rows[0]
    assert one["class"] == "multirotor"
    assert one["class_votes"] == {"multirotor": 2, "bird": 1}
    assert (one["frames"], one["first_frame"], one["last_frame"]) == (3, 0, 2)
    assert one["duration_s"] == pytest.approx(0.2)
    assert one["median_box_px"] == pytest.approx(6.0)


def test_draw_detections_enlarges_tiny_boxes() -> None:
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    out = draw_detections(img, [Detection(4, 3, 0.9, (50, 50, 52, 52))], NAMES, caption="f 1")
    assert img.sum() == 0
    assert out[47, 51].any()  # edge of the 8 px minimum box, outside the 2 px object
    assert out[2, 2].any()  # caption background


def test_open_writer_and_transcode(tmp_path: Path) -> None:
    raw = tmp_path / "raw.mp4"
    writer = open_writer(raw, 10.0, 64, 48)
    for _ in range(5):
        writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
    writer.release()
    assert probe(raw).frame_count == 5
    if shutil.which("ffmpeg") is None:
        assert not transcode_h264(raw, tmp_path / "h264.mp4")
        return
    assert transcode_h264(raw, tmp_path / "h264.mp4")
    cap = cv2.VideoCapture(str(tmp_path / "h264.mp4"))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 5
    cap.release()


def _run(*args: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")}
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args], env=env, capture_output=True, text=True
    )


def test_missing_source_fails(tmp_path: Path) -> None:
    weights = tmp_path / "w.pt"
    weights.write_bytes(b"")
    result = _run("--weights", str(weights), "--source", str(tmp_path / "nope.mp4"))
    assert result.returncode == 1
    assert "source not found" in result.stderr


@pytest.mark.skipif(not LOCAL_WEIGHTS.exists(), reason="needs runs/weights/yolo11s.pt")
@pytest.mark.parametrize("mode", [[], ["--sahi", "--slice", "32", "--tracker", "none"]])
def test_cli_end_to_end(make_video, tmp_path: Path, mode: list[str]) -> None:
    video = make_video("clip.avi", n_frames=6, fps=10.0)
    out = tmp_path / "out"
    args = ["--weights", str(LOCAL_WEIGHTS), "--source", str(video), "--output", str(out)]
    result = _run(*args, "--device", "cpu", "--imgsz", "64", *mode)
    assert result.returncode == 0, result.stderr
    records = json.loads((out / "clip_detections.json").read_text(encoding="utf-8"))
    assert [r["frame"] for r in records] == list(range(6))
    assert records[1]["timestamp_s"] == pytest.approx(0.1)
    assert probe(out / "clip_annotated.mp4").frame_count == 6
    assert not list(out.glob("*.mp4v.mp4"))
    assert (out / "clip_tracks.csv").read_text(encoding="utf-8").startswith("track_id,class")
    run = json.loads((out / "infer_run.json").read_text(encoding="utf-8"))
    assert run["videos"][0]["frames"] == 6
    assert (run["settings"]["sahi"] is not None) == ("--sahi" in mode)
