import pytest

from fpvdd.halmstad import video_ground_truth
from fpvdd.video_eval import (
    GTBox,
    PredBox,
    add_counts,
    evaluate_sequence,
    match_frame,
    preds_from_records,
    summarize,
)

NAMES = ["multirotor", "fixed_wing", "vtol", "bird", "unknown_aerial"]
DRONE = (100.0, 100.0, 120.0, 120.0)
NEAR = (101.0, 101.0, 121.0, 121.0)  # IoU ~0.82 with DRONE
FAR = (300.0, 300.0, 320.0, 320.0)


def gt(box=DRONE, cls=0, obj=1):
    return GTBox(cls, box, obj)


def pred(track, box=NEAR, cls=0, conf=0.9):
    return PredBox(cls, conf, box, track)


def test_match_frame_is_class_aware_and_one_to_one() -> None:
    matches, _ = match_frame([gt()], [pred(1), pred(2, box=DRONE)], 0.5)
    assert matches == [(0, 1)]  # the higher IoU wins, only one match
    matches, _ = match_frame([gt()], [pred(1, cls=3)], 0.5)
    assert matches == []
    matches, _ = match_frame([gt(cls=None)], [pred(1)], 0.5)  # e.g. an airplane label
    assert matches == []


def test_match_frame_keeps_previous_track() -> None:
    matches, _ = match_frame([gt()], [pred(1), pred(2, box=DRONE)], 0.5, previous={1: 1})
    assert matches == [(0, 0)]  # track 1 still overlaps enough, so no switch to track 2


def test_perfect_single_track() -> None:
    frames = range(5)
    r = evaluate_sequence({f: [gt()] for f in frames}, {f: [pred(7)] for f in frames}, NAMES)
    s = summarize(add_counts([r]))
    assert (s["tp"], s["fp"], s["fn"]) == (5, 0, 0)
    assert s["identity"]["mota"] == pytest.approx(1.0)
    assert s["identity"]["idf1"] == pytest.approx(1.0)
    assert s["identity"]["tracks_per_object"] == 1.0
    assert s["identity"]["idsw"] == s["identity"]["frag"] == 0


def test_id_switch_fragmentation_and_idf1() -> None:
    gts = {f: [gt()] for f in range(6)}
    # track 1 for frames 0-1, missed in frame 2, track 2 for frames 3-5
    preds = {0: [pred(1)], 1: [pred(1)], 2: [], 3: [pred(2)], 4: [pred(2)], 5: [pred(2)]}
    s = summarize(add_counts([evaluate_sequence(gts, preds, NAMES)]))
    ident = s["identity"]
    assert ident["idsw"] == 1
    assert ident["frag"] == 1
    assert ident["tracks_per_object"] == 2.0
    # MOTA = 1 - (FN 1 + FP 0 + IDSW 1) / 6
    assert ident["mota"] == pytest.approx(1 - 2 / 6)
    # IDF1: best pairing is object 1 <-> track 2 (3 frames); IDFP 2, IDFN 3
    assert ident["idf1"] == pytest.approx(6 / (6 + 2 + 3))


def test_false_positive_breakdown_and_false_tracks() -> None:
    gts = {f: [gt(), GTBox(None, FAR)] for f in range(3)}
    preds = {
        # duplicate bird box on the matched drone, a box on the airplane
        0: [pred(1), pred(2, box=DRONE, cls=3), pred(3, box=FAR)],
        # drone only seen as bird, plus a box in empty sky
        1: [pred(2, box=DRONE, cls=3), pred(4, box=(500, 500, 510, 510))],
        2: [pred(1), pred(5, box=(108, 108, 128, 128))],  # offset box, IoU ~0.24
    }
    s = summarize(add_counts([evaluate_sequence(gts, preds, NAMES)]))
    assert s["fp_by_kind"] == {
        "wrong_class": 1,
        "duplicate": 1,
        "localization": 1,
        "other_object": 1,
        "background": 1,
    }
    assert s["per_class"]["bird"] == {
        "tp": 0,
        "fp": 2,
        "fn": 0,
        "precision": 0.0,
        "recall": None,
    }
    assert s["identity"]["tracks"] == 5
    assert s["identity"]["false_tracks"] == 4


def test_without_ids_only_detection_metrics() -> None:
    gts = {0: [GTBox(3, DRONE), GTBox(3, FAR)]}
    preds = {0: [pred(None, cls=3), pred(None, box=FAR, cls=3)]}
    r = evaluate_sequence(gts, preds, NAMES)
    assert r["id"] is None
    total = add_counts([r, evaluate_sequence({0: [gt()]}, {0: [pred(1)]}, NAMES)])
    assert total["videos_with_ids"] == 1
    s = summarize(total)
    assert s["tp"] == 3
    assert s["identity"]["objects"] == 1


def test_only_predicted_frames_count() -> None:
    gts = {f: [gt()] for f in range(10)}
    s = summarize(add_counts([evaluate_sequence(gts, {0: [pred(1)]}, NAMES)]))
    assert (s["frames"], s["gt"], s["fn"]) == (1, 1, 0)


def test_preds_from_records() -> None:
    records = [
        {
            "frame": 3,
            "timestamp_s": 0.1,
            "detections": [
                {"track_id": 2, "class": "bird", "confidence": 0.4, "bbox_xyxy": [1, 2, 3, 4]},
                {"track_id": 5, "class": "bird", "confidence": 0.05, "bbox_xyxy": [1, 2, 3, 4]},
            ],
        }
    ]
    assert preds_from_records(records, NAMES, conf=0.1) == {3: [PredBox(3, 0.4, (1, 2, 3, 4), 2)]}


def test_halmstad_video_ground_truth() -> None:
    def map_class(name: str) -> int | None:
        return {"drone": 0, "bird": 3}.get(name.lower())

    # MATLAB [x, y, w, h], first pixel at 0.5
    frames = [{"DRONE": [(10.5, 20.5, 4, 6)], "AIRPLANE": [(50.5, 50.5, 10, 10)]}, {"DRONE": []}]
    out, ids = video_ground_truth(frames, map_class)
    assert ids
    assert out[0] == [GTBox(0, (10, 20, 14, 26), 1), GTBox(None, (50, 50, 60, 60), None)]
    assert out[1] == []

    flock = [{"BIRD": [(0.5, 0.5, 2, 2), (10.5, 10.5, 2, 2)]}]
    out, ids = video_ground_truth(flock, map_class)
    assert not ids
    assert {b.obj_id for b in out[0]} == {None}


def _load_script():
    import importlib.util

    from fpvdd.paths import REPO_ROOT

    spec = importlib.util.spec_from_file_location(
        "evaluate_video_script", REPO_ROOT / "scripts" / "evaluate_video.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_end_to_end(tmp_path, monkeypatch) -> None:
    import json

    script = _load_script()
    raw = tmp_path / "raw"
    raw.mkdir()
    for name in ("V_DRONE_001", "V_AIRPLANE_002"):
        (raw / f"{name}.mp4").touch()
        (raw / f"{name}_LABELS.mat").touch()
    labels = {
        "V_DRONE_001": [{"DRONE": [(100.5, 100.5, 20, 20)]} for _ in range(4)],
        "V_AIRPLANE_002": [{"AIRPLANE": [(300.5, 300.5, 20, 20)]} for _ in range(4)],
    }
    monkeypatch.setattr(script, "load_labels", lambda clip: labels[clip.video.stem])

    def records(dets_per_frame):
        return [
            {"frame": i, "timestamp_s": i / 30, "detections": dets}
            for i, dets in enumerate(dets_per_frame)
        ]

    def det(track, cls, box):
        return {"track_id": track, "class": cls, "confidence": 0.8, "bbox_xyxy": box}

    pred = tmp_path / "pred"
    pred.mkdir()
    drone = [[det(1, "multirotor", [100, 100, 120, 120])]] * 2
    drone += [[det(2, "multirotor", [100, 100, 120, 120])], []]
    plane = [[det(5, "multirotor", [300, 300, 320, 320])]] * 4
    for name, dets in (("V_DRONE_001", drone), ("V_AIRPLANE_002", plane)):
        (pred / f"{name}_detections.json").write_text(json.dumps(records(dets)))
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "video_id,split\nhalmstad_V_DRONE_001,test\nhalmstad_V_AIRPLANE_002,train\n"
    )

    args = ["--pred", str(pred), "--raw", str(raw), "--manifest", str(manifest)]
    assert script.main(args) == 0
    results = json.loads((pred / "video_eval" / "metrics.json").read_text())
    assert results["leaked"] == ["V_AIRPLANE_002"]
    drone_m = results["groups"]["DRONE"]["metrics"]
    assert (drone_m["tp"], drone_m["fn"]) == (3, 1)
    assert drone_m["identity"]["idsw"] == 1
    plane_m = results["groups"]["AIRPLANE"]["metrics"]
    assert plane_m["fp_by_kind"]["other_object"] == 4
    assert plane_m["identity"]["false_tracks"] == 1
    report = (pred / "video_eval" / "report.md").read_text(encoding="utf-8")
    assert "training or validation split" in report
    assert (pred / "video_eval" / "per_video.csv").read_text().startswith("video,group,split")
