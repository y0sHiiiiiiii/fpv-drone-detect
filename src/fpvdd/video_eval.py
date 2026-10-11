"""Evaluate tracker output on video against per-frame ground truth.

Matching per frame is class-aware and one-to-one (IoU >= threshold). As in CLEAR MOT, a ground
truth object stays matched to its previous track while that track still overlaps it enough;
the remaining pairs are assigned by maximum IoU (Hungarian).

Detection metrics (P/R per class, false-positive breakdown) work on every video. Identity
metrics (MOTA, IDF1, ID switches, fragmentation, tracks per object) need ground-truth object
ids and track ids; videos without them only count towards the detection metrics.
"""

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from fpvdd.evaluation import box_iou

Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class GTBox:
    class_id: int | None  # None: labeled object outside our classes (e.g. an airplane)
    bbox: Box  # xyxy pixels
    obj_id: int | None = None


@dataclass(frozen=True)
class PredBox:
    class_id: int
    confidence: float
    bbox: Box
    track_id: int | None = None


def preds_from_records(
    records: Iterable[Mapping[str, Any]], names: Sequence[str], conf: float = 0.0
) -> dict[int, list[PredBox]]:
    """Frame index -> predictions, from ``<name>_detections.json`` records."""
    index = {n: i for i, n in enumerate(names)}
    return {
        int(rec["frame"]): [
            PredBox(index[d["class"]], float(d["confidence"]), tuple(d["bbox_xyxy"]), d["track_id"])
            for d in rec["detections"]
            if d["confidence"] >= conf
        ]
        for rec in records
    }


def match_frame(
    gt: Sequence[GTBox],
    preds: Sequence[PredBox],
    iou_thr: float,
    previous: Mapping[int, int] | None = None,
) -> tuple[list[tuple[int, int]], np.ndarray]:
    """One-to-one ``(gt_index, pred_index)`` matches and the full IoU matrix.

    Only boxes of the same class can match; ground truth with ``class_id`` None never matches.
    ``previous`` maps object ids to the track that matched them last; such pairs are kept if
    their IoU still reaches ``iou_thr``.
    """
    g = np.array([b.bbox for b in gt], dtype=float).reshape(-1, 4)
    p = np.array([b.bbox for b in preds], dtype=float).reshape(-1, 4)
    ious = box_iou(p, g).T  # [gt, pred]
    same = np.array(
        [[gb.class_id is not None and gb.class_id == pb.class_id for pb in preds] for gb in gt],
        dtype=bool,
    ).reshape(len(gt), len(preds))
    valid = same & (ious >= iou_thr)

    matches: list[tuple[int, int]] = []
    used_g: set[int] = set()
    used_p: set[int] = set()
    for gi, gb in enumerate(gt):
        track = (previous or {}).get(gb.obj_id) if gb.obj_id is not None else None
        if track is None:
            continue
        for pi, pb in enumerate(preds):
            if pb.track_id == track and pi not in used_p and valid[gi, pi]:
                matches.append((gi, pi))
                used_g.add(gi)
                used_p.add(pi)
                break

    free_g = [i for i in range(len(gt)) if i not in used_g]
    free_p = [i for i in range(len(preds)) if i not in used_p]
    if free_g and free_p:
        sub = np.where(valid[np.ix_(free_g, free_p)], ious[np.ix_(free_g, free_p)], 0.0)
        for r, c in zip(*linear_sum_assignment(-sub), strict=True):
            if sub[r, c] > 0:
                matches.append((free_g[r], free_p[c]))
    return matches, ious


FP_KINDS = ("wrong_class", "duplicate", "localization", "other_object", "background")


def fp_kind(gt: Sequence[GTBox], ious: np.ndarray, matched: set[int], iou_thr: float) -> str:
    """Why an unmatched prediction is a false positive, given its IoU with each GT box.

    - ``wrong_class``: right place, but on an object of our classes that it does not match
    - ``duplicate``: right place, on an object that another prediction already matches
    - ``localization``: overlaps an object of our classes only with ``0.1 <= IoU < iou_thr``
    - ``other_object``: on a labeled object outside our classes (e.g. an airplane)
    - ``background``: no labeled object there
    """
    target = [g for g, b in enumerate(gt) if b.class_id is not None]
    if any(ious[g] >= iou_thr and g not in matched for g in target):
        return "wrong_class"
    if any(ious[g] >= iou_thr for g in target):
        return "duplicate"
    if any(ious[g] >= 0.1 for g in target):
        return "localization"
    if any(ious[g] >= iou_thr for g, b in enumerate(gt) if b.class_id is None):
        return "other_object"
    return "background"


def evaluate_sequence(
    gt_frames: Mapping[int, Sequence[GTBox]],
    pred_frames: Mapping[int, Sequence[PredBox]],
    names: Sequence[str],
    iou_thr: float = 0.5,
    identities: bool = True,
) -> dict[str, Any]:
    """Additive counts for one video (sum them with :func:`add_counts`).

    Only frames present in ``pred_frames`` are evaluated. ``identities`` requires object ids on
    all ground truth of our classes and track ids on all predictions; otherwise the ``id``
    part is None.
    """
    det: Counter[str] = Counter()
    per_class = {k: Counter() for k in ("tp", "fp", "fn")}
    if identities:
        identities = all(
            b.obj_id is not None for f in gt_frames.values() for b in f if b.class_id is not None
        ) and all(p.track_id is not None for f in pred_frames.values() for p in f)

    last_track: dict[int, int] = {}  # object id -> last matched track (CLEAR MOT)
    tracked_before: dict[int, bool] = {}  # object id -> matched in its previous GT frame
    pair_frames: Counter[tuple[int, int]] = Counter()  # (object, track) -> frames with IoU ok
    track_boxes: Counter[int] = Counter()
    track_hits: Counter[int] = Counter()
    object_tracks: dict[int, set[int]] = defaultdict(set)
    gt_objects: set[int] = set()
    ids: Counter[str] = Counter()

    for frame in sorted(pred_frames):
        gt = list(gt_frames.get(frame, []))
        preds = list(pred_frames[frame])
        matches, ious = match_frame(gt, preds, iou_thr, last_track if identities else None)
        det["frames"] += 1
        targets = [i for i, b in enumerate(gt) if b.class_id is not None]
        det["gt"] += len(targets)
        det["pred"] += len(preds)
        det["tp"] += len(matches)
        det["iou_sum"] += float(sum(ious[g, p] for g, p in matches))
        matched_g = {g for g, _ in matches}
        matched_p = {p for _, p in matches}
        for g, _ in matches:
            per_class["tp"][gt[g].class_id] += 1
        for g in targets:
            if g not in matched_g:
                det["fn"] += 1
                per_class["fn"][gt[g].class_id] += 1
        for p, pb in enumerate(preds):
            if p in matched_p:
                continue
            det["fp"] += 1
            per_class["fp"][pb.class_id] += 1
            det[f"fp_{fp_kind(gt, ious[:, p], matched_g, iou_thr)}"] += 1

        if not identities:
            continue
        current = {gt[g].obj_id: preds[p].track_id for g, p in matches}
        for g in targets:
            obj = gt[g].obj_id
            gt_objects.add(obj)
            track = current.get(obj)
            if track is not None:
                if obj in last_track and last_track[obj] != track:
                    ids["idsw"] += 1
                if obj in tracked_before and not tracked_before[obj]:
                    ids["frag"] += 1
                last_track[obj] = track
                object_tracks[obj].add(track)
            tracked_before[obj] = track is not None
            for p, pb in enumerate(preds):
                if pb.class_id == gt[g].class_id and ious[g, p] >= iou_thr:
                    pair_frames[(obj, pb.track_id)] += 1
        for p, pb in enumerate(preds):
            track_boxes[pb.track_id] += 1
            track_hits[pb.track_id] += p in matched_p

    result: dict[str, Any] = {
        "det": dict(det),
        "per_class": {
            name: {k: int(per_class[k][c]) for k in per_class} for c, name in enumerate(names)
        },
        "id": None,
    }
    if identities:
        objects = sorted(gt_objects)
        tracks = sorted(track_boxes)
        idtp = 0
        if objects and tracks:
            m = np.array([[pair_frames[(o, t)] for t in tracks] for o in objects])
            rows, cols = linear_sum_assignment(-m)
            idtp = int(m[rows, cols].sum())
        result["id"] = {
            "gt": det["gt"],
            "tp": det["tp"],
            "fp": det["fp"],
            "fn": det["fn"],
            "idsw": ids["idsw"],
            "frag": ids["frag"],
            "idtp": idtp,
            "idfp": det["pred"] - idtp,
            "idfn": det["gt"] - idtp,
            "objects": len(gt_objects),
            "objects_tracked": len(object_tracks),
            "object_tracks": sum(len(t) for t in object_tracks.values()),
            "tracks": len(tracks),
            "false_tracks": sum(1 for t in tracks if track_hits[t] < 0.5 * track_boxes[t]),
        }
    return result


def add_counts(results: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Sum the counts of several :func:`evaluate_sequence` results."""
    det: Counter[str] = Counter()
    per_class: dict[str, Counter[str]] = defaultdict(Counter)
    ident: Counter[str] = Counter()
    n_id = 0
    for r in results:
        det.update(r["det"])
        for name, c in r["per_class"].items():
            per_class[name].update(c)
        if r["id"] is not None:
            ident.update(r["id"])
            n_id += 1
    return {
        "det": dict(det),
        "per_class": {k: dict(v) for k, v in per_class.items()},
        "id": dict(ident) if n_id else None,
        "videos_with_ids": n_id,
    }


def _ratio(a: float, b: float) -> float | None:
    return a / b if b else None


def summarize(counts: Mapping[str, Any]) -> dict[str, Any]:
    """Derived metrics from (summed) counts."""
    d = counts["det"]
    tp, fp, fn = d.get("tp", 0), d.get("fp", 0), d.get("fn", 0)
    out: dict[str, Any] = {
        "frames": d.get("frames", 0),
        "gt": d.get("gt", 0),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": _ratio(tp, tp + fp),
        "recall": _ratio(tp, tp + fn),
        "f1": _ratio(2 * tp, 2 * tp + fp + fn),
        "mean_iou": _ratio(d.get("iou_sum", 0.0), tp),
        "fp_by_kind": {k: d.get(f"fp_{k}", 0) for k in FP_KINDS},
        "per_class": {
            name: {
                **c,
                "precision": _ratio(c["tp"], c["tp"] + c["fp"]),
                "recall": _ratio(c["tp"], c["tp"] + c["fn"]),
            }
            for name, c in counts["per_class"].items()
            if c["tp"] + c["fp"] + c["fn"]
        },
        "identity": None,
    }
    i = counts.get("id")
    if i:
        out["identity"] = {
            "mota": (1 - (i["fn"] + i["fp"] + i["idsw"]) / i["gt"] if i["gt"] else None),
            "idf1": _ratio(2 * i["idtp"], 2 * i["idtp"] + i["idfp"] + i["idfn"]),
            "idsw": i["idsw"],
            "frag": i["frag"],
            "objects": i["objects"],
            "objects_tracked": i["objects_tracked"],
            "tracks": i["tracks"],
            "false_tracks": i["false_tracks"],
            # distinct tracks per object that was tracked at all; 1.0 is ideal
            "tracks_per_object": _ratio(i["object_tracks"], i["objects_tracked"]),
        }
    return out
