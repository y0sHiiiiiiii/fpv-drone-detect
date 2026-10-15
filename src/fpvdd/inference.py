"""Detections and tracks of video inference: conversion, JSON records, per-track summaries.

Detector output is an ``(N, 6)`` array ``[x1, y1, x2, y2, conf, cls]`` in pixels (the layout of
Ultralytics' ``Boxes.data``). Ultralytics trackers return ``(N, 8)`` arrays
``[x1, y1, x2, y2, track_id, conf, cls, det_idx]``.
"""

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Detection:
    track_id: int | None
    class_id: int
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]


def suppress_contained(dets: np.ndarray, threshold: float) -> np.ndarray:
    """Drop boxes that lie mostly inside a higher-confidence box (class-agnostic).

    Overlap is measured as intersection over the smaller box's area. On small objects the
    model often adds a second, larger box around the tight one; their IoU stays below the NMS
    threshold, but the smaller box lies almost entirely inside the larger one. Input and
    output are ``(N, 6)`` ``[x1, y1, x2, y2, conf, cls]`` arrays, sorted by confidence.
    """
    dets = np.asarray(dets, dtype=np.float32).reshape(-1, 6)
    dets = dets[np.argsort(-dets[:, 4], kind="stable")]
    area = (dets[:, 2] - dets[:, 0]) * (dets[:, 3] - dets[:, 1])
    keep: list[int] = []
    for i in range(len(dets)):
        if keep:
            k = np.asarray(keep)
            iw = np.minimum(dets[k, 2], dets[i, 2]) - np.maximum(dets[k, 0], dets[i, 0])
            ih = np.minimum(dets[k, 3], dets[i, 3]) - np.maximum(dets[k, 1], dets[i, 1])
            inter = np.clip(iw, 0, None) * np.clip(ih, 0, None)
            smaller = np.minimum(area[k], area[i])
            if np.any(inter > threshold * np.maximum(smaller, 1e-9)):
                continue
        keep.append(i)
    return dets[keep]


def detections_from_array(dets: np.ndarray) -> list[Detection]:
    """Untracked detections from an ``(N, 6)`` ``[x1, y1, x2, y2, conf, cls]`` array."""
    return [
        Detection(None, int(d[5]), float(d[4]), tuple(d[:4].tolist()))
        for d in np.asarray(dets, dtype=float).reshape(-1, 6)
    ]


def detections_from_tracks(tracks: np.ndarray) -> list[Detection]:
    """Tracked detections from an Ultralytics tracker output ``(N, 8)`` array."""
    return [
        Detection(int(t[4]), int(t[6]), float(t[5]), tuple(t[:4].tolist()))
        for t in np.asarray(tracks, dtype=float).reshape(-1, 8)
    ]


def frame_record(
    frame: int, timestamp_s: float, detections: Iterable[Detection], names: Sequence[str]
) -> dict[str, Any]:
    """One entry of ``<name>_detections.json`` (format documented in the README)."""
    return {
        "frame": frame,
        "timestamp_s": round(timestamp_s, 4),
        "detections": [
            {
                "track_id": d.track_id,
                "class": names[d.class_id],
                "confidence": round(d.confidence, 4),
                "bbox_xyxy": [round(v, 1) for v in d.bbox_xyxy],
            }
            for d in detections
        ],
    }


def summarize_tracks(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per track id: lifetime, majority class, mean confidence and box size.

    The tracker keeps the class of the latest detection, so a track's class can change between
    frames; ``class`` is the most frequent one and ``class_votes`` lists all of them.
    """
    tracks: dict[int, dict[str, Any]] = {}
    for rec in records:
        for det in rec["detections"]:
            tid = det["track_id"]
            if tid is None:
                continue
            x1, y1, x2, y2 = det["bbox_xyxy"]
            t = tracks.setdefault(
                tid,
                {
                    "first_frame": rec["frame"],
                    "first_s": rec["timestamp_s"],
                    "votes": Counter(),
                    "conf": [],
                    "side": [],
                },
            )
            t["last_frame"] = rec["frame"]
            t["last_s"] = rec["timestamp_s"]
            t["votes"][det["class"]] += 1
            t["conf"].append(det["confidence"])
            t["side"].append(((x2 - x1) * (y2 - y1)) ** 0.5)
    rows = []
    for tid in sorted(tracks):
        t = tracks[tid]
        rows.append(
            {
                "track_id": tid,
                "class": t["votes"].most_common(1)[0][0],
                "frames": len(t["conf"]),
                "first_frame": t["first_frame"],
                "last_frame": t["last_frame"],
                "duration_s": round(t["last_s"] - t["first_s"], 3),
                "mean_conf": round(float(np.mean(t["conf"])), 4),
                "median_box_px": round(float(np.median(t["side"])), 1),
                "class_votes": dict(t["votes"].most_common()),
            }
        )
    return rows
