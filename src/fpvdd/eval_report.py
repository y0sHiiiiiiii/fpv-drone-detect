"""Markdown reports for evaluate.py and evaluate_video.py results."""

from typing import Any

SIZES = ("small", "medium", "large")


def _f(value: float | None, digits: int = 3) -> str:
    return "–" if value is None else f"{value:.{digits}f}"


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return lines


def present_classes(results: dict[str, Any]) -> list[str]:
    """Classes with ground truth or predictions; others are left out of the tables."""
    names = results["names"]
    gt = results["gt_counts"]
    confusion = results["operating_point"]["confusion"]
    present = []
    for i, name in enumerate(names):
        has_gt = sum(gt[name].values()) > 0
        has_pred = sum(confusion[i]) > 0
        if has_gt or has_pred:
            present.append(name)
    return present


def render_report(results: dict[str, Any]) -> str:
    """Human-readable summary of one evaluation."""
    names = results["names"]
    classes = present_classes(results)
    ul = results["ultralytics"]
    coco = results["coco"]
    op = results["operating_point"]
    lines = [
        f"# Evaluation: {results['run']} on `{results['split']}`",
        "",
        f"- weights: `{results['weights']}`",
        f"- images: {results['images']} (imgsz {results['imgsz']})"
        + (f", random subset of {results['limit']}" if results.get("limit") else ""),
        f"- dataset fingerprint: `{results['dataset']['sha256'][:12]}`" + _dataset_note(results),
        f"- split leakage check (video ids across train/val/test): {results['leakage_check']}",
        *(
            ["- **training run not finished: intermediate weights**"]
            if results.get("training_finished") is False
            else []
        ),
        "",
        "## Overall",
        "",
    ]
    lines += _table(
        ["metric", "Ultralytics", "COCO (pycocotools)"],
        [
            ["mAP@0.5", _f(ul["map50"]), _f(coco["all"]["overall"]["ap50"])],
            ["mAP@0.5:0.95", _f(ul["map50_95"]), _f(coco["all"]["overall"]["ap"])],
        ],
    )
    lines += [
        "",
        "Ultralytics and COCO interpolate the PR curve differently, so small differences are "
        "expected.",
        "",
        "## Per class",
        "",
    ]
    rows = []
    for name in classes:
        u = ul["per_class"].get(name, {})
        o = op["per_class"][name]
        rows.append(
            [
                name,
                str(sum(results["gt_counts"][name].values())),
                _f(u.get("ap50")),
                _f(u.get("ap50_95")),
                _f(u.get("precision")),
                _f(u.get("recall")),
                _f(o["precision"]),
                _f(o["recall"]),
            ]
        )
    lines += _table(
        [
            "class",
            "GT boxes",
            "AP@0.5",
            "AP@0.5:0.95",
            "P (UL)",
            "R (UL)",
            f"P @conf {op['conf']}",
            f"R @conf {op['conf']}",
        ],
        rows,
    )
    lines += [
        "",
        "P/R (UL) are Ultralytics' values at the confidence with the best F1. "
        f"P/R @conf {op['conf']} use a fixed threshold and IoU ≥ {op['iou']}, matched "
        "class-agnostically; a box with the wrong class counts as FP for the predicted and as a "
        "miss for the true class.",
        "",
        "## By object size",
        "",
        "COCO area ranges in pixels² of the original image: small < 32², medium 32²–96², "
        "large > 96².",
        "",
    ]
    rows = []
    for name in ["overall", *classes]:
        row = [name]
        for size in SIZES:
            n = (
                sum(results["gt_counts"][c][size] for c in classes)
                if name == "overall"
                else results["gt_counts"][name][size]
            )
            stats = coco[size][name]
            row.append(f"{n} · {_f(stats['ap50'])} / {_f(stats['ap'])}")
        rows.append(row)
    lines += _table(["class", *[f"{s}: GT · AP50 / AP" for s in SIZES]], rows)
    lines += ["", f"## Confusion matrix (conf ≥ {op['conf']}, IoU ≥ {op['iou']})", ""]
    labels = [*names, "background"]
    shown = [labels.index(c) for c in classes] + [len(names)]
    matrix = op["confusion"]
    rows = [[f"**{labels[p]}**", *[str(matrix[p][t]) for t in shown]] for p in shown]
    lines += _table(["predicted ↓ / true →", *[labels[t] for t in shown]], rows)
    lines += [
        "",
        "Row `background` = missed objects (FN), column `background` = false positives.",
        *_bird_note(names, matrix),
    ]
    return "\n".join(lines) + "\n"


def _dataset_note(results: dict[str, Any]) -> str:
    match = results.get("dataset_matches_training")
    if match is None:
        return ""
    return " (same as training)" if match else " (**differs from the training run's dataset**)"


def _bird_note(names: list[str], matrix: list[list[int]]) -> list[str]:
    if "multirotor" not in names or "bird" not in names:
        return []
    m, b = names.index("multirotor"), names.index("bird")
    return [
        "",
        "**multirotor ↔ bird:** "
        f"{matrix[m][b]} birds detected as multirotor, "
        f"{matrix[b][m]} multirotors detected as bird.",
    ]


def _identity_cells(ident: dict[str, Any] | None) -> list[str]:
    if ident is None:
        return ["–"] * 7
    return [
        _f(ident["mota"]),
        _f(ident["idf1"]),
        str(ident["idsw"]),
        str(ident["frag"]),
        str(ident["tracks"]),
        str(ident["false_tracks"]),
        _f(ident["tracks_per_object"], 2),
    ]


def render_video_report(results: dict[str, Any]) -> str:
    """Human-readable summary of one evaluate_video.py run."""
    s = results["settings"]
    n_videos = len(results["videos"])
    no_ids = [v["video"] for v in results["videos"] if v["metrics"]["identity"] is None]
    lines = [
        f"# Video evaluation: `{s['pred']}`",
        "",
        f"- dataset: {s['dataset']}, {n_videos} videos, matching IoU ≥ {s['iou']}, "
        f"class-aware" + (f", confidence ≥ {s['conf']}" if s["conf"] else ""),
        f"- inference settings: {results.get('inference') or 'unknown (no infer_run.json)'}",
        "- identity metrics (MOTA, IDF1, ID switches, tracks) only cover videos with object ids"
        + (f"; without ids: {', '.join(no_ids)}" if no_ids else ""),
        *(
            [f"- **videos from the training or validation split: {', '.join(results['leaked'])}**"]
            if results["leaked"]
            else []
        ),
        "",
        "## Summary",
        "",
    ]
    header = ["group", "videos", "frames", "GT", "P", "R", "F1", "mean IoU"]
    header += ["MOTA", "IDF1", "ID sw", "frag", "tracks", "false tracks", "tracks/object"]
    rows = []
    for group, g in [*results["groups"].items(), ("**all**", results["overall"])]:
        m = g["metrics"]
        rows.append(
            [
                group,
                str(g["videos"]),
                str(m["frames"]),
                str(m["gt"]),
                _f(m["precision"]),
                _f(m["recall"]),
                _f(m["f1"]),
                _f(m["mean_iou"]),
                *_identity_cells(m["identity"]),
            ]
        )
    lines += _table(header, rows)

    kinds = list(results["overall"]["metrics"]["fp_by_kind"])
    lines += ["", "## False positives by kind", ""]
    lines += [
        "wrong_class: on one of our objects, wrong class · duplicate: second box on an object "
        "that is already matched · localization: IoU 0.1 to the matching threshold · "
        "other_object: on a labeled object outside our classes · background: nothing labeled",
        "",
    ]
    rows = [
        [group, str(g["metrics"]["fp"]), *(str(g["metrics"]["fp_by_kind"][k]) for k in kinds)]
        for group, g in [*results["groups"].items(), ("**all**", results["overall"])]
    ]
    lines += _table(["group", "FP", *kinds], rows)

    lines += ["", "## Per class", ""]
    rows = [
        [name, str(c["tp"]), str(c["fp"]), str(c["fn"]), _f(c["precision"]), _f(c["recall"])]
        for name, c in results["overall"]["metrics"]["per_class"].items()
    ]
    lines += _table(["class", "TP", "FP", "FN", "precision", "recall"], rows)

    def badness(v: dict[str, Any]) -> tuple[int, int]:
        ident = v["metrics"]["identity"] or {}
        extra = ident.get("tracks", 0) - ident.get("objects_tracked", 0)
        return (
            ident.get("false_tracks", 0) + ident.get("idsw", 0) + max(extra, 0),
            v["metrics"]["fp"],
        )

    worst = sorted(results["videos"], key=badness, reverse=True)[:15]
    worst = [v for v in worst if badness(v) != (0, 0)]
    if worst:
        lines += ["", "## Videos to look at", ""]
        lines += [
            "Sorted by false tracks + ID switches + surplus tracks, then false positives.",
            "",
        ]
        rows = []
        for v in worst:
            m = v["metrics"]
            rows.append(
                [
                    v["video"],
                    v["group"],
                    str(m["gt"]),
                    _f(m["precision"]),
                    _f(m["recall"]),
                    *_identity_cells(m["identity"])[2:],
                ]
            )
        header = ["video", "group", "GT", "P", "R", "ID sw", "frag", "tracks", "false tracks"]
        lines += _table([*header, "tracks/object"], rows)
    return "\n".join(lines) + "\n"
