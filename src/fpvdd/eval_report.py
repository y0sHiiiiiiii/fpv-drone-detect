"""Markdown report for evaluate.py results."""

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
