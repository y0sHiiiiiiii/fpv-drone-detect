import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from fpvdd.draw import contact_sheet, draw_boxes
from fpvdd.images import write_image
from fpvdd.labels import YoloBox, write_label_file
from fpvdd.naming import frame_filename
from fpvdd.paths import REPO_ROOT


def test_draw_boxes_does_not_modify_input() -> None:
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    out = draw_boxes(img, [YoloBox(3, 0.5, 0.5, 0.4, 0.4)])
    assert img.sum() == 0
    assert out[30, 50].any()  # top edge of the box is drawn


def test_contact_sheet_shape() -> None:
    tiles = [np.zeros((50, 100, 3), dtype=np.uint8)] * 5
    sheet = contact_sheet(tiles, cols=4, tile_width=40)
    assert sheet.shape == (2 * 20, 4 * 40, 3)


def test_visualize_cli(tmp_path: Path) -> None:
    labeled = tmp_path / "labeled"
    for i in range(4):
        img = labeled / "images" / "ds" / frame_filename("v", i)
        img.parent.mkdir(parents=True, exist_ok=True)
        write_image(img, np.zeros((48, 64, 3), dtype=np.uint8))
        boxes = [YoloBox(i % 2 * 3, 0.5, 0.5, 0.2, 0.2)] if i < 3 else []
        write_label_file(labeled / "labels" / "ds" / f"{img.stem}.txt", boxes)

    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")}
    cmd = [sys.executable, str(REPO_ROOT / "scripts" / "visualize_labels.py")]
    out = tmp_path / "qa"
    args = ["--input", str(labeled), "--output", str(out), "--class", "bird"]
    result = subprocess.run(cmd + args, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    samples = sorted(p.name for p in out.glob("v_*.jpg"))
    assert samples == [frame_filename("v", 1)]  # only frame 1 has a bird (class 3)
    assert (out / "_sheet_00.jpg").exists()
