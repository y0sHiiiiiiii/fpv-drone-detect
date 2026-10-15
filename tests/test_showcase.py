import importlib.util

import numpy as np

from fpvdd.paths import REPO_ROOT

spec = importlib.util.spec_from_file_location(
    "make_showcase", REPO_ROOT / "scripts" / "make_showcase.py"
)
showcase = importlib.util.module_from_spec(spec)
spec.loader.exec_module(showcase)


def test_crop_zoom_clamps_window_to_frame() -> None:
    frame = np.zeros((512, 640, 3), np.uint8)
    frame[500:, 630:] = 255  # bright corner
    zoom, x0, y0, scale = showcase.crop_zoom(frame, 639, 511, 100, 400, 300)
    assert zoom.shape == (300, 400, 3)
    assert (x0, y0) == (540, 437)  # window pushed back inside the frame
    assert scale == 4.0
    assert zoom[-1, -1].mean() > 200  # the corner is at the inset's corner


def test_main_track_is_longest_track() -> None:
    det = showcase.Det
    dets = {
        0: [det(1, "bird", 0.9, (0, 0, 1, 1)), det(2, "bird", 0.5, (0, 0, 1, 1))],
        1: [det(2, "bird", 0.5, (0, 0, 1, 1))],
        2: [det(2, "bird", 0.5, (0, 0, 1, 1)), det(None, "bird", 0.5, (0, 0, 1, 1))],
    }
    assert showcase.main_track(dets, range(3)) == 2
    assert showcase.main_track({}, range(3)) is None


def test_text_card_size() -> None:
    card = showcase.text_card([("title", "bold", 40, (255, 255, 255))], showcase.Fonts())
    assert card.shape == (showcase.HEIGHT, showcase.WIDTH, 3)
