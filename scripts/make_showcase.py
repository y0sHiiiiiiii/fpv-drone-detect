"""Render a showcase video from infer_video.py output: title card, annotated clips, end card.

Each clip shows the detections and track IDs from ``<pred>/<stem>_detections.json`` with a short
trail per track, plus a side panel with a magnified view of the main track (the objects are often
only 15-30 px wide) and the tracks in the current frame. No model runs here, so it only needs the
CPU and can run while the GPU trains.

Writes an H.264 MP4 (1280x720) and, with --gif, a shorter GIF for the README.

Examples:
    python scripts/make_showcase.py                       # configs/showcase.yaml
    python scripts/make_showcase.py --gif
    python scripts/make_showcase.py --config my_showcase.yaml --output out/demo.mp4
"""

import argparse
import json
import logging
import shutil
import subprocess
import sys
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from fpvdd.draw import CLASS_COLORS
from fpvdd.halmstad import find_clips
from fpvdd.labels import CLASS_NAMES
from fpvdd.log import setup_logging
from fpvdd.paths import CONFIGS_DIR
from fpvdd.yolo_env import resolve_path

log = logging.getLogger("make_showcase")

WIDTH, HEIGHT = 1280, 720
VIDEO_W = 900  # left area; Halmstad's 640x512 scales to exactly 900x720
PANEL_X, PAD = VIDEO_W, 24
BG = (17, 14, 13)  # BGR, near-black panel
FG, MUTED, LINE = (240, 240, 240), (150, 150, 150), (55, 52, 50)
TRAIL_FRAMES = 45
FADE = 10  # frames of fade in/out per segment and card

# Tried in order; the first that exists is used. Pillow's own font is the fallback.
FONTS = {
    "regular": ["segoeui.ttf", "DejaVuSans.ttf", "Arial.ttf"],
    "bold": ["segoeuib.ttf", "DejaVuSans-Bold.ttf", "Arial Bold.ttf"],
    "mono": ["consola.ttf", "DejaVuSansMono.ttf", "Menlo.ttc"],
}
FONT_DIRS = [
    Path("C:/Windows/Fonts"),
    Path("/usr/share/fonts/truetype/dejavu"),
    Path("/Library/Fonts"),
]


@dataclass
class Det:
    track_id: int | None
    cls: str
    conf: float
    box: tuple[float, float, float, float]

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.box
        return (x1 + x2) / 2, (y1 + y2) / 2


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, default=CONFIGS_DIR / "showcase.yaml")
    parser.add_argument("--output", type=Path, default=None, help="MP4 path (overrides config)")
    parser.add_argument("--gif", action="store_true", help="also write the README GIF")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


class Fonts:
    """Lazy cache of fonts by (style, size)."""

    def __init__(self) -> None:
        self._cache: dict[tuple[str, int], ImageFont.FreeTypeFont] = {}

    def __call__(self, style: str, size: int) -> ImageFont.FreeTypeFont:
        key = (style, size)
        if key not in self._cache:
            self._cache[key] = self._load(style, size)
        return self._cache[key]

    @staticmethod
    def _load(style: str, size: int) -> ImageFont.FreeTypeFont:
        for name in FONTS[style]:
            for folder in FONT_DIRS:
                if (folder / name).exists():
                    return ImageFont.truetype(str(folder / name), size)
        return ImageFont.load_default(size=size)


def rgb(bgr: tuple[int, int, int]) -> tuple[int, int, int]:
    return bgr[2], bgr[1], bgr[0]


def class_color(cls: str) -> tuple[int, int, int]:
    return CLASS_COLORS[CLASS_NAMES.index(cls)]


def load_detections(path: Path) -> dict[int, list[Det]]:
    records = json.loads(path.read_text(encoding="utf-8"))
    return {
        r["frame"]: [
            Det(d["track_id"], d["class"], d["confidence"], tuple(d["bbox_xyxy"]))
            for d in r["detections"]
        ]
        for r in records
    }


def read_frames(video: Path, start: int, count: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames = []
    for _ in range(count):
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames


def main_track(dets: dict[int, list[Det]], frames: range) -> int | None:
    """The track present in the most frames of the segment."""
    counts = Counter(d.track_id for f in frames for d in dets.get(f, []) if d.track_id is not None)
    return counts.most_common(1)[0][0] if counts else None


class ZoomCamera:
    """Smoothly follows a point and a window size, so the zoom inset does not jitter."""

    def __init__(self, alpha: float = 0.2) -> None:
        self.alpha = alpha
        self.state: np.ndarray | None = None  # cx, cy, window width

    def update(self, cx: float, cy: float, width: float) -> tuple[float, float, float]:
        target = np.array([cx, cy, width])
        self.state = (
            target if self.state is None else self.state + self.alpha * (target - self.state)
        )
        return tuple(self.state)


def crop_zoom(
    frame: np.ndarray, cx: float, cy: float, win_w: float, out_w: int, out_h: int
) -> tuple[np.ndarray, float, float, float]:
    """Crop a window with the output's aspect ratio around (cx, cy), clamped to the frame."""
    h, w = frame.shape[:2]
    win_w = min(win_w, w)
    win_h = win_w * out_h / out_w
    if win_h > h:
        win_h, win_w = h, h * out_w / out_h
    x0 = min(max(cx - win_w / 2, 0), w - win_w)
    y0 = min(max(cy - win_h / 2, 0), h - win_h)
    m = np.float32(
        [[out_w / win_w, 0, -x0 * out_w / win_w], [0, out_h / win_h, -y0 * out_h / win_h]]
    )
    zoom = cv2.warpAffine(
        frame, m, (out_w, out_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    return zoom, x0, y0, out_w / win_w


def draw_box(
    img: np.ndarray,
    box: tuple[float, ...],
    color: tuple[int, int, int],
    scale: float,
    offset=(0.0, 0.0),
    min_side: int = 14,
    thickness: int = 2,
) -> tuple[int, int, int, int]:  # noqa: E501
    x1, y1, x2, y2 = ((v - offset[i % 2]) * scale for i, v in enumerate(box))
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    hw, hh = max(x2 - x1 + 6, min_side) / 2, max(y2 - y1 + 6, min_side) / 2
    p1, p2 = (round(cx - hw), round(cy - hh)), (round(cx + hw), round(cy + hh))
    cv2.rectangle(img, p1, p2, color, thickness, cv2.LINE_AA)
    return p1[0], p1[1], p2[0], p2[1]


def draw_trails(img: np.ndarray, trails: dict[int, deque], scale: float) -> None:
    overlay = img.copy()
    for points in trails.values():
        if len(points) < 2:
            continue
        pts = list(points)[:-3]  # end the trail behind the object, not on it
        for i in range(1, len(pts)):
            (_, x0, y0, color), (_, x1, y1, _) = pts[i - 1], pts[i]
            thickness = 1 + round(2 * i / len(pts))
            cv2.line(
                overlay,
                (round(x0 * scale), round(y0 * scale)),
                (round(x1 * scale), round(y1 * scale)),
                color,
                thickness,
                cv2.LINE_AA,
            )
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, dst=img)


def chip(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font,
    fill,
    text_fill,
    pad=(7, 3),
    radius=5,
    anchor_bottom=True,
) -> None:  # noqa: E501
    """Rounded label; placed above ``xy`` (or below it if there is no room)."""
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    w, h = right - left + 2 * pad[0], bottom - top + 2 * pad[1]
    x, y = xy
    if anchor_bottom:
        y = y - h - 3 if y - h - 3 >= 0 else y + 3
    x = min(max(x, 0), VIDEO_W - w)
    draw.rounded_rectangle((x, y, x + w, y + h), radius=radius, fill=fill)
    draw.text((x + pad[0] - left, y + pad[1] - top), text, font=font, fill=text_fill)


def render_clip_frame(
    frame: np.ndarray,
    dets: list[Det],
    trails: dict[int, deque],
    zoom_cam: ZoomCamera,
    focus: int | None,
    seg: dict[str, Any],
    cfg: dict[str, Any],
    frame_idx: int,
    n_frames: int,
    fonts: Fonts,
    seg_pos: int,
) -> np.ndarray:
    canvas = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    canvas[:] = BG
    scale = VIDEO_W / frame.shape[1]
    video = cv2.resize(
        frame, (VIDEO_W, round(frame.shape[0] * scale)), interpolation=cv2.INTER_CUBIC
    )
    draw_trails(video, trails, scale)
    boxes = []
    for d in sorted(dets, key=lambda d: d.conf):
        color = class_color(d.cls)
        boxes.append((draw_box(video, d.box, color, scale), d, color))
    canvas[: video.shape[0], :VIDEO_W] = video

    # zoom inset of the main track
    inset_w, inset_h = WIDTH - PANEL_X - 2 * PAD, 249
    inset_y = 188
    target = next((d for d in dets if d.track_id == focus), None) or max(
        dets, key=lambda d: d.conf, default=None
    )
    if target is not None:
        x1, y1, x2, y2 = target.box
        cx, cy, win = zoom_cam.update(
            *target.center, float(np.clip(3.2 * max(x2 - x1, y2 - y1), 64, inset_w / 1.5))
        )
    elif zoom_cam.state is not None:
        cx, cy, win = zoom_cam.state
    else:
        cx, cy, win = frame.shape[1] / 2, frame.shape[0] / 2, 200.0
    zoom, zx, zy, zscale = crop_zoom(frame, cx, cy, win, inset_w, inset_h)
    for d in dets:
        draw_box(zoom, d.box, class_color(d.cls), zscale, (zx, zy), min_side=10)
    canvas[inset_y : inset_y + inset_h, PANEL_X + PAD : PANEL_X + PAD + inset_w] = zoom
    cv2.rectangle(
        canvas,
        (PANEL_X + PAD - 1, inset_y - 1),
        (PANEL_X + PAD + inset_w, inset_y + inset_h),
        LINE,
        1,
    )
    cv2.line(canvas, (PANEL_X, 0), (PANEL_X, HEIGHT), LINE, 1)

    img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    for (bx1, by1, _, _), d, color in boxes:
        label = f"{d.cls} {d.conf:.2f}" + (f"  #{d.track_id}" if d.track_id is not None else "")
        chip(draw, (bx1, by1), label, fonts("bold", 14), rgb(color), (15, 15, 15))

    # caption on the video, fading in
    alpha = min(1.0, seg_pos / FADE)
    cap_font = fonts("bold", 22)
    left, top, right, bottom = draw.textbbox((0, 0), seg["caption"], font=cap_font)
    cap = Image.new("RGBA", (right - left + 28, bottom - top + 18), (0, 0, 0, 0))
    ImageDraw.Draw(cap).rounded_rectangle(
        (0, 0, cap.width - 1, cap.height - 1), radius=8, fill=(12, 12, 14, round(185 * alpha))
    )
    ImageDraw.Draw(cap).text(
        (14 - left, 9 - top),
        seg["caption"],
        font=cap_font,
        fill=(255, 255, 255, round(255 * alpha)),
    )
    img.paste(cap, (20, 20), cap)

    # side panel
    x = PANEL_X + PAD
    draw.text((x, 22), cfg["title"], font=fonts("bold", 28), fill=rgb(FG))
    draw.text((x, 60), cfg["model_line"], font=fonts("regular", 15), fill=rgb(MUTED))
    draw.line((x, 94, WIDTH - PAD, 94), fill=rgb(LINE))
    draw.text((x, 108), "CLIP", font=fonts("bold", 12), fill=rgb(MUTED))
    draw.text((x, 126), seg["video"], font=fonts("mono", 19), fill=rgb(FG))
    draw.text((x, 152), cfg["clip_note"], font=fonts("regular", 13), fill=rgb(MUTED))
    zoom_label = f"ZOOM ×{zscale:.1f}"
    if target is not None and target.track_id is not None:
        zoom_label += f"  ·  TRACK #{target.track_id}"
    draw.text((x, inset_y - 2 + inset_h + 8), zoom_label, font=fonts("bold", 12), fill=rgb(MUTED))

    list_y = inset_y + inset_h + 44
    draw.text((x, list_y), "IN THIS FRAME", font=fonts("bold", 12), fill=rgb(MUTED))
    rows = sorted(dets, key=lambda d: -d.conf)[:6]
    if not rows:
        draw.text((x, list_y + 24), "no detections", font=fonts("mono", 16), fill=rgb(MUTED))
    for i, d in enumerate(rows):
        y = list_y + 24 + i * 26
        draw.ellipse((x, y + 4, x + 11, y + 15), fill=rgb(class_color(d.cls)))
        tid = f"#{d.track_id}" if d.track_id is not None else "–"
        draw.text(
            (x + 20, y), f"{tid:<4} {d.cls:<11} {d.conf:.2f}", font=fonts("mono", 16), fill=rgb(FG)
        )

    legend_y = HEIGHT - 120
    draw.text((x, legend_y), "CLASSES", font=fonts("bold", 12), fill=rgb(MUTED))
    for i, name in enumerate(cfg.get("legend", [])):
        lx, ly = x + (i % 2) * 160, legend_y + 22 + (i // 2) * 22
        draw.ellipse((lx, ly + 4, lx + 10, ly + 14), fill=rgb(class_color(name)))
        draw.text((lx + 18, ly), name, font=fonts("regular", 14), fill=rgb(FG))

    fps = cfg.get("fps", 30)
    footer = f"frame {frame_idx:>3}   t = {frame_idx / fps:4.1f} s"
    draw.text((x, HEIGHT - 34), footer, font=fonts("mono", 14), fill=rgb(MUTED))
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def text_card(lines: list[tuple[str, str, int, tuple[int, int, int]]], fonts: Fonts) -> np.ndarray:
    """Centered lines of (text, style, size, BGR colour); each line is 1.5x its size high."""
    img = Image.new("RGB", (WIDTH, HEIGHT), rgb(BG))
    draw = ImageDraw.Draw(img)
    heights = [round(size * 1.5) for _, _, size, _ in lines]
    y = (HEIGHT - sum(heights)) / 2
    for (text, style, size, color), height in zip(lines, heights, strict=True):
        if text:
            draw.text(
                (WIDTH / 2, y + height / 2),
                text,
                font=fonts(style, size),
                fill=rgb(color),
                anchor="mm",
            )
        y += height
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def faded(img: np.ndarray, pos: int, total: int) -> np.ndarray:
    a = min(1.0, (pos + 1) / FADE, (total - pos) / FADE)
    return img if a >= 1 else (img.astype(np.float32) * a).astype(np.uint8)


def open_ffmpeg(output: Path, fps: float) -> subprocess.Popen:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found on PATH")
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg, "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{WIDTH}x{HEIGHT}", "-r", str(fps), "-i", "-",
        "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(output),
    ]  # fmt: skip
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def make_gif(video: Path, gif: Path, start: float, duration: float, width: int, fps: int) -> None:
    gif.parent.mkdir(parents=True, exist_ok=True)
    filters = f"fps={fps},scale={width}:-1:flags=lanczos"
    palette = f"{filters},split[a][b];[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle"  # noqa: E501
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-ss", str(start), "-t", str(duration),
           "-i", str(video), "-filter_complex", palette, str(gif)]  # fmt: skip
    subprocess.run(cmd, check=True)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(args.verbose)
    cfg = yaml.safe_load(resolve_path(args.config).read_text(encoding="utf-8"))
    fps = cfg.get("fps", 30)
    pred = resolve_path(cfg["pred"])
    output = resolve_path(args.output or cfg["output"])
    clips = {c.video.stem: c for c in find_clips(resolve_path(cfg["raw"]), ("V", "IR"))}
    for seg in cfg["segments"]:
        if seg["video"] not in clips or not (pred / f"{seg['video']}_detections.json").exists():
            log.error(
                "%s: video or detections missing (raw %s, pred %s)", seg["video"], cfg["raw"], pred
            )
            return 1

    fonts = Fonts()
    proc = open_ffmpeg(output, fps)
    written = 0

    def emit(img: np.ndarray) -> None:
        nonlocal written
        proc.stdin.write(img.tobytes())
        written += 1

    title = text_card(
        [
            (cfg["title"], "bold", 64, FG),
            (cfg["subtitle"], "regular", 26, FG),
            ("", "regular", 10, FG),
            (cfg["tagline"], "regular", 18, MUTED),
        ],
        fonts,
    )
    n = round(2.5 * fps)
    for i in range(n):
        emit(faded(title, i, n))

    for seg in tqdm(cfg["segments"], desc="clips", unit="clip"):
        dets = load_detections(pred / f"{seg['video']}_detections.json")
        frames = read_frames(clips[seg["video"]].video, seg["start"], seg["frames"])
        idxs = range(seg["start"], seg["start"] + len(frames))
        focus = main_track(dets, idxs)
        trails: dict[int, deque] = {}
        zoom_cam = ZoomCamera()
        # warm up the trails with the frames before the segment
        for f in range(max(0, seg["start"] - TRAIL_FRAMES), seg["start"]):
            for d in dets.get(f, []):
                if d.track_id is not None:
                    trails.setdefault(d.track_id, deque(maxlen=TRAIL_FRAMES)).append(
                        (f, *d.center, class_color(d.cls))
                    )
        for pos, (f, frame) in enumerate(zip(idxs, frames, strict=True)):
            frame_dets = dets.get(f, [])
            for d in frame_dets:
                if d.track_id is not None:
                    trails.setdefault(d.track_id, deque(maxlen=TRAIL_FRAMES)).append(
                        (f, *d.center, class_color(d.cls))
                    )
            for tid in list(trails):  # drop points of tracks that ended
                while trails[tid] and trails[tid][0][0] < f - TRAIL_FRAMES:
                    trails[tid].popleft()
                if not trails[tid]:
                    del trails[tid]
            img = render_clip_frame(
                frame, frame_dets, trails, zoom_cam, focus, seg, cfg, f, len(frames), fonts, pos
            )
            emit(faded(img, pos, len(frames)))

    end_lines = [(cfg["title"], "bold", 48, FG), ("", "regular", 8, FG)]
    end_lines += [(line, "mono", 22, FG) for line in cfg["end_lines"]]
    end_lines += [("", "regular", 8, FG), (cfg["footer"], "regular", 18, MUTED)]
    end = text_card(end_lines, fonts)
    n = round(4 * fps)
    for i in range(n):
        emit(faded(end, i, n))

    proc.stdin.close()
    if proc.wait() != 0:
        log.error("ffmpeg failed")
        return 1
    log.info("wrote %s (%d frames, %.1f s)", output, written, written / fps)
    if args.gif:
        gif = resolve_path(cfg["gif"])
        make_gif(
            output, gif, cfg["gif_start"], cfg["gif_duration"], cfg["gif_width"], cfg["gif_fps"]
        )
        log.info("wrote %s (%.1f MB)", gif, gif.stat().st_size / 1e6)
    return 0


if __name__ == "__main__":
    sys.exit(main())
