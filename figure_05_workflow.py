#!/usr/bin/env python3
"""Build the manuscript workflow figure from study-derived inputs.

The waveform traces are digitized from ``data/waveform_examples.png`` (Fig. 4),
and the MFDFA curves are taken from representative SAW records in
``data/mfdfa_features.csv``.  The script uses only Pillow, NumPy, and
Pandas so that it remains reproducible in the manuscript workspace.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "generated_figures" / "figure_05_workflow.png"

WIDTH, HEIGHT = 3300, 2100
MARGIN_X, TOP = 110, 190
GAP_X, GAP_Y = 70, 115
PANEL_W = (WIDTH - 2 * MARGIN_X - 2 * GAP_X) // 3
PANEL_H = 790

BLUE = "#0072B2"       # icequake
ORANGE = "#D55E00"     # tectonic earthquake
NAVY = "#173B57"
TEAL = "#147D82"
GOLD = "#D99B17"
INK = "#24323D"
MUTED = "#61717D"
GRID = "#D8E0E5"
PANEL_BORDER = "#B9C7D0"
PANEL_BG = "#F8FAFB"
PALE_BLUE = "#EAF4FA"
PALE_GOLD = "#FFF5D9"
WHITE = "#FFFFFF"


from fonts import font as portable_font


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return portable_font(size, bold)


def text_center(draw: ImageDraw.ImageDraw, xy, value: str, fnt, fill=INK):
    box = draw.textbbox((0, 0), value, font=fnt)
    w = box[2] - box[0]
    h = box[3] - box[1]
    draw.text((xy[0] - w / 2, xy[1] - h / 2), value, font=fnt, fill=fill)


def round_box(draw, box, fill=PANEL_BG, outline=PANEL_BORDER, radius=28, width=4):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def panel(draw, box, number: int, title: str):
    round_box(draw, box)
    x0, y0, _, _ = box
    draw.ellipse((x0 + 28, y0 + 26, x0 + 106, y0 + 104), fill=NAVY)
    text_center(draw, (x0 + 67, y0 + 65), str(number), font(42, True), WHITE)
    draw.text((x0 + 128, y0 + 34), title, font=font(57, True), fill=NAVY)


def arrow(draw, start, end, color=NAVY, width=18):
    draw.line((start, end), fill=color, width=width)
    x1, y1 = start
    x2, y2 = end
    angle = np.arctan2(y2 - y1, x2 - x1)
    head = 34
    wing = 23
    p1 = (x2 - head * np.cos(angle) + wing * np.sin(angle),
          y2 - head * np.sin(angle) - wing * np.cos(angle))
    p2 = (x2 - head * np.cos(angle) - wing * np.sin(angle),
          y2 - head * np.sin(angle) + wing * np.cos(angle))
    draw.polygon([(x2, y2), p1, p2], fill=color)


def extract_waveform_trace(image: Image.Image, crop_box) -> np.ndarray:
    """Recover the purple trace in a Fig. 4 crop as normalized amplitudes."""
    arr = np.asarray(image.crop(crop_box).convert("RGB"))
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    mask = (b > 90) & (r > 35) & (r < 185) & (g < 120) & (b > r + 12)
    ys = np.full(arr.shape[1], np.nan)
    for x in range(arr.shape[1]):
        hit = np.flatnonzero(mask[:, x])
        if hit.size:
            ys[x] = np.median(hit)
    valid = np.isfinite(ys)
    ys = np.interp(np.arange(len(ys)), np.flatnonzero(valid), ys[valid])
    amp = -(ys - np.median(ys))
    scale = np.percentile(np.abs(amp), 99)
    return amp / scale


def draw_trace(draw, rect, amp, color, label):
    x0, y0, x1, y1 = rect
    mid = (y0 + y1) / 2
    draw.line((x0, mid, x1, mid), fill=GRID, width=3)
    idx = np.linspace(0, len(amp) - 1, int(x1 - x0) + 1)
    vals = np.interp(idx, np.arange(len(amp)), amp)
    gain = 0.43 * (y1 - y0)
    pts = [(x0 + i, mid - gain * v) for i, v in enumerate(vals)]
    draw.line(pts, fill=color, width=4, joint="curve")
    draw.text((x0 + 10, y0 - 6), label, font=font(39, True), fill=color)


def parse_array(value) -> np.ndarray:
    if isinstance(value, str):
        return np.asarray(ast.literal_eval(value.replace("  ", ", ").replace(" ", ", ")))
    return np.asarray(value)


def parse_space_array(value) -> np.ndarray:
    text = str(value).strip()
    try:
        return np.asarray(ast.literal_eval(text), dtype=float)
    except (SyntaxError, ValueError):
        return np.fromstring(text.strip("[]"), sep=" ")


def representative_curves():
    df = pd.read_csv(ROOT / "data" / "mfdfa_features.csv")
    oof = pd.read_csv(
        ROOT / "results" / "nested_oof_predictions.csv"
    )
    selected = {}
    for label in (0.0, 1.0):
        retained = oof.loc[
            (oof["station"] == "SAW") & (oof["label_true"] == int(label)),
            "source_row_index",
        ]
        group = df.loc[retained]
        group = group[(group["station"] == "SAW") & (group["label"] == label)].copy()
        median_width = group["Delta_alpha"].median()
        row = group.loc[(group["Delta_alpha"] - median_width).abs().idxmin()]
        selected[int(label)] = {
            "h": parse_space_array(row["h(q)"]),
            "alpha": parse_space_array(row["alpha"]),
            "f": parse_space_array(row["f_alpha"]),
        }
    return selected


def plot_xy(draw, rect, series, xlabel, ylabel, xlim=None, ylim=None):
    x0, y0, x1, y1 = rect
    pad_l, pad_r, pad_t, pad_b = 82, 22, 18, 66
    ax = (x0 + pad_l, y0 + pad_t, x1 - pad_r, y1 - pad_b)
    ax0, ay0, ax1, ay1 = ax
    if xlim is None:
        xs = np.concatenate([np.asarray(s[0]) for s in series])
        xlim = (float(xs.min()), float(xs.max()))
    if ylim is None:
        ys = np.concatenate([np.asarray(s[1]) for s in series])
        margin = 0.08 * max(ys.max() - ys.min(), 1e-6)
        ylim = (float(ys.min() - margin), float(ys.max() + margin))

    for k in range(4):
        yy = ay0 + k * (ay1 - ay0) / 3
        draw.line((ax0, yy, ax1, yy), fill=GRID, width=2)
    draw.line((ax0, ay0, ax0, ay1), fill=INK, width=4)
    draw.line((ax0, ay1, ax1, ay1), fill=INK, width=4)

    def tr(x, y):
        px = ax0 + (x - xlim[0]) / (xlim[1] - xlim[0]) * (ax1 - ax0)
        py = ay1 - (y - ylim[0]) / (ylim[1] - ylim[0]) * (ay1 - ay0)
        return px, py

    for xs, ys, color in series:
        pts = [tr(float(x), float(y)) for x, y in zip(xs, ys)]
        draw.line(pts, fill=color, width=7, joint="curve")
        for px, py in pts:
            draw.ellipse((px - 6, py - 6, px + 6, py + 6), fill=color)

    text_center(draw, ((ax0 + ax1) / 2, y1 - 22), xlabel, font(37), INK)
    # Horizontal compact y label avoids tiny rotated raster text.
    draw.text((x0 + 5, y0 + 2), ylabel, font=font(34, True), fill=INK)
    draw.text((ax0 - 6, ay1 + 9), f"{xlim[0]:g}", font=font(29), fill=MUTED)
    end = f"{xlim[1]:g}"
    eb = draw.textbbox((0, 0), end, font=font(29))
    draw.text((ax1 - (eb[2] - eb[0]), ay1 + 9), end, font=font(29), fill=MUTED)
    return tr


def main():
    canvas = Image.new("RGB", (WIDTH, HEIGHT), WHITE)
    draw = ImageDraw.Draw(canvas)

    text_center(draw, (WIDTH / 2, 72),
                "Station-specific MFDFA–SVM classification workflow",
                font(78, True), NAVY)
    #text_center(draw, (WIDTH / 2, 142),
     #           "Real BHZ waveform morphology and study-derived multifractal curves are shown",
     #           font(40), MUTED)

    xs = [MARGIN_X + i * (PANEL_W + GAP_X) for i in range(3)]
    y_top = TOP
    y_bottom = TOP + PANEL_H + GAP_Y
    top_boxes = [(x, y_top, x + PANEL_W, y_top + PANEL_H) for x in xs]
    # Snake order: stages 4, 5, 6 run from bottom-right to bottom-left.
    bottom_boxes = [(x, y_bottom, x + PANEL_W, y_bottom + PANEL_H) for x in xs]

    panel(draw, top_boxes[0], 1, "BHZ waveform input")
    panel(draw, top_boxes[1], 2, "Signal preparation")
    panel(draw, top_boxes[2], 3, "MFDFA analysis")
    panel(draw, bottom_boxes[2], 4, "15-feature representation")
    panel(draw, bottom_boxes[1], 5, "Station-specific validation")
    panel(draw, bottom_boxes[0], 6, "Out-of-fold performance")

    # Stage 1: waveform shapes traced from Fig. 4.
    wav = Image.open(ROOT / "data" / "waveform_examples.png")
    ice = extract_waveform_trace(wav, (75, 28, 742, 198))
    earth = extract_waveform_trace(wav, (803, 28, 1472, 198))
    b = top_boxes[0]
    draw_trace(draw, (b[0] + 55, b[1] + 170, b[2] - 45, b[1] + 385),
               ice, BLUE, "Icequake")
    draw_trace(draw, (b[0] + 55, b[1] + 455, b[2] - 45, b[1] + 670),
               earth, ORANGE, "Tectonic earthquake")
    text_center(draw, ((b[0] + b[2]) / 2, b[3] - 54),
                "Representative AK.SAW.BHZ morphology (Fig. 4)", font(34), MUTED)

    # Stage 2: exact preparation sequence used for every record.
    b = top_boxes[1]
    prep = [
        ("300 s BHZ waveform", "50 Hz  •  15,000 samples"),
        ("Linear detrend + demean", "remove offset and linear trend"),
        ("10% cosine taper", "reduce edge discontinuities"),
        ("0.5–23 Hz band-pass", "zero-phase, four-corner Butterworth"),
    ]
    yy = b[1] + 145
    for j, (title, note) in enumerate(prep):
        fill = PALE_BLUE if j % 2 == 0 else WHITE
        round_box(draw, (b[0] + 80, yy, b[2] - 80, yy + 120), fill=fill,
                  outline="#C8D8E2", radius=22, width=3)
        text_center(draw, ((b[0] + b[2]) / 2, yy + 42), title, font(42, True), NAVY)
        text_center(draw, ((b[0] + b[2]) / 2, yy + 90), note, font(31), MUTED)
        if j < len(prep) - 1:
            arrow(draw, ((b[0] + b[2]) / 2, yy + 124),
                  ((b[0] + b[2]) / 2, yy + 151), color=TEAL, width=9)
        yy += 145

    # Real, median-spectrum-width SAW examples for panels 3 and 4.
    curves = representative_curves()
    q = np.arange(-5, 6)
    b = top_boxes[2]
    plot_xy(draw, (b[0] + 38, b[1] + 145, b[2] - 35, b[1] + 545),
            [(q, curves[1]["h"], BLUE), (q, curves[0]["h"], ORANGE)],
            "moment order q", "H(q)", xlim=(-5, 5))
    formula_y = b[1] + 595
    text_center(draw, ((b[0] + b[2]) / 2, formula_y),
                "Y(i)  →  segments s  →  local detrending", font(38, True), INK)
    text_center(draw, ((b[0] + b[2]) / 2, formula_y + 63),
                "Fq(s) ∝ sᴴ⁽q⁾   •   q = −5, …, +5", font(39), TEAL)
    text_center(draw, ((b[0] + b[2]) / 2, formula_y + 122),
                "scale-dependent fluctuation functions", font(31), MUTED)

    # Stage 4: real singularity spectra plus all 15 scalar features.
    b = bottom_boxes[2]
    plot_xy(draw, (b[0] + 32, b[1] + 130, b[2] - 35, b[1] + 455),
            [(curves[1]["alpha"], curves[1]["f"], BLUE),
             (curves[0]["alpha"], curves[0]["f"], ORANGE)],
            "Hölder exponent α", "f(α)")
    text_center(draw, ((b[0] + b[2]) / 2, b[1] + 488),
                "τ(q)=qH(q)−1   •   α=dτ/dq   •   f=qα−τ", font(34), TEAL)
    feature_lines = [
        "Δα, Δf, f_max, f_min",
        "α_min, α_0, α_max, mean(α)",
        "Δα_left, Δα_right, Δs, A",
        "D0, D1, D2",
    ]
    fy = b[1] + 540
    for line in feature_lines:
        text_center(draw, ((b[0] + b[2]) / 2, fy), line, font(37, True), INK)
        fy += 49

    # Stage 5: station-specific nested validation, with no pooling.
    b = bottom_boxes[1]
    station_text = "BAE  BAT  FID  GLI  KLU  KNK  M23K  SAW  SCM"
    round_box(draw, (b[0] + 62, b[1] + 135, b[2] - 62, b[1] + 245),
              fill=PALE_BLUE, outline="#AFCADA", radius=22, width=3)
    text_center(draw, ((b[0] + b[2]) / 2, b[1] + 172),
                "Separate records by station first", font(41, True), NAVY)
    text_center(draw, ((b[0] + b[2]) / 2, b[1] + 220),
                station_text, font(28), BLUE)
    arrow(draw, ((b[0] + b[2]) / 2, b[1] + 252),
          ((b[0] + b[2]) / 2, b[1] + 292), color=TEAL, width=10)
    round_box(draw, (b[0] + 62, b[1] + 300, b[2] - 62, b[1] + 464),
              fill=PALE_GOLD, outline="#E4C56B", radius=22, width=3)
    text_center(draw, ((b[0] + b[2]) / 2, b[1] + 342),
                "Outer stratified 10-fold CV", font(42, True), INK)
    text_center(draw, ((b[0] + b[2]) / 2, b[1] + 394),
                "held-out performance estimation", font(32), MUTED)
    text_center(draw, ((b[0] + b[2]) / 2, b[1] + 438),
                "one OOF prediction per record", font(32, True), GOLD)
    arrow(draw, ((b[0] + b[2]) / 2, b[1] + 472),
          ((b[0] + b[2]) / 2, b[1] + 510), color=TEAL, width=10)
    round_box(draw, (b[0] + 62, b[1] + 518, b[2] - 62, b[1] + 705),
              fill=WHITE, outline="#AFCADA", radius=22, width=3)
    text_center(draw, ((b[0] + b[2]) / 2, b[1] + 558),
                "Inner stratified 10-fold GridSearchCV", font(35, True), NAVY)
    text_center(draw, ((b[0] + b[2]) / 2, b[1] + 610),
                "StandardScaler  →  RBF SVC", font(38, True), TEAL)
    text_center(draw, ((b[0] + b[2]) / 2, b[1] + 657),
                "C={10,100,1000}  •  γ={scale,0.01,0.001}", font(29), INK)
    text_center(draw, ((b[0] + b[2]) / 2, b[1] + 696),
                "class_weight = balanced  •  9 independent models", font(29), MUTED)

    # Stage 6: adopted station-macro results.
    b = bottom_boxes[0]
    metrics = [
        ("Accuracy", "94.28%"),
        ("Balanced accuracy", "94.28%"),
        ("Macro-F1", "93.87%"),
        ("Sensitivity", "94.07%"),
        ("Specificity", "94.49%"),
    ]
    yy = b[1] + 135
    for label, value in metrics:
        draw.text((b[0] + 82, yy), label, font=font(39), fill=INK)
        vb = draw.textbbox((0, 0), value, font=font(43, True))
        draw.text((b[2] - 82 - (vb[2] - vb[0]), yy - 3), value,
                  font=font(43, True), fill=TEAL)
        draw.line((b[0] + 82, yy + 56, b[2] - 82, yy + 56), fill=GRID, width=2)
        yy += 82
    round_box(draw, (b[0] + 72, b[1] + 575, b[2] - 72, b[1] + 704),
              fill=PALE_BLUE, outline="#AFCADA", radius=22, width=3)
    text_center(draw, ((b[0] + b[2]) / 2, b[1] + 617),
                "28,022 held-out predictions", font(39, True), NAVY)
    text_center(draw, ((b[0] + b[2]) / 2, b[1] + 666),
                "station-macro reporting across nine models", font(30), MUTED)

    # Arrows showing the complete snake-shaped sequence.
    arrow(draw, (top_boxes[0][2] + 10, y_top + PANEL_H / 2),
          (top_boxes[1][0] - 12, y_top + PANEL_H / 2))
    arrow(draw, (top_boxes[1][2] + 10, y_top + PANEL_H / 2),
          (top_boxes[2][0] - 12, y_top + PANEL_H / 2))
    arrow(draw, (top_boxes[2][2] - 90, top_boxes[2][3] + 10),
          (bottom_boxes[2][2] - 90, bottom_boxes[2][1] - 12))
    arrow(draw, (bottom_boxes[2][0] - 10, y_bottom + PANEL_H / 2),
          (bottom_boxes[1][2] + 12, y_bottom + PANEL_H / 2))
    arrow(draw, (bottom_boxes[1][0] - 10, y_bottom + PANEL_H / 2),
          (bottom_boxes[0][2] + 12, y_bottom + PANEL_H / 2))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUT, dpi=(300, 300), optimize=True)
    print(f"Wrote {OUT} ({WIDTH}x{HEIGHT}, 300 dpi)")


if __name__ == "__main__":
    main()
