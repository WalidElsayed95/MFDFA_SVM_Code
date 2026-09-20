#!/usr/bin/env python3
"""Build revised Fig. 7 and its statistical supplement from the final feature CSV."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageColor, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "mfdfa_features.csv"
OUT = ROOT / "supplementary"

STATIONS = ["BAE", "BAT", "FID", "GLI", "KLU", "KNK", "M23K", "SAW", "SCM"]
REPRESENTATIVE_STATIONS = ["BAT", "KLU", "KNK", "SAW"]
FEATURES = [
    ("D1", "D₁"),
    ("D2", "D₂"),
    ("alpha_0", "α₀"),
    ("alpha_min", "αmin"),
    ("Delta_alpha", "Δα"),
]
TESTED_FEATURES = ["alpha_0", "Delta_alpha"]

COLORS = {0.0: "#D55E00", 1.0: "#0072B2"}
FILLS = {0.0: "#F4D5C5", 1.0: "#CFE8F3"}
LABELS = {0.0: "Earthquake", 1.0: "Icequake"}

from fonts import font as portable_font


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return portable_font(size, bold)


def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    required = {"station", "label", "serial_no", *(name for name, _ in FEATURES)}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    subset = df[df["station"].isin(STATIONS)].copy()
    if len(subset) != 28022:
        raise ValueError(f"Expected 28,022 records; found {len(subset):,}.")
    if set(subset["station"].unique()) != set(STATIONS):
        raise ValueError("Expected exactly the nine final stations.")
    if set(subset["label"].dropna().unique()) != {0.0, 1.0}:
        raise ValueError("Expected label 0 for earthquakes and label 1 for icequakes.")
    if subset.duplicated(["station", "serial_no"]).any():
        raise ValueError("Duplicate station--serial records were found.")
    if subset[[name for name, _ in FEATURES]].isna().any().any():
        raise ValueError("Missing values were found in the plotted features.")
    counts = subset.groupby(["station", "label"]).size().unstack(fill_value=0)
    if (counts == 0).any().any():
        raise ValueError("Both classes must be represented at every station.")
    return subset


def density_curve(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Gaussian KDE using Silverman's bandwidth, evaluated in memory-safe chunks."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    std = values.std(ddof=1)
    iqr = np.subtract(*np.percentile(values, [75, 25]))
    scale = min(std, iqr / 1.349) if iqr > 0 else std
    if not np.isfinite(scale) or scale <= 0:
        scale = max(abs(values.mean()) * 1e-3, 1e-3)
    bandwidth = max(0.9 * scale * n ** (-1 / 5), np.finfo(float).eps)
    result = np.zeros_like(grid, dtype=float)
    for start in range(0, n, 1000):
        z = (grid[:, None] - values[None, start : start + 1000]) / bandwidth
        result += np.exp(-0.5 * z * z).sum(axis=1)
    return result / (n * bandwidth * math.sqrt(2 * math.pi))


def feature_range(df: pd.DataFrame, feature: str) -> tuple[float, float]:
    values = df[feature].to_numpy(dtype=float)
    lo, hi = float(values.min()), float(values.max())
    padding = max(0.035 * (hi - lo), 1e-5)
    return lo - padding, hi + padding


def station_curves(df: pd.DataFrame, station: str, feature: str, xlim):
    local = df[df["station"].eq(station)]
    grid = np.linspace(xlim[0], xlim[1], 260)
    curves = {}
    for label in (0.0, 1.0):
        values = local.loc[local["label"].eq(label), feature].to_numpy(dtype=float)
        curves[label] = density_curve(values, grid)
    return grid, curves


def text_center(draw: ImageDraw.ImageDraw, xy, value: str, selected_font, fill="#222222"):
    box = draw.textbbox((0, 0), value, font=selected_font)
    draw.text((xy[0] - (box[2] - box[0]) / 2, xy[1] - (box[3] - box[1]) / 2), value, font=selected_font, fill=fill)


def draw_dashed(draw: ImageDraw.ImageDraw, points, fill, width=5, dash=18, gap=10):
    for first, second in zip(points[:-1], points[1:]):
        x1, y1 = first
        x2, y2 = second
        length = math.hypot(x2 - x1, y2 - y1)
        if length == 0:
            continue
        ux, uy = (x2 - x1) / length, (y2 - y1) / length
        position = 0.0
        while position < length:
            end = min(position + dash, length)
            draw.line(
                [(x1 + ux * position, y1 + uy * position), (x1 + ux * end, y1 + uy * end)],
                fill=fill,
                width=width,
            )
            position += dash + gap


def tick_label(value: float) -> str:
    absolute = abs(value)
    if absolute >= 10:
        return f"{value:.0f}"
    if absolute >= 1:
        return f"{value:.1f}"
    return f"{value:.2f}"


def draw_panel(
    image: Image.Image,
    box,
    grid: np.ndarray,
    curves: dict[float, np.ndarray],
    xlim,
    ymax: float,
    station_title: str | None = None,
    show_y_labels: bool = False,
    show_x_labels: bool = True,
):
    draw = ImageDraw.Draw(image)
    x0, y0, x1, y1 = box
    left, right, top, bottom = 64, 16, 34 if station_title else 14, 54
    px0, px1 = x0 + left, x1 - right
    py0, py1 = y0 + top, y1 - bottom

    if station_title:
        text_center(draw, ((x0 + x1) / 2, y0 + 15), station_title, font(24, bold=True))

    for fraction in (0.0, 0.5, 1.0):
        y = py1 - fraction * (py1 - py0)
        draw.line([(px0, y), (px1, y)], fill="#D9DEE5", width=2)
        if show_y_labels:
            label = tick_label(fraction * ymax)
            bounds = draw.textbbox((0, 0), label, font=font(15))
            draw.text((px0 - 10 - (bounds[2] - bounds[0]), y - 9), label, font=font(15), fill="#4A4F55")

    draw.line([(px0, py0), (px0, py1)], fill="#4A4F55", width=2)
    draw.line([(px0, py1), (px1, py1)], fill="#4A4F55", width=2)

    if show_x_labels:
        for value in np.linspace(xlim[0], xlim[1], 4):
            x = px0 + (value - xlim[0]) / (xlim[1] - xlim[0]) * (px1 - px0)
            draw.line([(x, py1), (x, py1 + 7)], fill="#4A4F55", width=2)
            text_center(draw, (x, py1 + 26), tick_label(value), font(15), fill="#4A4F55")

    for label in (0.0, 1.0):
        density = curves[label]
        points = [
            (
                px0 + (float(x) - xlim[0]) / (xlim[1] - xlim[0]) * (px1 - px0),
                py1 - float(y) / ymax * (py1 - py0),
            )
            for x, y in zip(grid, density)
        ]
        polygon = [(points[0][0], py1), *points, (points[-1][0], py1)]
        overlay = Image.new("RGBA", image.size, (255, 255, 255, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        rgb = ImageColor.getrgb(FILLS[label])
        overlay_draw.polygon(polygon, fill=(*rgb, 115))
        image.alpha_composite(overlay)
        draw = ImageDraw.Draw(image)
        if label == 1.0:
            draw_dashed(draw, points, COLORS[label], width=5)
        else:
            draw.line(points, fill=COLORS[label], width=5, joint="curve")


def draw_legend(image: Image.Image, center_x: int, y: int):
    draw = ImageDraw.Draw(image)
    items = [(0.0, "Earthquake"), (1.0, "Icequake")]
    widths = [draw.textbbox((0, 0), text, font=font(25))[2] + 90 for _, text in items]
    x = center_x - sum(widths) / 2
    for (label, text), width in zip(items, widths):
        yline = y + 16
        if label == 1.0:
            draw_dashed(draw, [(x, yline), (x + 55, yline)], COLORS[label], width=6, dash=16, gap=8)
        else:
            draw.line([(x, yline), (x + 55, yline)], fill=COLORS[label], width=6)
        draw.text((x + 66, y), text, font=font(25), fill="#222222")
        x += width


def make_main_figure(df: pd.DataFrame) -> Path:
    width, height = 3000, 3300
    image = Image.new("RGBA", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw_legend(image, width // 2, 30)

    left, right, top, bottom = 200, 40, 150, 95
    col_gap, row_gap = 28, 30
    panel_width = (width - left - right - 3 * col_gap) / 4
    panel_height = (height - top - bottom - 4 * row_gap) / 5

    for col, station in enumerate(REPRESENTATIVE_STATIONS):
        x = left + col * (panel_width + col_gap) + panel_width / 2
        text_center(draw, (x, 110), station, font(30, bold=True))

    row_letters = ["(a)", "(b)", "(c)", "(d)", "(e)"]
    for row, (feature, symbol) in enumerate(FEATURES):
        xlim = feature_range(df, feature)
        prepared = [station_curves(df, station, feature, xlim) for station in REPRESENTATIVE_STATIONS]
        ymax = max(float(curve.max()) for _, curves in prepared for curve in curves.values()) * 1.06
        row_y = top + row * (panel_height + row_gap)
        text_center(draw, (82, row_y + panel_height / 2 - 18), row_letters[row], font(28, bold=True))
        text_center(draw, (82, row_y + panel_height / 2 + 25), symbol, font(30, bold=True))
        for col, (grid, curves) in enumerate(prepared):
            x0 = left + col * (panel_width + col_gap)
            draw_panel(
                image,
                (x0, row_y, x0 + panel_width, row_y + panel_height),
                grid,
                curves,
                xlim,
                ymax,
                show_y_labels=col == 0,
                show_x_labels=True,
            )
    draw.text((16, height // 2 - 50), "Density", font=font(24), fill="#333333")
    path = OUT / "figure7_representative_feature_distributions.png"
    image.convert("RGB").save(path, dpi=(300, 300), quality=95)
    return path


def make_supplement_figures(df: pd.DataFrame) -> list[Path]:
    paths = []
    for feature, symbol in FEATURES:
        width, height = 3000, 2450
        image = Image.new("RGBA", (width, height), "white")
        draw = ImageDraw.Draw(image)
        draw_legend(image, width // 2, 25)
        left, right, top, bottom = 130, 40, 110, 80
        col_gap, row_gap = 30, 30
        panel_width = (width - left - right - 2 * col_gap) / 3
        panel_height = (height - top - bottom - 2 * row_gap) / 3
        xlim = feature_range(df, feature)
        prepared = {station: station_curves(df, station, feature, xlim) for station in STATIONS}
        ymax = max(float(curve.max()) for _, curves in prepared.values() for curve in curves.values()) * 1.06
        for index, station in enumerate(STATIONS):
            row, col = divmod(index, 3)
            x0 = left + col * (panel_width + col_gap)
            y0 = top + row * (panel_height + row_gap)
            grid, curves = prepared[station]
            draw_panel(
                image,
                (x0, y0, x0 + panel_width, y0 + panel_height),
                grid,
                curves,
                xlim,
                ymax,
                station_title=station,
                show_y_labels=col == 0,
                show_x_labels=True,
            )
        draw.text((20, height // 2 - 30), "Density", font=font(24), fill="#333333")
        text_center(draw, (width / 2, height - 30), f"Feature value: {symbol}", font(25))
        path = OUT / f"supplement_{feature}_nine_stations.png"
        image.convert("RGB").save(path, dpi=(300, 300), quality=95)
        paths.append(path)
    return paths


def mann_whitney_summary(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """Two-sided asymptotic Mann--Whitney p-value, AUC, and Cliff's delta."""
    combined = np.concatenate([a, b])
    ranks = pd.Series(combined).rank(method="average").to_numpy()
    n_a, n_b = len(a), len(b)
    u_b = ranks[n_a:].sum() - n_b * (n_b + 1) / 2
    auc = u_b / (n_a * n_b)
    cliff = 2 * auc - 1
    _, ties = np.unique(combined, return_counts=True)
    total = n_a + n_b
    variance = n_a * n_b / 12 * (
        (total + 1) - (ties**3 - ties).sum() / (total * (total - 1))
    )
    z = (u_b - n_a * n_b / 2) / math.sqrt(variance)
    p_value = math.erfc(abs(z) / math.sqrt(2))
    return p_value, auc, cliff


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values, dtype=float)
    running_max = 0.0
    count = len(p_values)
    for rank, index in enumerate(order):
        current = min(1.0, (count - rank) * p_values[index])
        running_max = max(running_max, current)
        adjusted[index] = running_max
    return adjusted


def p_text(value: float) -> str:
    if value == 0:
        return r"$<10^{-300}$"
    if value < 0.001:
        exponent = math.floor(math.log10(value))
        coefficient = value / (10**exponent)
        return rf"${coefficient:.2f}\times10^{{{exponent}}}$"
    return f"{value:.3f}"


def make_statistics(df: pd.DataFrame) -> tuple[Path, Path]:
    rows = []
    for feature in TESTED_FEATURES:
        group = []
        for station in STATIONS:
            local = df[df["station"].eq(station)]
            earthquake = local.loc[local["label"].eq(0.0), feature].to_numpy(dtype=float)
            icequake = local.loc[local["label"].eq(1.0), feature].to_numpy(dtype=float)
            p_value, auc, cliff = mann_whitney_summary(icequake, earthquake)
            group.append(
                {
                    "feature": feature,
                    "station": station,
                    "n_icequake": len(icequake),
                    "n_earthquake": len(earthquake),
                    "median_icequake": float(np.median(icequake)),
                    "median_earthquake": float(np.median(earthquake)),
                    "U_AUC_earthquake_greater": auc,
                    "cliffs_delta_earthquake_minus_icequake": cliff,
                    "p_raw": p_value,
                }
            )
        adjusted = holm_adjust(np.array([row["p_raw"] for row in group]))
        for row, adjusted_p in zip(group, adjusted):
            row["p_holm_within_feature"] = adjusted_p
            row["significant_0.05"] = adjusted_p < 0.05
            rows.append(row)
    result = pd.DataFrame(rows)
    csv_path = OUT / "supplementary_mannwhitney_holm_cliffs_delta.csv"
    result.to_csv(csv_path, index=False)

    symbols = {"alpha_0": r"$\alpha_0$", "Delta_alpha": r"$\Delta\alpha$"}
    lines = [
        r"\begin{longtable}{llrrrrrl}",
        r"\caption{Station-specific two-sided Mann--Whitney tests for $\alpha_0$ and $\Delta\alpha$. Holm adjustment was applied separately across the nine station comparisons for each feature. Positive Cliff's $\delta$ indicates larger earthquake values; negative values indicate larger icequake values.}\label{tab:supp_tests}\\",
        r"\toprule",
        r"Feature & Station & $N_{IQ}$ & $N_{EQ}$ & Median$_{IQ}$ & Median$_{EQ}$ & Cliff's $\delta$ & Holm-adjusted $p$ \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Feature & Station & $N_{IQ}$ & $N_{EQ}$ & Median$_{IQ}$ & Median$_{EQ}$ & Cliff's $\delta$ & Holm-adjusted $p$ \\",
        r"\midrule",
        r"\endhead",
    ]
    for row in rows:
        lines.append(
            f"{symbols[row['feature']]} & {row['station']} & {row['n_icequake']:,} & "
            f"{row['n_earthquake']:,} & {row['median_icequake']:.3f} & "
            f"{row['median_earthquake']:.3f} & "
            f"{row['cliffs_delta_earthquake_minus_icequake']:+.3f} & "
            f"{p_text(row['p_holm_within_feature'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{longtable}"])
    tex_path = OUT / "supplementary_statistics_table.tex"
    tex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, tex_path


def main() -> None:
    OUT.mkdir(exist_ok=True)
    dataframe = validate_data(pd.read_csv(DATA))
    main_figure = make_main_figure(dataframe)
    supplement_figures = make_supplement_figures(dataframe)
    csv_path, tex_path = make_statistics(dataframe)
    print(f"Validated records: {len(dataframe):,}")
    print(f"Main figure: {main_figure}")
    for path in supplement_figures:
        print(f"Supplement figure: {path}")
    print(f"Statistics CSV: {csv_path}")
    print(f"Statistics LaTeX: {tex_path}")


if __name__ == "__main__":
    main()
