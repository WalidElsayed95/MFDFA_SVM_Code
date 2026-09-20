#!/usr/bin/env python3
"""Draw the nine common receivers with observed class-specific record counts."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from station_metadata import STATIONS, parse_station_file


WEST, EAST, SOUTH, NORTH = -149.0, -144.0, 60.0, 62.0
GLACIER_LAT, GLACIER_LON = 61.219722, -146.895278
EQ_COLOR = "#2ca25f"
IQ_COLOR = "#2b6cb0"
RECEIVER_COLOR = "#d7301f"
INK = "#20242a"
GRID = "#d9dde3"

LABEL_OFFSETS = {
    "BAE": (-88, -55),
    "BAT": (42, 45),
    "KNK": (-110, -70),
    "M23K": (-122, 52),
    "SAW": (-100, -65),
    "SCM": (45, -65),
    "KLU": (45, -65),
    "FID": (31, 31),
    "GLI": (45, -65),
}


def escape(value: object) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def text(x: float, y: float, value: object, **attributes: object) -> str:
    # Enlarge all labels for readability at the manuscript's printed width.
    if "font_size" in attributes:
        attributes["font_size"] = f'{float(attributes["font_size"]) * 1.5:g}'
    attrs = " ".join(
        f'{key.replace("_", "-")}="{escape(attribute)}"'
        for key, attribute in attributes.items()
    )
    return f'<text x="{x:.1f}" y="{y:.1f}" {attrs}>{escape(value)}</text>'


def read_counts(path: Path) -> dict[str, dict[str, int]]:
    counts = {station: {} for station in STATIONS}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            station = row["station"]
            event_class = row["class"]
            if station in counts and event_class in {"earthquake", "icequake"}:
                counts[station][event_class] = int(row["selected_waveforms"])
    for station, station_counts in counts.items():
        if set(station_counts) != {"earthquake", "icequake"}:
            raise ValueError(f"Missing one class at station {station}: {station_counts}")
    return counts


def write_map(output: Path, counts_path: Path, stations_path: Path) -> None:
    counts = read_counts(counts_path)
    metadata = parse_station_file(stations_path)
    missing = set(STATIONS) - set(metadata)
    if missing:
        raise ValueError(f"Missing station coordinates: {sorted(missing)}")

    width, height = 1540, 1080
    left, top = 110, 140
    plot_width, plot_height = 1030, 830
    right, bottom = left + plot_width, top + plot_height
    max_count = max(value for station in counts.values() for value in station.values())

    def px(longitude: float) -> float:
        return left + (longitude - WEST) / (EAST - WEST) * plot_width

    def py(latitude: float) -> float:
        return top + (NORTH - latitude) / (NORTH - SOUTH) * plot_height

    def bubble_radius(count: int) -> float:
        return 7.0 + 18.0 * math.sqrt(count / max_count)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        text(
            width / 2,
            43,
            "Earthquake and icequake records at the nine common receiving stations",
            text_anchor="middle",
            font_family="Arial",
            font_size="26",
            font_weight="bold",
            fill=INK,
        ),
        f'<rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" fill="#f8fafc" stroke="{INK}" stroke-width="1.5"/>',
    ]

    for longitude in range(-149, -143):
        x = px(longitude)
        lines.append(
            f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{bottom}" stroke="{GRID}" stroke-width="1"/>'
        )
        lines.append(
            text(
                x,
                bottom + 29,
                f"{abs(longitude)}°W",
                text_anchor="middle",
                font_family="Arial",
                font_size="14",
                fill="#4b5563",
            )
        )
    for latitude_times_two in range(120, 125):
        latitude = latitude_times_two / 2
        y = py(latitude)
        lines.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>'
        )
        lines.append(
            text(
                left - 15,
                y + 5,
                f"{latitude:g}°N",
                text_anchor="end",
                font_family="Arial",
                font_size="14",
                fill="#4b5563",
            )
        )

    glacier_x, glacier_y = px(GLACIER_LON), py(GLACIER_LAT)
    star_points = []
    for index in range(10):
        angle = -math.pi / 2 + index * math.pi / 5
        radius = 15 if index % 2 == 0 else 6.5
        star_points.append(
            f"{glacier_x + radius * math.cos(angle):.1f},{glacier_y + radius * math.sin(angle):.1f}"
        )
    lines.append(
        f'<polygon points="{" ".join(star_points)}" fill="#d9a400" stroke="{INK}" stroke-width="1.4"/>'
    )
    lines.append(
        text(
            glacier_x + 21,
            glacier_y + 5,
            "Columbia Glacier",
            font_family="Arial",
            font_size="14",
            font_weight="bold",
            fill=INK,
        )
    )

    for station in STATIONS:
        x = px(float(metadata[station]["longitude"]))
        y = py(float(metadata[station]["latitude"]))
        earthquake_n = counts[station]["earthquake"]
        icequake_n = counts[station]["icequake"]
        earthquake_r = bubble_radius(earthquake_n)
        icequake_r = bubble_radius(icequake_n)
        separation = max(earthquake_r, icequake_r) * 0.58
        lines.append(
            f'<circle cx="{x-separation:.1f}" cy="{y:.1f}" r="{earthquake_r:.1f}" fill="{EQ_COLOR}" fill-opacity="0.86" stroke="{INK}" stroke-width="1.1"/>'
        )
        lines.append(
            f'<circle cx="{x+separation:.1f}" cy="{y:.1f}" r="{icequake_r:.1f}" fill="{IQ_COLOR}" fill-opacity="0.86" stroke="{INK}" stroke-width="1.1"/>'
        )
        # The central diamond marks the shared physical receiver.
        diamond = f"{x:.1f},{y-7:.1f} {x+7:.1f},{y:.1f} {x:.1f},{y+7:.1f} {x-7:.1f},{y:.1f}"
        lines.append(
            f'<polygon points="{diamond}" fill="{RECEIVER_COLOR}" stroke="#ffffff" stroke-width="1.2"/>'
        )
        dx, dy = LABEL_OFFSETS[station]
        label_x, label_y = x + dx, y + dy
        lines.append(
            text(
                label_x,
                label_y,
                station,
                font_family="Arial",
                font_size="15",
                font_weight="bold",
                fill=INK,
            )
        )
        lines.append(
            text(
                label_x,
                label_y + 28,
                f"EQ {earthquake_n:,} | IQ {icequake_n:,}",
                font_family="Arial",
                font_size="12.5",
                fill="#4b5563",
            )
        )

    legend_x = right + 75
    lines.append(
        text(legend_x, top + 20, "Legend", font_family="Arial", font_size="21", font_weight="bold", fill=INK)
    )
    lines.append(
        f'<circle cx="{legend_x+20}" cy="{top+64}" r="18" fill="{EQ_COLOR}" fill-opacity="0.86" stroke="{INK}" stroke-width="1.1"/>'
    )
    lines.append(
        text(legend_x + 52, top + 70, "Earthquake record", font_family="Arial", font_size="16", fill=INK)
    )
    lines.append(
        f'<circle cx="{legend_x+20}" cy="{top+123}" r="18" fill="{IQ_COLOR}" fill-opacity="0.86" stroke="{INK}" stroke-width="1.1"/>'
    )
    lines.append(
        text(legend_x + 52, top + 129, "Icequake record", font_family="Arial", font_size="16", fill=INK)
    )
    legend_diamond = (
        f"{legend_x+20},{top+165} {legend_x+28},{top+173} "
        f"{legend_x+20},{top+181} {legend_x+12},{top+173}"
    )
    lines.append(
        f'<polygon points="{legend_diamond}" fill="{RECEIVER_COLOR}" stroke="{INK}" stroke-width="0.8"/>'
    )
    lines.append(
        text(legend_x + 52, top + 179, "Common receiver", font_family="Arial", font_size="16", fill=INK)
    )
    lines.append(
        text(legend_x, top + 228, "Bubble area", font_family="Arial", font_size="15", font_weight="bold", fill=INK)
    )
    lines.append(
        text(legend_x, top + 251, "scales with record count", font_family="Arial", font_size="14", fill="#4b5563")
    )
    totals = {
        event_class: sum(counts[station][event_class] for station in STATIONS)
        for event_class in ("earthquake", "icequake")
    }
    lines.append(
        text(legend_x, bottom - 116, "Totals", font_family="Arial", font_size="16", font_weight="bold", fill=INK)
    )
    lines.append(
        text(legend_x, bottom - 88, f"Earthquake: {totals['earthquake']:,}", font_family="Arial", font_size="14", fill=INK)
    )
    lines.append(
        text(legend_x, bottom - 64, f"Icequake: {totals['icequake']:,}", font_family="Arial", font_size="14", fill=INK)
    )
    lines.append(
        text(legend_x, bottom - 36, f"All records: {sum(totals.values()):,}", font_family="Arial", font_size="14", font_weight="bold", fill=INK)
    )
    lines.append(
        text(
            left,
            height - 32,
            "Receiver locations: data/stations.txt. Counts: data/station_class_counts.csv.",
            font_family="Arial",
            font_size="13",
            fill="#4b5563",
        )
    )
    lines.append("</svg>")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--counts",
        type=Path,
        default=Path("data/station_class_counts.csv"),
    )
    parser.add_argument("--station-file", type=Path, default=Path("data/stations.txt"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("generated_figures/figure_03_receiver_counts.svg"),
    )
    args = parser.parse_args()
    write_map(args.output, args.counts, args.station_file)
    print(args.output)


if __name__ == "__main__":
    main()
