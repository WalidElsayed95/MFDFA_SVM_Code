#!/usr/bin/env python3
"""Generate the event-and-station location map used for Figure 2.

This script requires the source earthquake and icequake catalogs, PyGMT, GMT,
and access to the GMT ``@earth_relief_30s`` remote grid. See README.md.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pygmt


ROOT = Path(__file__).resolve().parent


def build_map(
    stations_path: Path,
    earthquake_catalog: Path,
    icequake_catalog: Path,
    output_path: Path,
) -> None:
    stations = pd.read_csv(stations_path, sep="|", skiprows=[2, 6])
    station_names = stations.iloc[:, 1].astype(str).to_numpy()
    station_lats = stations.iloc[:, 2].astype(float).to_numpy()
    station_lons = stations.iloc[:, 3].astype(float).to_numpy()

    earthquakes = pd.read_csv(earthquake_catalog)
    icequakes = pd.read_csv(icequake_catalog)
    eq_lats, eq_lons = earthquakes.iloc[:, 1].to_numpy(), earthquakes.iloc[:, 2].to_numpy()
    iq_lats, iq_lons = icequakes.iloc[:, 1].to_numpy(), icequakes.iloc[:, 2].to_numpy()

    glacier_lat, glacier_lon = 61.219722, -146.895278
    region = [-149.0, -144.0, 60.0, 62.0]
    figure = pygmt.Figure()
    pygmt.makecpt(cmap="relief", series="-5000/5000/1000", continuous=True)
    figure.grdimage(
        grid="@earth_relief_30s",
        region=region,
        projection="M5i",
        shading=True,
        cmap=True,
        frame=["WSne", "xaf", "yaf"],
    )
    figure.coast(area_thresh=10000, shorelines="1p,black", region=region)
    figure.plot(x=eq_lons, y=eq_lats, style="c0.05c", fill="green", transparency=50)
    figure.plot(x=iq_lons, y=iq_lats, style="c0.05c", fill="blue", transparency=50)
    figure.plot(x=np.nan, y=np.nan, style="c0.4c", fill="green", label="Earthquakes")
    figure.plot(x=np.nan, y=np.nan, style="c0.4c", fill="blue", label="Icequakes")
    figure.plot(
        x=glacier_lon,
        y=glacier_lat,
        style="a0.6c",
        fill="yellow",
        pen="0.5p,black",
        label="Columbia Glacier",
    )
    figure.plot(
        x=station_lons,
        y=station_lats,
        style="i0.5c",
        fill="red",
        pen="1p,black",
        label="Seismic stations",
    )
    figure.text(
        text=station_names[3:],
        y=station_lats[3:] + 0.05,
        x=station_lons[3:] + 0.25,
        font="8p,Helvetica-Bold,black",
        fill="white",
    )
    with pygmt.config(FONT_ANNOT_PRIMARY="12p,Helvetica-Bold,black"):
        figure.legend(position="JTR+jTR+o0.2c", box=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stations", type=Path, default=ROOT / "data" / "stations.txt")
    parser.add_argument("--earthquakes", type=Path, required=True)
    parser.add_argument("--icequakes", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "generated_figures" / "figure_02_event_station_map.png",
    )
    args = parser.parse_args()
    build_map(args.stations, args.earthquakes, args.icequakes, args.output)


if __name__ == "__main__":
    main()
