"""Station metadata reader used by the receiver-count figure."""

from __future__ import annotations

from pathlib import Path

STATIONS = ("BAE", "BAT", "FID", "GLI", "KLU", "KNK", "M23K", "SAW", "SCM")

def parse_station_file(path: Path) -> dict[str, dict[str, str]]:
    metadata: dict[str, dict[str, str]] = {}
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or "DATACENTER" in line.upper() or "NETWORK" in line.upper():
            continue
        if line.startswith("#"):
            line = line[1:]
        fields = [field.strip() for field in line.split("|")]
        if len(fields) < 8:
            continue
        network, station, latitude, longitude, elevation, sitename, start, end = fields[:8]
        metadata[station] = {
            "network": network,
            "latitude": latitude,
            "longitude": longitude,
            "elevation_m": elevation,
            "sitename": sitename,
            "start": start,
            "end": end,
        }
    return metadata
