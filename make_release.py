#!/usr/bin/env python3
"""Create checksums and the clean Zenodo upload ZIP for this release."""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT.parent / "MFDFA_SVM_Reproducibility_Code_v1.0.1.zip"
CHECKSUMS = ROOT / "SHA256SUMS.txt"
EXCLUDED_PARTS = {"__pycache__", ".cache", ".ipynb_checkpoints", ".venv"}


def included_files(include_checksums: bool) -> list[Path]:
    files = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or any(part in EXCLUDED_PARTS for part in path.parts):
            continue
        if path.suffix in {".pyc", ".pyo"}:
            continue
        if not include_checksums and path == CHECKSUMS:
            continue
        files.append(path)
    return sorted(files, key=lambda item: item.relative_to(ROOT).as_posix())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    lines = [
        f"{sha256(path)}  {path.relative_to(ROOT).as_posix()}"
        for path in included_files(include_checksums=False)
    ]
    CHECKSUMS.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with zipfile.ZipFile(ARCHIVE, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in included_files(include_checksums=True):
            member = Path(ROOT.name) / path.relative_to(ROOT)
            archive.write(path, member.as_posix())
    print(f"Created: {ARCHIVE}")
    print(f"Files: {len(included_files(include_checksums=True))}")
    print(f"Archive SHA-256: {sha256(ARCHIVE)}")


if __name__ == "__main__":
    main()
