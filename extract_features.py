#!/usr/bin/env python3
"""Extract MFDFA features from the selected raw waveforms.

The archived derived feature table is complete, so the default invocation
validates it and proceeds directly to the nested SVM evaluation. To regenerate
the table from source waveforms, place ``all_waveforms.npy`` and
``all_labels.npy`` in ``data/raw/`` and remove or rename the existing derived
table. The raw arrays are not redistributed in this software release.
"""

from __future__ import annotations

import csv
import json
import time
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

import mfdfa as mf
import nested_svm as nested
from evaluate_models import finalize_analysis


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
FEATURE_TABLE = DATA_DIR / "mfdfa_features.csv"
SELECTION_TABLE = DATA_DIR / "selected_records.csv"
COUNT_TABLE = DATA_DIR / "station_class_counts.csv"
RESULTS_DIR = ROOT / "results"
CACHE_DIR = ROOT / ".cache"


@lru_cache(maxsize=None)
def design(window: int) -> tuple[np.ndarray, float]:
    x = np.arange(window, dtype=float)
    centered = x - x.mean()
    return centered, float(np.sum(centered**2))


def linear_trends(segments: np.ndarray, window: int, order: int = 1) -> np.ndarray:
    """Vectorized least-squares linear trends for all segments."""
    if order != 1:
        raise ValueError("This implementation supports polynomial order 1 only.")
    x, denominator = design(int(window))
    return segments.mean(axis=1)[:, None] + ((segments @ x) / denominator)[:, None] * x


def vectorized_fluctuation(
    segments: np.ndarray, trends: np.ndarray, q: np.ndarray
) -> np.ndarray:
    """Stable q-order fluctuation function, vectorized over q."""
    variance = np.mean((segments - trends) ** 2, axis=1)
    if np.any(~np.isfinite(variance)) or np.any(variance <= 0):
        raise ValueError("Nonpositive or nonfinite segment variance.")
    log_variance = np.log(variance)
    zero = q == 0
    result = np.empty(q.shape, dtype=float)
    result[zero] = np.exp(0.5 * np.mean(log_variance))
    terms = 0.5 * q[~zero, None] * log_variance[None, :]
    maxima = np.max(terms, axis=1)
    result[~zero] = np.exp(
        (maxima + np.log(np.mean(np.exp(terms - maxima[:, None]), axis=1))) / q[~zero]
    )
    return result


def initialize_worker(cache_path: str) -> None:
    global NUMERIC_WAVES
    NUMERIC_WAVES = np.load(cache_path, mmap_mode="r", allow_pickle=False)
    mf.get_trends = linear_trends
    mf._fractal_dfa_fluctuation = vectorized_fluctuation


def extract_one(task: tuple[int, int, str, int]) -> dict[str, object]:
    index, serial_no, station, label = task
    signal = np.asarray(NUMERIC_WAVES[index], dtype=float)
    if not np.isfinite(signal).all():
        raise ValueError(f"Nonfinite original samples at serial {serial_no}.")
    values = mf.MFDFA(signal, np.arange(-5.0, 6.0))
    arrays = np.concatenate([np.atleast_1d(value) for value in values.values()])
    if not np.isfinite(arrays).all():
        raise ValueError(f"Nonfinite MFDFA feature at serial {serial_no}.")
    result = {
        key: json.dumps(value.tolist()) if isinstance(value, np.ndarray) else float(value)
        for key, value in values.items()
    }
    result.update(label=int(label), station=station, serial_no=int(serial_no))
    return result


def validate_selection(selected: pd.DataFrame) -> pd.DataFrame:
    required = {"label", "station", "serial_no"}
    if not required.issubset(selected.columns):
        raise ValueError(f"Selection table must contain {sorted(required)}.")
    selected = selected.loc[selected["station"].isin(nested.STATIONS)].copy()
    if len(selected) != 28_022 or not selected["serial_no"].is_unique:
        raise ValueError("Expected 28,022 rows with unique serial_no values.")
    return selected


def finalize_existing_table(selected: pd.DataFrame) -> bool:
    if not FEATURE_TABLE.is_file():
        return False
    completed = pd.read_csv(FEATURE_TABLE)
    metadata_match = completed[["label", "station", "serial_no"]].astype(str).equals(
        selected[["label", "station", "serial_no"]].astype(str)
    )
    if len(completed) == len(selected) and completed["serial_no"].is_unique and metadata_match:
        print("The archived feature table is complete; extraction is not repeated.", flush=True)
        finalize_analysis(FEATURE_TABLE, COUNT_TABLE, RESULTS_DIR, n_jobs=1)
        return True
    return False


def prepare_numeric_cache(selected: pd.DataFrame, labels: np.ndarray) -> Path:
    wave_path = RAW_DIR / "all_waveforms.npy"
    if not wave_path.is_file():
        raise FileNotFoundError(
            f"Raw waveform file not found: {wave_path}. See README.md for the required format."
        )
    CACHE_DIR.mkdir(exist_ok=True)
    cache = CACHE_DIR / "selected_waveforms_float64.npy"
    metadata = CACHE_DIR / "selected_waveforms_cache.json"
    ids = selected["serial_no"].to_numpy(dtype=int)
    if cache.is_file() and metadata.is_file():
        saved = json.loads(metadata.read_text(encoding="utf-8"))
        if saved.get("serial_numbers") == ids.tolist():
            return cache

    waves = np.load(wave_path, allow_pickle=True)
    if waves.shape != (len(labels), 3001):
        raise ValueError(f"Expected raw waveform shape ({len(labels)}, 3001), found {waves.shape}.")
    np.testing.assert_array_equal(waves[ids, -1].astype(str), selected["station"].to_numpy())
    numeric = np.lib.format.open_memmap(
        cache, mode="w+", dtype="float64", shape=(len(selected), 3000), fortran_order=True
    )
    for column in range(3000):
        numeric[:, column] = np.asarray(waves[:, column], dtype=float)[ids]
        if column % 500 == 0:
            print(f"Cached {column}/3000 sample columns", flush=True)
    numeric.flush()
    metadata.write_text(
        json.dumps({"serial_numbers": ids.tolist(), "shape": [len(selected), 3000]}),
        encoding="utf-8",
    )
    return cache


def main() -> None:
    started = time.monotonic()
    selected = validate_selection(pd.read_csv(SELECTION_TABLE))
    if finalize_existing_table(selected):
        print("COMPLETE", flush=True)
        return

    label_path = RAW_DIR / "all_labels.npy"
    if not label_path.is_file():
        raise FileNotFoundError(
            f"Raw label file not found: {label_path}. See README.md for the required format."
        )
    labels = np.load(label_path, allow_pickle=False)
    ids = selected["serial_no"].to_numpy(dtype=int)
    np.testing.assert_array_equal(labels[ids], selected["label"].to_numpy())
    cache = prepare_numeric_cache(selected, labels)

    numeric = np.load(cache, mmap_mode="r", allow_pickle=False)
    sample = np.asarray(numeric[0], dtype=float)
    reference = mf.MFDFA(sample, np.arange(-5.0, 6.0))
    mf.get_trends = linear_trends
    mf._fractal_dfa_fluctuation = vectorized_fluctuation
    optimized = mf.MFDFA(sample, np.arange(-5.0, 6.0))
    for key in reference:
        np.testing.assert_allclose(reference[key], optimized[key], rtol=1e-8, atol=1e-9)

    tasks = [
        (index, int(row.serial_no), row.station, int(row.label))
        for index, row in enumerate(selected.itertuples(index=False))
    ]
    with FEATURE_TABLE.open("w", newline="", encoding="utf-8") as handle, ProcessPoolExecutor(
        max_workers=3, initializer=initialize_worker, initargs=(str(cache),)
    ) as pool:
        writer = None
        for count, result in enumerate(pool.map(extract_one, tasks, chunksize=16), start=1):
            if writer is None:
                writer = csv.DictWriter(handle, fieldnames=list(result))
                writer.writeheader()
            writer.writerow(result)
            if count % 100 == 0:
                handle.flush()
                print(
                    f"Extracted {count}/28022; elapsed {time.monotonic() - started:.1f}s",
                    flush=True,
                )

    completed = pd.read_csv(FEATURE_TABLE)
    if len(completed) != 28_022 or not completed["serial_no"].is_unique:
        raise AssertionError("Feature extraction did not produce 28,022 unique records.")
    finalize_analysis(FEATURE_TABLE, COUNT_TABLE, RESULTS_DIR, n_jobs=1)
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    main()
