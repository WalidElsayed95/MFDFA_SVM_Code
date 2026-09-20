#!/usr/bin/env python3
"""Finalize metrics and figures for the 15-feature nested 10x10 SVC run."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn

import nested_svm as nested
from reporting import (
    build_station_outputs,
    draw_confusion_matrices,
    draw_station_metrics,
    read_expected_counts,
    validate_label_mapping,
)

ROOT = Path(__file__).resolve().parent


def portable_path(path: Path) -> str:
    """Return a package-relative path when the target is inside this package."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


FEATURES = (
    "Delta_alpha",
    "Delta_f",
    "f_max",
    "f_min",
    "alpha_max",
    "alpha_min",
    "mean_alpha",
    "alpha_0",
    "Delta_alpha_right",
    "Delta_alpha_left",
    "Delta_s",
    "A",
    "D0",
    "D1",
    "D2",
)
EXPECTED_TOTAL = 28_022
DEFAULT_INPUT = ROOT / "data" / "mfdfa_features.csv"
DEFAULT_COUNTS = ROOT / "data" / "station_class_counts.csv"
DEFAULT_OUTPUT = ROOT / "results"


def validate_outputs(
    fold_results: pd.DataFrame,
    predictions: pd.DataFrame,
    grid_candidates: pd.DataFrame,
    metrics: pd.DataFrame,
    confusion: pd.DataFrame,
    balance: pd.DataFrame,
    pooled: np.ndarray,
    summary: dict[str, Any],
) -> dict[str, Any]:
    checks = {
        "exactly_28022_oof_predictions": len(predictions) == EXPECTED_TOTAL,
        "every_serial_no_predicted_once": (
            predictions["serial_no"].nunique() == EXPECTED_TOTAL
            and predictions.groupby("serial_no").size().eq(1).all()
        ),
        "nine_independent_stations": set(predictions["station"]) == set(nested.STATIONS),
        "ten_outer_folds_per_station": (
            len(fold_results) == 90
            and fold_results.groupby("station")["outer_fold"].nunique().eq(10).all()
        ),
        "nine_grid_candidates_per_outer_fold": (
            len(grid_candidates) == 810
            and grid_candidates.groupby(["station", "outer_fold"]).size().eq(9).all()
        ),
        "confusion_counts_cover_all_rows": int(pooled.sum()) == EXPECTED_TOTAL,
        "station_confusions_reconcile": int(confusion["total"].sum()) == EXPECTED_TOTAL,
        "station_balance_reconciles": int(balance["total_records"].sum()) == EXPECTED_TOTAL,
        "fifteen_features_only": summary["feature_columns"] == list(FEATURES),
        "nested_inner_and_outer_10": summary["inner_folds"] == 10 and summary["outer_folds"] == 10,
        "no_cross_station_pooling": (
            summary["stations_modeled_independently"] is True
            and summary["cross_station_pooling"] is False
        ),
        "metrics_in_unit_interval": bool(
            metrics.drop(columns=["station", "n_waveforms"])
            .apply(lambda column: column.between(0, 1).all())
            .all()
        ),
    }
    checks = {name: bool(value) for name, value in checks.items()}
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise AssertionError(f"Validation failures: {failures}")
    return {
        "status": "PASS",
        "checks": checks,
        "input_rows": EXPECTED_TOTAL,
        "stations": list(nested.STATIONS),
        "features": list(FEATURES),
        "outer_folds": 10,
        "inner_folds": 10,
        "total_outer_folds": 90,
        "grid_candidates_evaluated_across_outer_folds": 810,
        "total_oof_predictions": EXPECTED_TOTAL,
        "pooled_confusion_matrix_descriptive_only": pooled.tolist(),
        "note": "Pooled counts summarize predictions from nine separate models; no pooled model was trained.",
    }


def finalize_analysis(
    input_csv: Path = DEFAULT_INPUT,
    count_table: Path = DEFAULT_COUNTS,
    output_dir: Path = DEFAULT_OUTPUT,
    n_jobs: int = -1,
) -> dict[str, Any]:
    expected_counts = read_expected_counts(count_table)
    validate_label_mapping(input_csv, expected_counts)
    if tuple(nested.FINAL_SVC_NUMERIC_FEATURES) != FEATURES:
        raise AssertionError("The reusable nested engine no longer has the expected 15 features")

    results = nested.run_analysis(
        input_csv=input_csv,
        output_dir=output_dir,
        n_jobs=n_jobs,
        reuse_existing=True,
    )
    fold_results = results["fold_results"]
    predictions = results["predictions"]
    grid_candidates = results["grid_candidates"]
    station_summary = results["station_summary"]
    summary = results["summary"]
    summary["class_label_mapping"] = {"0": "earthquake", "1": "icequake"}
    summary["feature_set_definition"] = "All 15 scalar numeric MFDFA descriptors"
    summary["important_caveat"] = (
        "serial_no is a unique feature-row identifier, not a verified cross-station event ID. "
        "No cross-station event leakage is possible because stations are never pooled."
    )
    summary["data_profile"]["input_csv"] = portable_path(input_csv)
    (output_dir / "nested_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )

    metrics, confusion, balance, pooled = build_station_outputs(
        predictions, station_summary, expected_counts
    )
    metrics.to_csv(output_dir / "station_oof_metrics.csv", index=False)
    confusion.to_csv(output_dir / "station_confusion_counts.csv", index=False)
    balance.to_csv(output_dir / "station_class_balance.csv", index=False)
    pd.DataFrame(
        pooled,
        index=["actual_earthquake", "actual_icequake"],
        columns=["predicted_earthquake", "predicted_icequake"],
    ).to_csv(output_dir / "summed_oof_confusion_descriptive_only.csv")

    aggregate = {
        "station_macro_accuracy": float(metrics["accuracy"].mean()),
        "station_macro_balanced_accuracy": float(metrics["balanced_accuracy"].mean()),
        "station_macro_precision": float(metrics["macro_precision"].mean()),
        "station_macro_recall": float(metrics["macro_recall"].mean()),
        "station_macro_f1": float(metrics["macro_f1"].mean()),
        "waveform_weighted_oof_accuracy_descriptive_only": float(predictions["correct"].mean()),
        "earthquake_records": int(pooled[0].sum()),
        "icequake_records": int(pooled[1].sum()),
        "total_records": int(pooled.sum()),
        "summed_confusion_matrix_descriptive_only": pooled.tolist(),
    }
    (output_dir / "aggregate_metrics.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True), encoding="utf-8"
    )
    draw_confusion_matrices(predictions, output_dir / "confusion_matrices_9_stations.png")
    draw_station_metrics(
        metrics,
        output_dir / "station_metrics_15features.png",
        aggregate["station_macro_accuracy"],
    )
    validation = validate_outputs(
        fold_results,
        predictions,
        grid_candidates,
        metrics,
        confusion,
        balance,
        pooled,
        summary,
    )
    (output_dir / "validation_report.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True), encoding="utf-8"
    )
    provenance = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "input_csv": portable_path(input_csv),
        "input_sha256": summary["data_profile"]["input_sha256"],
        "count_table": portable_path(count_table),
        "output_directory": portable_path(output_dir),
    }
    (output_dir / "software_and_input_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({"aggregate_metrics": aggregate, "validation": validation}, indent=2))
    return {
        **results,
        "metrics": metrics,
        "confusion": confusion,
        "balance": balance,
        "aggregate": aggregate,
        "validation": validation,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--count-table",
        type=Path,
        default=DEFAULT_COUNTS,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args()
    finalize_analysis(args.input_csv, args.count_table, args.output_dir, args.n_jobs)


if __name__ == "__main__":
    main()
