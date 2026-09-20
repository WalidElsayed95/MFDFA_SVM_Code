#!/usr/bin/env python3
"""Station-specific nested 10 x 10 cross-validation for the MFDFA feature table.

The nine manuscript stations are modeled independently, metadata and
non-numeric array columns are excluded, and the 15 scalar numeric MFDFA
descriptors are used as predictors. The validation procedure is:

* outer 10 folds: held-out performance estimation;
* inner 10 folds: GridSearchCV on the outer-training partition only;
* StandardScaler and SVC are kept in one Pipeline;
* the headline result is the unweighted mean of nine station accuracies.

The feature-table ``serial_no`` column is a unique row identifier, not an event
group identifier, and is never used as a predictor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


STATIONS = ("BAE", "BAT", "FID", "GLI", "KLU", "KNK", "M23K", "SAW", "SCM")

# Numeric predictors after label, station, serial_no, and array-valued columns
# are excluded.
FINAL_SVC_NUMERIC_FEATURES = (
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

PARAM_GRID = {
    "svc__C": (10, 100, 1000),
    "svc__gamma": ("scale", 0.01, 0.001),
    "svc__kernel": ("rbf",),
}

OUTER_FOLDS = 10
INNER_FOLDS = 10
RANDOM_STATE = 101


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_and_validate_data(input_csv: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = pd.read_csv(input_csv)
    required = {"label", "station", "serial_no", *FINAL_SVC_NUMERIC_FEATURES}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    selected = frame.loc[frame["station"].isin(STATIONS)].copy()
    if len(selected) != 28_022:
        raise ValueError(f"Expected 28,022 rows at the nine stations, found {len(selected):,}")
    if set(selected["station"]) != set(STATIONS):
        raise ValueError("The selected station set does not match the manuscript stations")
    if set(selected["label"].dropna().unique()) != {0.0, 1.0}:
        raise ValueError(f"Unexpected labels: {sorted(selected['label'].dropna().unique())}")
    if selected["serial_no"].duplicated().any():
        raise ValueError("serial_no is not unique at the feature-row grain")

    numeric_selected_like_original = (
        selected.drop(columns=["label", "station", "serial_no"])
        .select_dtypes(include=[np.number])
        .columns.tolist()
    )
    if numeric_selected_like_original != list(FINAL_SVC_NUMERIC_FEATURES):
        raise ValueError(
            "The explicit feature list no longer matches Final_SVC numeric selection: "
            f"discovered={numeric_selected_like_original}"
        )

    feature_values = selected.loc[:, FINAL_SVC_NUMERIC_FEATURES].to_numpy(dtype=float)
    finite_rows = np.isfinite(feature_values).all(axis=1)
    removed_nonfinite = int((~finite_rows).sum())
    selected = selected.loc[finite_rows].copy()
    selected["label"] = selected["label"].astype(int)

    station_label_counts = (
        selected.groupby(["station", "label"], observed=False)
        .size()
        .unstack(fill_value=0)
        .reindex(index=STATIONS, columns=[0, 1], fill_value=0)
    )
    if (station_label_counts.min(axis=1) < max(OUTER_FOLDS, INNER_FOLDS)).any():
        raise ValueError("At least one station/class cannot support 10 stratified folds")

    profile = {
        "input_csv": input_csv.as_posix(),
        "input_sha256": sha256_file(input_csv),
        "full_rows": int(len(frame)),
        "selected_rows": int(len(selected)),
        "removed_nonfinite_rows": removed_nonfinite,
        "stations": list(STATIONS),
        "feature_count": len(FINAL_SVC_NUMERIC_FEATURES),
        "feature_columns": list(FINAL_SVC_NUMERIC_FEATURES),
        "serial_no_unique_rows": int(selected["serial_no"].nunique()),
        "station_label_counts": {
            station: {
                "label_0": int(station_label_counts.loc[station, 0]),
                "label_1": int(station_label_counts.loc[station, 1]),
            }
            for station in STATIONS
        },
    }
    return selected, profile


def make_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("svc", SVC(class_weight="balanced")),
        ]
    )


def calculate_fold_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "sensitivity_label_1": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "specificity_label_0": float(tn / (tn + fp)) if (tn + fp) else math.nan,
        "precision_label_1": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_label_1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def run_station_nested_cv(
    station_frame: pd.DataFrame,
    station: str,
    n_jobs: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    X = station_frame.loc[:, FINAL_SVC_NUMERIC_FEATURES]
    y = station_frame["label"].to_numpy(dtype=int)
    serial_numbers = station_frame["serial_no"].to_numpy()
    source_indices = station_frame.index.to_numpy()

    outer_cv = StratifiedKFold(
        n_splits=OUTER_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    fold_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []

    for outer_fold, (train_indices, test_indices) in enumerate(outer_cv.split(X, y), start=1):
        fold_start = time.perf_counter()
        inner_cv = StratifiedKFold(
            n_splits=INNER_FOLDS,
            shuffle=True,
            random_state=RANDOM_STATE,
        )
        search = GridSearchCV(
            estimator=make_pipeline(),
            param_grid=PARAM_GRID,
            scoring="accuracy",
            cv=inner_cv,
            n_jobs=n_jobs,
            refit=True,
            return_train_score=False,
            error_score="raise",
        )
        # Critical leakage control: GridSearchCV only receives the outer-training data.
        search.fit(X.iloc[train_indices], y[train_indices])
        predictions = search.best_estimator_.predict(X.iloc[test_indices])
        metrics = calculate_fold_metrics(y[test_indices], predictions)
        elapsed = time.perf_counter() - fold_start

        fold_rows.append(
            {
                "station": station,
                "outer_fold": outer_fold,
                "n_train": int(len(train_indices)),
                "n_test": int(len(test_indices)),
                **metrics,
                "inner_best_accuracy": float(search.best_score_),
                "best_C": search.best_params_["svc__C"],
                "best_gamma": search.best_params_["svc__gamma"],
                "best_kernel": search.best_params_["svc__kernel"],
                "runtime_seconds": elapsed,
            }
        )

        for local_index, prediction in zip(test_indices, predictions, strict=True):
            prediction_rows.append(
                {
                    "station": station,
                    "outer_fold": outer_fold,
                    "source_row_index": int(source_indices[local_index]),
                    "serial_no": int(serial_numbers[local_index]),
                    "label_true": int(y[local_index]),
                    "label_predicted": int(prediction),
                    "correct": int(y[local_index] == prediction),
                }
            )

        candidate_frame = pd.DataFrame(search.cv_results_)
        for _, candidate in candidate_frame.iterrows():
            candidate_rows.append(
                {
                    "station": station,
                    "outer_fold": outer_fold,
                    "C": candidate["param_svc__C"],
                    "gamma": candidate["param_svc__gamma"],
                    "kernel": candidate["param_svc__kernel"],
                    "mean_inner_accuracy": float(candidate["mean_test_score"]),
                    "std_inner_accuracy": float(candidate["std_test_score"]),
                    "rank_inner_accuracy": int(candidate["rank_test_score"]),
                    "mean_fit_time_seconds": float(candidate["mean_fit_time"]),
                }
            )

        print(
            f"{station} outer_fold={outer_fold}/{OUTER_FOLDS} "
            f"accuracy={metrics['accuracy']:.4f} "
            f"best_C={search.best_params_['svc__C']} "
            f"best_gamma={search.best_params_['svc__gamma']} "
            f"elapsed={elapsed:.1f}s",
            flush=True,
        )

    return (
        pd.DataFrame(fold_rows),
        pd.DataFrame(prediction_rows),
        pd.DataFrame(candidate_rows),
    )


def summarize_results(
    fold_results: pd.DataFrame,
    predictions: pd.DataFrame,
    data_profile: dict[str, Any],
    runtime_seconds: float,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    summaries: list[dict[str, Any]] = []
    for station in STATIONS:
        station_folds = fold_results.loc[fold_results["station"] == station]
        station_predictions = predictions.loc[predictions["station"] == station]
        best_pairs = Counter(
            zip(station_folds["best_C"], station_folds["best_gamma"], strict=True)
        )
        modal_pair, modal_count = best_pairs.most_common(1)[0]
        summaries.append(
            {
                "station": station,
                "n_waveforms": int(len(station_predictions)),
                "label_0": data_profile["station_label_counts"][station]["label_0"],
                "label_1": data_profile["station_label_counts"][station]["label_1"],
                "outer_accuracy_mean": float(station_folds["accuracy"].mean()),
                "outer_accuracy_std": float(station_folds["accuracy"].std(ddof=1)),
                "oof_accuracy": float(station_predictions["correct"].mean()),
                "outer_balanced_accuracy_mean": float(station_folds["balanced_accuracy"].mean()),
                "outer_sensitivity_mean": float(station_folds["sensitivity_label_1"].mean()),
                "outer_specificity_mean": float(station_folds["specificity_label_0"].mean()),
                "outer_f1_label_1_mean": float(station_folds["f1_label_1"].mean()),
                "modal_best_C": modal_pair[0],
                "modal_best_gamma": modal_pair[1],
                "modal_selection_folds": int(modal_count),
            }
        )
    station_summary = pd.DataFrame(summaries)

    parameter_frequency = (
        fold_results.groupby(["station", "best_C", "best_gamma"], dropna=False)
        .size()
        .rename("outer_folds_selected")
        .reset_index()
        .sort_values(["station", "outer_folds_selected", "best_C"], ascending=[True, False, True])
    )
    station_macro_accuracy = float(station_summary["outer_accuracy_mean"].mean())
    oof_station_macro_accuracy = float(station_summary["oof_accuracy"].mean())
    waveform_weighted_accuracy = float(predictions["correct"].mean())
    report = {
        "status": "PASS",
        "method": "station-wise nested stratified cross-validation",
        "outer_folds": OUTER_FOLDS,
        "inner_folds": INNER_FOLDS,
        "random_state": RANDOM_STATE,
        "scoring": "accuracy",
        "stations_modeled_independently": True,
        "cross_station_pooling": False,
        "feature_count": len(FINAL_SVC_NUMERIC_FEATURES),
        "feature_columns": list(FINAL_SVC_NUMERIC_FEATURES),
        "station_macro_accuracy": station_macro_accuracy,
        "station_macro_accuracy_percent": 100 * station_macro_accuracy,
        "oof_station_macro_accuracy": oof_station_macro_accuracy,
        "waveform_weighted_oof_accuracy": waveform_weighted_accuracy,
        "total_outer_predictions": int(len(predictions)),
        "total_outer_folds": int(len(fold_results)),
        "runtime_seconds": float(runtime_seconds),
        "data_profile": data_profile,
        "important_caveat": (
            "This exact Final_SVC reproduction uses 15 numeric predictors. "
            "It must not be described as a ten-feature model without rerunning "
            "the nested analysis using an explicit ten-feature list."
        ),
    }
    return station_summary, report, parameter_frequency


def output_is_complete(output_dir: Path, input_hash: str) -> bool:
    required = (
        "nested_fold_results.csv",
        "nested_oof_predictions.csv",
        "nested_grid_candidates.csv",
        "station_summary.csv",
        "parameter_selection_frequency.csv",
        "nested_summary.json",
    )
    if not all((output_dir / name).is_file() for name in required):
        return False
    try:
        summary = json.loads((output_dir / "nested_summary.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        summary.get("status") == "PASS"
        and summary.get("outer_folds") == OUTER_FOLDS
        and summary.get("inner_folds") == INNER_FOLDS
        and summary.get("data_profile", {}).get("input_sha256") == input_hash
        and summary.get("total_outer_predictions") == 28_022
        and summary.get("total_outer_folds") == len(STATIONS) * OUTER_FOLDS
    )


def load_saved_results(output_dir: Path) -> dict[str, Any]:
    return {
        "fold_results": pd.read_csv(output_dir / "nested_fold_results.csv"),
        "predictions": pd.read_csv(output_dir / "nested_oof_predictions.csv"),
        "grid_candidates": pd.read_csv(output_dir / "nested_grid_candidates.csv"),
        "station_summary": pd.read_csv(output_dir / "station_summary.csv"),
        "parameter_frequency": pd.read_csv(output_dir / "parameter_selection_frequency.csv"),
        "summary": json.loads((output_dir / "nested_summary.json").read_text(encoding="utf-8")),
    }


def run_analysis(
    input_csv: Path | str = Path("data/mfdfa_features.csv"),
    output_dir: Path | str = Path("results"),
    n_jobs: int = -1,
    reuse_existing: bool = True,
) -> dict[str, Any]:
    input_csv = Path(input_csv)
    output_dir = Path(output_dir)
    data, data_profile = load_and_validate_data(input_csv)
    if reuse_existing and output_is_complete(output_dir, data_profile["input_sha256"]):
        print(f"Reusing validated nested results in {output_dir}", flush=True)
        return load_saved_results(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "station_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    analysis_start = time.perf_counter()
    all_folds: list[pd.DataFrame] = []
    all_predictions: list[pd.DataFrame] = []
    all_candidates: list[pd.DataFrame] = []

    for station in STATIONS:
        fold_path = checkpoint_dir / f"{station}_folds.csv"
        prediction_path = checkpoint_dir / f"{station}_predictions.csv"
        candidate_path = checkpoint_dir / f"{station}_candidates.csv"
        if reuse_existing and all(path.is_file() for path in (fold_path, prediction_path, candidate_path)):
            station_folds = pd.read_csv(fold_path)
            station_predictions = pd.read_csv(prediction_path)
            station_candidates = pd.read_csv(candidate_path)
            valid_checkpoint = (
                len(station_folds) == OUTER_FOLDS
                and len(station_predictions) == int((data["station"] == station).sum())
                and station_predictions["serial_no"].nunique() == len(station_predictions)
            )
            if valid_checkpoint:
                print(f"Reusing completed station checkpoint: {station}", flush=True)
                all_folds.append(station_folds)
                all_predictions.append(station_predictions)
                all_candidates.append(station_candidates)
                continue

        station_frame = data.loc[data["station"] == station].copy()
        station_folds, station_predictions, station_candidates = run_station_nested_cv(
            station_frame,
            station,
            n_jobs,
        )
        station_folds.to_csv(fold_path, index=False)
        station_predictions.to_csv(prediction_path, index=False)
        station_candidates.to_csv(candidate_path, index=False)
        all_folds.append(station_folds)
        all_predictions.append(station_predictions)
        all_candidates.append(station_candidates)

    fold_results = pd.concat(all_folds, ignore_index=True)
    predictions = pd.concat(all_predictions, ignore_index=True)
    grid_candidates = pd.concat(all_candidates, ignore_index=True)
    runtime_seconds = time.perf_counter() - analysis_start

    if len(fold_results) != len(STATIONS) * OUTER_FOLDS:
        raise AssertionError("Nested fold table does not contain 90 outer folds")
    if len(predictions) != len(data):
        raise AssertionError("The outer-fold predictions do not cover every selected row")
    if predictions["serial_no"].nunique() != len(predictions):
        raise AssertionError("Outer-fold predictions contain duplicate or missing feature rows")

    station_summary, summary, parameter_frequency = summarize_results(
        fold_results,
        predictions,
        data_profile,
        runtime_seconds,
    )
    fold_results.to_csv(output_dir / "nested_fold_results.csv", index=False)
    predictions.to_csv(output_dir / "nested_oof_predictions.csv", index=False)
    grid_candidates.to_csv(output_dir / "nested_grid_candidates.csv", index=False)
    station_summary.to_csv(output_dir / "station_summary.csv", index=False)
    parameter_frequency.to_csv(output_dir / "parameter_selection_frequency.csv", index=False)
    (output_dir / "nested_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(
        f"STATION-MACRO ACCURACY: {summary['station_macro_accuracy']:.6f} "
        f"({summary['station_macro_accuracy_percent']:.3f}%)",
        flush=True,
    )
    return {
        "fold_results": fold_results,
        "predictions": predictions,
        "grid_candidates": grid_candidates,
        "station_summary": station_summary,
        "parameter_frequency": parameter_frequency,
        "summary": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=Path("data/mfdfa_features.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--no-reuse", action="store_true")
    args = parser.parse_args()
    run_analysis(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        n_jobs=args.n_jobs,
        reuse_existing=not args.no_reuse,
    )


if __name__ == "__main__":
    main()
