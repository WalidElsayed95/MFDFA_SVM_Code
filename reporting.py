"""Shared metrics and plots used by the final 15-feature analysis."""

from __future__ import annotations

import csv

from pathlib import Path

from typing import Any

import matplotlib.pyplot as plt

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

import nested_svm as nested

CLASS_NAMES = {0: "earthquake", 1: "icequake"}

EXPECTED_TOTAL = 28_022

def read_expected_counts(path: Path) -> dict[tuple[str, str], int]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    counts = {
        (row["station"], row["class"]): int(row["feature_table_target"])
        for row in rows
    }
    expected_keys = {
        (station, event_class)
        for station in nested.STATIONS
        for event_class in CLASS_NAMES.values()
    }
    if set(counts) != expected_keys:
        raise ValueError(f"Expected-count keys differ: {sorted(set(counts) ^ expected_keys)}")
    if sum(counts.values()) != EXPECTED_TOTAL:
        raise ValueError(f"Expected counts sum to {sum(counts.values()):,}, not {EXPECTED_TOTAL:,}")
    return counts

def validate_label_mapping(
    input_csv: Path,
    expected_counts: dict[tuple[str, str], int],
) -> None:
    frame = pd.read_csv(input_csv, usecols=["station", "label"])
    frame = frame.loc[frame["station"].isin(nested.STATIONS)].copy()
    observed = frame.groupby(["station", "label"]).size()
    for station in nested.STATIONS:
        for label, event_class in CLASS_NAMES.items():
            actual = int(observed.loc[(station, float(label))])
            expected = expected_counts[(station, event_class)]
            if actual != expected:
                raise ValueError(
                    f"Label mapping failed at {station}: label {label} has {actual}, "
                    f"but {event_class} target is {expected}"
                )

def class_metrics(y_true: np.ndarray, y_pred: np.ndarray, label: int) -> dict[str, float]:
    return {
        "precision": float(precision_score(y_true, y_pred, pos_label=label, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=label, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=label, zero_division=0)),
    }

def build_station_outputs(
    predictions: pd.DataFrame,
    station_summary: pd.DataFrame,
    expected_counts: dict[tuple[str, str], int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray]:
    metric_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    balance_rows: list[dict[str, Any]] = []
    pooled = np.zeros((2, 2), dtype=int)

    for station in nested.STATIONS:
        station_predictions = predictions.loc[predictions["station"] == station]
        y_true = station_predictions["label_true"].to_numpy(dtype=int)
        y_pred = station_predictions["label_predicted"].to_numpy(dtype=int)
        matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
        pooled += matrix
        eq = class_metrics(y_true, y_pred, 0)
        iq = class_metrics(y_true, y_pred, 1)
        macro_precision = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
        macro_recall = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
        macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        metric_rows.append(
            {
                "station": station,
                "n_waveforms": int(len(y_true)),
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
                "earthquake_precision": eq["precision"],
                "earthquake_recall": eq["recall"],
                "earthquake_f1": eq["f1"],
                "icequake_precision": iq["precision"],
                "icequake_recall": iq["recall"],
                "icequake_f1": iq["f1"],
                "macro_precision": macro_precision,
                "macro_recall": macro_recall,
                "macro_f1": macro_f1,
            }
        )
        confusion_rows.append(
            {
                "station": station,
                "actual_earthquake_predicted_earthquake": int(matrix[0, 0]),
                "actual_earthquake_predicted_icequake": int(matrix[0, 1]),
                "actual_icequake_predicted_earthquake": int(matrix[1, 0]),
                "actual_icequake_predicted_icequake": int(matrix[1, 1]),
                "total": int(matrix.sum()),
            }
        )

        earthquake_n = expected_counts[(station, "earthquake")]
        icequake_n = expected_counts[(station, "icequake")]
        total = earthquake_n + icequake_n
        exact_balance_discard = abs(earthquake_n - icequake_n)
        exact_balance_retained = 2 * min(earthquake_n, icequake_n)
        balance_rows.append(
            {
                "station": station,
                "earthquake_records": earthquake_n,
                "icequake_records": icequake_n,
                "total_records": total,
                "earthquake_percent": 100 * earthquake_n / total,
                "icequake_percent": 100 * icequake_n / total,
                "records_discarded_if_exact_station_balance": exact_balance_discard,
                "records_retained_if_exact_station_balance": exact_balance_retained,
                "full_station_reference_weight_earthquake": total / (2 * earthquake_n),
                "full_station_reference_weight_icequake": total / (2 * icequake_n),
                "training_rule": "SVC(class_weight='balanced') within each training fold",
            }
        )

    metrics = pd.DataFrame(metric_rows)
    confusion = pd.DataFrame(confusion_rows)
    balance = pd.DataFrame(balance_rows)
    station_macro_from_oof = float(metrics["accuracy"].mean())
    saved_station_macro = float(station_summary["oof_accuracy"].mean())
    if not np.isclose(station_macro_from_oof, saved_station_macro):
        raise AssertionError("OOF station-macro accuracy does not reconcile")
    return metrics, confusion, balance, pooled

def draw_confusion_matrices(predictions: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(13.5, 12.5))
    labels = ["Earthquake", "Icequake"]
    for station, axis in zip(nested.STATIONS, axes.flat, strict=True):
        station_predictions = predictions.loc[predictions["station"] == station]
        matrix = confusion_matrix(
            station_predictions["label_true"],
            station_predictions["label_predicted"],
            labels=[0, 1],
        )
        row_totals = matrix.sum(axis=1, keepdims=True)
        row_percent = np.divide(
            matrix,
            row_totals,
            out=np.zeros_like(matrix, dtype=float),
            where=row_totals != 0,
        )
        axis.imshow(row_percent, cmap="Blues", vmin=0, vmax=1)
        for row in range(2):
            for column in range(2):
                color = "white" if row_percent[row, column] > 0.55 else "#172033"
                axis.text(
                    column,
                    row,
                    f"{matrix[row, column]:,}\n({100 * row_percent[row, column]:.1f}%)",
                    ha="center",
                    va="center",
                    fontsize=11,
                    color=color,
                    fontweight="bold",
                )
        accuracy = accuracy_score(
            station_predictions["label_true"], station_predictions["label_predicted"]
        )
        axis.set_title(f"{station} — accuracy {100 * accuracy:.2f}%", fontweight="bold")
        axis.set_xticks([0, 1], labels=labels, rotation=15)
        axis.set_yticks([0, 1], labels=labels)
        axis.set_xlabel("Predicted class")
        axis.set_ylabel("Actual class")
    fig.suptitle(
        "Out-of-fold confusion matrices: station-wise nested 10×10 SVC",
        fontsize=17,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)

def draw_station_metrics(metrics: pd.DataFrame, output: Path, macro_accuracy: float) -> None:
    x = np.arange(len(metrics))
    width = 0.26
    fig, axis = plt.subplots(figsize=(12, 6.5))
    axis.bar(x - width, 100 * metrics["accuracy"], width, label="Accuracy", color="#225ea8")
    axis.bar(x, 100 * metrics["macro_precision"], width, label="Macro precision", color="#41ab5d")
    axis.bar(x + width, 100 * metrics["macro_f1"], width, label="Macro F1", color="#f16913")
    axis.axhline(
        100 * macro_accuracy,
        color="#cb181d",
        linestyle="--",
        linewidth=1.5,
        label=f"Station-macro accuracy = {100 * macro_accuracy:.2f}%",
    )
    axis.set_xticks(x, labels=metrics["station"])
    axis.set_ylim(0, 100)
    axis.set_ylabel("Out-of-fold score (%)")
    axis.set_xlabel("Station")
    axis.set_title("Station-specific performance from nested 10×10 cross-validation")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(ncol=2, loc="lower right")
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
