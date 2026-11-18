"""Deterministic diagnostics for completed GeoAI experiments.

The functions in this module consume stored confusion counts and training
support.  They never reconstruct row-level predictions and never use held-out
class support to choose an eligible class or majority baseline.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd


def _class_axis(
    class_ids: Sequence[int], class_names: Sequence[str] | None
) -> tuple[list[int], list[str]]:
    ids = [int(value) for value in class_ids]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("class_ids must be a non-empty sequence of unique values")
    if class_names is None:
        names = [str(value) for value in ids]
    else:
        names = [str(value) for value in class_names]
        if len(names) != len(ids):
            raise ValueError("class_names must align with class_ids")
    return ids, names


def _confusion_array(matrix: Any, size: int) -> np.ndarray:
    values = np.asarray(matrix)
    if values.shape != (size, size):
        raise ValueError(f"confusion matrix must have shape {(size, size)}")
    if not np.issubdtype(values.dtype, np.number):
        raise TypeError("confusion matrix must be numeric")
    values = values.astype(float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("confusion matrix must contain finite non-negative counts")
    if not np.allclose(values, np.rint(values), atol=0.0, rtol=0.0):
        raise ValueError("confusion matrix counts must be integers")
    return values


def _eligible_indices(class_ids: Sequence[int], eligible_class_ids: Sequence[int] | None) -> list[int]:
    ids = [int(value) for value in class_ids]
    eligible = ids if eligible_class_ids is None else [int(value) for value in eligible_class_ids]
    if not eligible or len(eligible) != len(set(eligible)):
        raise ValueError("eligible_class_ids must be a non-empty unique sequence")
    missing = sorted(set(eligible) - set(ids))
    if missing:
        raise ValueError(f"eligible classes are absent from the confusion axis: {missing}")
    return [ids.index(value) for value in eligible]


def normalize_confusion(matrix: Any) -> np.ndarray:
    """Return row-normalized confusion counts; unsupported rows remain zero."""
    values = np.asarray(matrix)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("confusion matrix must be square")
    counts = _confusion_array(values, values.shape[0])
    totals = counts.sum(axis=1, keepdims=True)
    return np.divide(counts, totals, out=np.zeros_like(counts), where=totals > 0)


def confusion_metrics(
    matrix: Any,
    class_ids: Sequence[int],
    *,
    class_names: Sequence[str] | None = None,
    eligible_class_ids: Sequence[int] | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Calculate per-class and aggregate metrics from stored confusion counts.

    Aggregate macro and balanced metrics cover training-eligible classes only.
    A class with zero held-out support receives zero recall/F1 and is retained in
    the macro calculation, matching the frozen benchmark's explicit label list.
    """
    ids, names = _class_axis(class_ids, class_names)
    counts = _confusion_array(matrix, len(ids))
    indices = _eligible_indices(ids, eligible_class_ids)
    selected_ids = [ids[index] for index in indices]
    selected_names = [names[index] for index in indices]
    selected = counts[np.ix_(indices, indices)]
    support = selected.sum(axis=1)
    predicted = selected.sum(axis=0)
    correct = np.diag(selected)
    precision = np.divide(correct, predicted, out=np.zeros_like(correct), where=predicted > 0)
    recall = np.divide(correct, support, out=np.zeros_like(correct), where=support > 0)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(correct),
        where=(precision + recall) > 0,
    )
    total = float(support.sum())
    if total <= 0:
        raise ValueError("eligible confusion rows contain no held-out observations")
    per_class = pd.DataFrame(
        {
            "class_id": selected_ids,
            "class_name": selected_names,
            "support": support.astype(int),
            "predicted": predicted.astype(int),
            "correct": correct.astype(int),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    )
    aggregate = {
        "accuracy": float(correct.sum() / total),
        "macro_f1": float(f1.mean()),
        "weighted_f1": float(np.average(f1, weights=support)),
        "balanced_accuracy": float(recall.mean()),
        "support": total,
    }
    return per_class, aggregate


def training_majority_baseline(
    training_support: Mapping[str | int, int],
    evaluation_support: Mapping[str | int, int],
    eligible_class_ids: Sequence[int],
    *,
    class_names: Mapping[int, str] | None = None,
) -> dict[str, float | int | str]:
    """Evaluate a majority predictor selected exclusively from training counts."""
    eligible = [int(value) for value in eligible_class_ids]
    if not eligible or len(eligible) != len(set(eligible)):
        raise ValueError("eligible_class_ids must be a non-empty unique sequence")

    def support(mapping: Mapping[str | int, int], class_id: int) -> int:
        value = mapping.get(str(class_id), mapping.get(class_id, 0))
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
            raise ValueError("class support must contain non-negative integer counts")
        return int(value)

    train_counts = {class_id: support(training_support, class_id) for class_id in eligible}
    if not any(train_counts.values()):
        raise ValueError("eligible classes have no training support")
    # The class ID is the deterministic tie break; no held-out information enters selection.
    majority_id = min(eligible, key=lambda value: (-train_counts[value], value))
    test_counts = {class_id: support(evaluation_support, class_id) for class_id in eligible}
    total = sum(test_counts.values())
    if total <= 0:
        raise ValueError("eligible classes have no evaluation support")
    matrix = np.zeros((len(eligible), len(eligible)), dtype=int)
    majority_index = eligible.index(majority_id)
    for row, class_id in enumerate(eligible):
        matrix[row, majority_index] = test_counts[class_id]
    _, aggregate = confusion_metrics(matrix, eligible, eligible_class_ids=eligible)
    return {
        "predicted_class_id": majority_id,
        "predicted_class_name": (
            str(class_names[majority_id]) if class_names and majority_id in class_names else str(majority_id)
        ),
        "training_support": train_counts[majority_id],
        "evaluation_support": total,
        **aggregate,
    }


def paired_feature_ablations(split_metrics: pd.DataFrame) -> pd.DataFrame:
    """Pair bands-plus-indices with bands-only results by method and split seed."""
    required = {
        "method",
        "features",
        "split_seed",
        "macro_f1",
        "weighted_f1",
        "accuracy",
        "balanced_accuracy",
    }
    missing = sorted(required - set(split_metrics.columns))
    if missing:
        raise ValueError("split metrics are missing columns: " + ", ".join(missing))
    key = ["method", "split_seed"]
    if split_metrics.duplicated(key + ["features"]).any():
        raise ValueError("split metrics contain duplicate method/seed/feature rows")
    metrics = ["macro_f1", "weighted_f1", "accuracy", "balanced_accuracy"]
    bands = split_metrics.loc[split_metrics["features"] == "bands_only", key + metrics]
    indices = split_metrics.loc[
        split_metrics["features"] == "bands_plus_indices", key + metrics
    ]
    paired = bands.merge(
        indices,
        on=key,
        how="outer",
        validate="one_to_one",
        suffixes=("__bands_only", "__bands_plus_indices"),
        indicator=True,
    )
    if paired.empty or not paired["_merge"].eq("both").all():
        raise ValueError("feature ablation rows do not form complete method/seed pairs")
    result = paired.drop(columns="_merge")
    for metric in metrics:
        result[f"delta_{metric}__indices_minus_bands"] = (
            result[f"{metric}__bands_plus_indices"] - result[f"{metric}__bands_only"]
        )
    return result.sort_values(key, kind="stable").reset_index(drop=True)


def analyze_geoai_details(details: Mapping[str, Any]) -> dict[str, pd.DataFrame]:
    """Create split, per-class, majority-baseline and ablation tables.

    The input is the stored ``geoai_details.json`` object.  Repeated splits stay
    as separate rows; this function deliberately does not treat them as
    independent samples or synthesize row-level predictions.
    """
    class_ids, class_names = _class_axis(details.get("class_ids", []), details.get("class_names"))
    name_by_id = dict(zip(class_ids, class_names, strict=True))
    diagnostics = details.get("split_diagnostics")
    experiments = details.get("experiments")
    if not isinstance(diagnostics, list) or not isinstance(experiments, Mapping):
        raise TypeError("GeoAI details must contain split_diagnostics and experiments")
    diagnostic_by_seed: dict[int, Mapping[str, Any]] = {}
    for diagnostic in diagnostics:
        if not isinstance(diagnostic, Mapping) or "seed" not in diagnostic:
            raise TypeError("GeoAI split diagnostic is malformed")
        seed = int(diagnostic["seed"])
        if seed in diagnostic_by_seed:
            raise ValueError(f"Duplicate GeoAI split diagnostic seed: {seed}")
        diagnostic_by_seed[seed] = diagnostic

    class_rows: list[pd.DataFrame] = []
    split_rows: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []
    for experiment in experiments.values():
        if not isinstance(experiment, Mapping) or not isinstance(experiment.get("row"), Mapping):
            raise TypeError("GeoAI experiment is malformed")
        method = str(experiment["row"]["method"])
        features = str(experiment["row"]["features"])
        runs = experiment.get("runs")
        if not isinstance(runs, list):
            raise TypeError("GeoAI experiment runs must be a list")
        for run in runs:
            if not isinstance(run, Mapping):
                raise TypeError("GeoAI experiment run is malformed")
            seed = int(run["seed"])
            diagnostic = diagnostic_by_seed.get(seed)
            if diagnostic is None:
                raise ValueError(f"Missing split diagnostic for seed {seed}")
            eligible = [int(value) for value in run["eligible_class_ids"]]
            if eligible != [int(value) for value in diagnostic.get("eligible_class_ids", [])]:
                raise ValueError(f"Eligible classes disagree for split seed {seed}")
            per_class, aggregate = confusion_metrics(
                run["confusion_matrix"],
                class_ids,
                class_names=class_names,
                eligible_class_ids=eligible,
            )
            per_class.insert(0, "split_seed", seed)
            per_class.insert(0, "features", features)
            per_class.insert(0, "method", method)
            class_rows.append(per_class)
            run_row = run.get("row")
            if not isinstance(run_row, Mapping):
                raise TypeError("GeoAI run row is malformed")
            for key, stored_key in (
                ("macro_f1", "macro_f1"),
                ("weighted_f1", "weighted_f1"),
                ("accuracy", "accuracy_vs_worldcover"),
            ):
                if not np.isclose(aggregate[key], float(run_row[stored_key]), atol=1e-12, rtol=0):
                    raise ValueError(f"Stored {stored_key} disagrees with confusion counts")
            split_rows.append(
                {
                    "method": method,
                    "features": features,
                    "split_seed": seed,
                    **aggregate,
                }
            )
            evaluation_support = {
                int(row.class_id): int(row.support) for row in per_class.itertuples(index=False)
            }
            baseline_rows.append(
                {
                    "method": method,
                    "features": features,
                    "split_seed": seed,
                    **training_majority_baseline(
                        diagnostic.get("raw_train_class_support", {}),
                        evaluation_support,
                        eligible,
                        class_names=name_by_id,
                    ),
                }
            )
    split_frame = pd.DataFrame(split_rows)
    return {
        "per_class": pd.concat(class_rows, ignore_index=True) if class_rows else pd.DataFrame(),
        "split_metrics": split_frame,
        "majority_baselines": pd.DataFrame(baseline_rows),
        "paired_ablations": paired_feature_ablations(split_frame),
    }


__all__ = [
    "analyze_geoai_details",
    "confusion_metrics",
    "normalize_confusion",
    "paired_feature_ablations",
    "training_majority_baseline",
]
