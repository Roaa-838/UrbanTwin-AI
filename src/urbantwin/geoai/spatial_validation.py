"""Buffered, tiled spatial robustness study for the frozen GeoAI benchmark.

This module deliberately keeps the original repeated grouped benchmark intact.
It defines a secondary, prespecified 3x3 evaluation in which every valid pixel
is held out exactly once and training pixels can be excluded by a strict
geographic buffer around the complete held-out tile.  WorldCover remains a
pseudo-label product: reported scores measure agreement, not field accuracy.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.metrics import confusion_matrix

from urbantwin.common import (
    ROOT,
    configuration_hash_record,
    load_yaml,
    require_real_provenance,
    sha256_file,
    validate_configuration_hash,
)
from urbantwin.data.earth_observation import WORLDCOVER_CLASSES
from urbantwin.geoai import train as geoai_train
from urbantwin.geoai.error_analysis import analyze_geoai_details, confusion_metrics
from urbantwin.visualization.report_writer import (
    ArtifactIntegrityError,
    validate_geoai_artifacts,
)

CONFIG_SCHEMA = "urbantwin.geoai-spatial-validation-config/v1"
BUNDLE_SCHEMA = "urbantwin.geoai-spatial-validation/v1"
CLASS_SELECTION_PROTOCOL = "intersection_of_buffered_training_supported_classes"


@dataclass(frozen=True)
class SpatialFold:
    """One group-aligned tile and its condition-specific training masks."""

    fold_index: int
    tile_row: int
    tile_col: int
    seed: int
    test_mask: np.ndarray
    train_masks: dict[float, np.ndarray]
    test_group_ids: tuple[int, ...]
    train_group_ids: dict[float, tuple[int, ...]]
    tile_mask: np.ndarray
    tile_bounds_pixels: tuple[int, int, int, int]
    minimum_separation_m: dict[float, float]


def _as_float_key(value: float) -> float:
    result = float(value)
    if not np.isfinite(result) or result < 0:
        raise ValueError("buffer distances must be finite and non-negative")
    return result


def load_spatial_validation_config(path: str | Path) -> dict[str, Any]:
    """Load and strictly validate the versioned spatial-study configuration."""

    root = load_yaml(path)
    if root.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError(f"Expected configuration schema {CONFIG_SCHEMA!r}")
    cfg = root.get("spatial_validation")
    if not isinstance(cfg, dict):
        raise TypeError("Configuration must contain a spatial_validation mapping")
    required = {
        "block_pixels",
        "tile_rows",
        "tile_cols",
        "expected_resolution_m",
        "buffer_distances_m",
        "selection_buffer_m",
        "buffer_feasibility_candidates_m",
        "buffer_selection_rule",
        "fold_seeds",
        "min_train_per_class",
        "min_test_per_class",
        "max_train_per_class",
        "expected_core_class_ids",
        "classifiers",
        "ablation",
    }
    missing = sorted(required - set(cfg))
    if missing:
        raise ValueError("Spatial-validation configuration is missing: " + ", ".join(missing))

    for key in (
        "block_pixels",
        "tile_rows",
        "tile_cols",
        "min_train_per_class",
        "min_test_per_class",
        "max_train_per_class",
    ):
        value = cfg[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"spatial_validation.{key} must be a positive integer")
    if cfg["tile_rows"] != 3 or cfg["tile_cols"] != 3 or cfg["block_pixels"] != 20:
        raise ValueError("The frozen study requires 3x3 tiles aligned to 20-pixel groups")
    resolution = float(cfg["expected_resolution_m"])
    if not np.isfinite(resolution) or resolution <= 0:
        raise ValueError("expected_resolution_m must be positive and finite")

    distances = [_as_float_key(value) for value in cfg["buffer_distances_m"]]
    if distances != [0.0, 500.0]:
        raise ValueError("The frozen comparison conditions must be exactly [0, 500] metres")
    selection = _as_float_key(cfg["selection_buffer_m"])
    if selection != 500.0 or selection not in distances:
        raise ValueError("selection_buffer_m must be the prespecified 500 m condition")
    candidates = [_as_float_key(value) for value in cfg["buffer_feasibility_candidates_m"]]
    if candidates != [100.0, 200.0, 300.0, 400.0, 500.0, 600.0]:
        raise ValueError("buffer feasibility candidates must remain the frozen 100-600 m sequence")
    if (
        not isinstance(cfg["buffer_selection_rule"], str)
        or not cfg["buffer_selection_rule"].strip()
    ):
        raise ValueError("buffer_selection_rule must document the score-independent choice")

    seeds = cfg["fold_seeds"]
    expected_seeds = [42 + 100 * index for index in range(9)]
    if seeds != expected_seeds:
        raise ValueError(f"fold_seeds must equal the frozen sequence {expected_seeds}")
    classes = cfg["expected_core_class_ids"]
    if classes != [10, 50, 80]:
        raise ValueError("expected_core_class_ids must equal the prespecified [10, 50, 80]")
    if cfg["classifiers"] != ["random_forest", "hist_gradient_boosting"]:
        raise ValueError("classifiers differ from the frozen benchmark")
    if cfg["ablation"] != ["bands_only", "bands_plus_indices"]:
        raise ValueError("feature ablations differ from the frozen benchmark")
    if cfg["min_train_per_class"] != 20 or cfg["max_train_per_class"] != 1200:
        raise ValueError("training support/subsampling settings differ from the frozen benchmark")
    result = dict(cfg)
    result["expected_resolution_m"] = resolution
    result["buffer_distances_m"] = distances
    result["selection_buffer_m"] = selection
    result["buffer_feasibility_candidates_m"] = candidates
    return result


def _affine_coefficients(transform: Any) -> tuple[float, float, float, float, float, float]:
    if isinstance(transform, Sequence) and not isinstance(transform, (str, bytes)):
        values = list(transform)
        if len(values) != 6:
            raise ValueError("transform must contain six affine coefficients")
        return tuple(float(value) for value in values)  # type: ignore[return-value]
    try:
        return tuple(float(getattr(transform, key)) for key in ("a", "b", "c", "d", "e", "f"))  # type: ignore[return-value]
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("transform must expose six affine coefficients") from exc


def _pixel_centers(shape: tuple[int, int], transform: Any) -> np.ndarray:
    a, b, c, d, e, f = _affine_coefficients(transform)
    rows, cols = np.indices(shape, dtype=float)
    x = a * (cols + 0.5) + b * (rows + 0.5) + c
    y = d * (cols + 0.5) + e * (rows + 0.5) + f
    return np.column_stack((x.ravel(), y.ravel()))


def _group_grid(shape: tuple[int, int], block_pixels: int) -> np.ndarray:
    rows, cols = np.indices(shape)
    block_columns = int(np.ceil(shape[1] / block_pixels))
    return (rows // block_pixels) * block_columns + cols // block_pixels


def _tile_axis_groups(group_count: int, tile_count: int) -> list[np.ndarray]:
    if group_count < tile_count:
        raise ValueError("Raster has fewer base groups than requested tiles")
    return [
        np.asarray(values, dtype=int)
        for values in np.array_split(np.arange(group_count), tile_count)
    ]


def build_spatial_folds(
    labels: np.ndarray,
    valid_mask: np.ndarray,
    transform: Any,
    *,
    block_pixels: int,
    tile_rows: int,
    tile_columns: int,
    buffer_distances_m: Sequence[float | int],
    fold_seeds: Sequence[int],
) -> list[SpatialFold]:
    """Build deterministic group-aligned folds with exact center-distance buffers.

    Distances are measured from every candidate training pixel center to every
    pixel center in the complete test tile (including invalid source pixels).
    A training pixel is retained only when its distance is *strictly greater*
    than the configured separation.
    """

    labels = np.asarray(labels)
    valid = np.asarray(valid_mask, dtype=bool)
    if labels.ndim != 2 or valid.shape != labels.shape:
        raise ValueError("labels and valid_mask must be aligned two-dimensional arrays")
    if block_pixels < 1 or tile_rows < 1 or tile_columns < 1:
        raise ValueError("block and tile dimensions must be positive")
    distances = [_as_float_key(value) for value in buffer_distances_m]
    if len(distances) != len(set(distances)) or not distances:
        raise ValueError("buffer distances must be non-empty and unique")
    expected_folds = tile_rows * tile_columns
    if len(fold_seeds) != expected_folds or len(set(fold_seeds)) != expected_folds:
        raise ValueError("fold_seeds must contain one unique seed per tile")

    height, width = labels.shape
    group_row_count = int(np.ceil(height / block_pixels))
    group_col_count = int(np.ceil(width / block_pixels))
    row_tiles = _tile_axis_groups(group_row_count, tile_rows)
    col_tiles = _tile_axis_groups(group_col_count, tile_columns)
    group_ids = _group_grid(labels.shape, block_pixels)
    centers = _pixel_centers(labels.shape, transform)
    coverage = np.zeros(labels.shape, dtype=np.int16)
    folds: list[SpatialFold] = []

    for tile_row, row_groups in enumerate(row_tiles):
        for tile_col, col_groups in enumerate(col_tiles):
            fold_index = tile_row * tile_columns + tile_col
            row_start = int(row_groups[0] * block_pixels)
            row_stop = min(height, int((row_groups[-1] + 1) * block_pixels))
            col_start = int(col_groups[0] * block_pixels)
            col_stop = min(width, int((col_groups[-1] + 1) * block_pixels))
            tile_mask = np.zeros(labels.shape, dtype=bool)
            tile_mask[row_start:row_stop, col_start:col_stop] = True
            test_mask = valid & tile_mask
            coverage[test_mask] += 1
            tree = cKDTree(centers[tile_mask.ravel()])
            nearest = tree.query(centers, k=1, workers=-1)[0].reshape(labels.shape)
            train_masks: dict[float, np.ndarray] = {}
            train_groups: dict[float, tuple[int, ...]] = {}
            minimum_separation: dict[float, float] = {}
            test_groups = tuple(int(value) for value in np.unique(group_ids[tile_mask]))
            for distance in distances:
                train = valid & ~tile_mask & (nearest > distance)
                train_masks[distance] = train
                groups = tuple(int(value) for value in np.unique(group_ids[train]))
                if set(groups) & set(test_groups):
                    raise AssertionError("Original 20x20 group IDs overlap train and test")
                train_groups[distance] = groups
                minimum = float(nearest[train].min()) if np.any(train) else float("nan")
                if np.any(train) and not minimum > distance:
                    raise AssertionError("Training pixels violate strict geographic separation")
                minimum_separation[distance] = minimum
            folds.append(
                SpatialFold(
                    fold_index=fold_index,
                    tile_row=tile_row,
                    tile_col=tile_col,
                    seed=int(fold_seeds[fold_index]),
                    test_mask=test_mask,
                    train_masks=train_masks,
                    test_group_ids=test_groups,
                    train_group_ids=train_groups,
                    tile_mask=tile_mask,
                    tile_bounds_pixels=(row_start, row_stop, col_start, col_stop),
                    minimum_separation_m=minimum_separation,
                )
            )
    if not np.array_equal(coverage[valid], np.ones(int(valid.sum()), dtype=coverage.dtype)):
        raise AssertionError("Valid held-out pixels are not covered exactly once")
    if np.any(coverage[~valid]):
        raise AssertionError("Invalid pixels entered held-out evaluation coverage")
    return folds


def _support(labels: np.ndarray, mask: np.ndarray) -> dict[int, int]:
    values, counts = np.unique(labels[mask], return_counts=True)
    return {int(value): int(count) for value, count in zip(values, counts, strict=True)}


def select_core_classes_from_training(
    labels: np.ndarray,
    folds: Sequence[SpatialFold],
    *,
    selection_buffer_m: float,
    min_train_per_class: int,
    expected_core_class_ids: Sequence[int],
) -> tuple[list[int], pd.DataFrame]:
    """Select the common class axis using buffered training partitions only."""

    labels = np.asarray(labels)
    distance = _as_float_key(selection_buffer_m)
    if min_train_per_class < 1:
        raise ValueError("min_train_per_class must be positive")
    if not folds:
        raise ValueError("At least one spatial fold is required")
    eligible_sets: list[set[int]] = []
    rows: list[dict[str, Any]] = []
    for fold in folds:
        if distance not in fold.train_masks:
            raise ValueError(f"Fold {fold.fold_index} has no {distance:g} m training condition")
        train_mask = fold.train_masks[distance]
        if train_mask.shape != labels.shape:
            raise ValueError("Fold training mask is not aligned with labels")
        support = _support(labels, train_mask)
        eligible = {class_id for class_id, count in support.items() if count >= min_train_per_class}
        eligible_sets.append(eligible)
        for class_id in sorted(set(WORLDCOVER_CLASSES) | set(support)):
            rows.append(
                {
                    "fold_index": fold.fold_index,
                    "fold_seed": fold.seed,
                    "buffer_m": distance,
                    "class_id": class_id,
                    "training_support": support.get(class_id, 0),
                    "training_eligible": class_id in eligible,
                    "selection_source": "training_partition_only",
                }
            )
    core = sorted(set.intersection(*eligible_sets))
    expected = [int(value) for value in expected_core_class_ids]
    if core != expected:
        raise ValueError(
            "Buffered training-only class intersection differs from the expected core: "
            f"observed {core}, expected {expected}"
        )
    return core, pd.DataFrame(rows)


def fold_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    core_classes: Sequence[int],
    min_test_per_class: int,
) -> tuple[pd.DataFrame, dict[str, float | int], np.ndarray]:
    """Return fold metrics while marking low-support class estimates as N/A."""

    truth = np.asarray(y_true)
    predictions = np.asarray(y_pred)
    classes = [int(value) for value in core_classes]
    if truth.ndim != 1 or predictions.shape != truth.shape:
        raise ValueError("y_true and y_pred must be aligned one-dimensional arrays")
    if not classes or len(classes) != len(set(classes)):
        raise ValueError("core_classes must be a non-empty unique sequence")
    if min_test_per_class < 1:
        raise ValueError("min_test_per_class must be positive")
    if not np.isin(truth, classes).all() or not np.isin(predictions, classes).all():
        raise ValueError("Fold metrics require truth and predictions on the fixed core axis")
    # Recent scikit-learn releases reject empty inputs before applying the
    # explicit label axis.  Empty held-out folds are still useful diagnostics,
    # so construct their all-zero confusion matrix directly.
    matrix = (
        np.zeros((len(classes), len(classes)), dtype=int)
        if len(truth) == 0
        else confusion_matrix(truth, predictions, labels=classes)
    )
    if len(truth) == 0:
        aggregate: dict[str, float | int] = {
            "accuracy": float("nan"),
            "macro_f1": float("nan"),
            "weighted_f1": float("nan"),
            "balanced_accuracy": float("nan"),
            "support": 0,
        }
        raw = pd.DataFrame({"class_id": classes, "support": np.zeros(len(classes), dtype=int)})
        raw["precision"] = np.nan
        raw["recall"] = np.nan
        raw["f1"] = np.nan
    else:
        raw, calculated = confusion_metrics(matrix, classes, eligible_class_ids=classes)
        aggregate = {**calculated, "support": int(calculated["support"])}
    aggregate["evaluated_samples"] = len(truth)
    raw = raw.rename(columns={"support": "test_support"})
    support = raw["test_support"].astype(int)
    raw["status"] = np.where(
        support.eq(0), "absent", np.where(support.lt(min_test_per_class), "low_support", "ok")
    )
    raw.loc[raw["status"] != "ok", ["precision", "recall", "f1"]] = np.nan
    return (
        raw[["class_id", "test_support", "precision", "recall", "f1", "status"]],
        aggregate,
        matrix,
    )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=str) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False, lineterminator="\n", float_format="%.12g")


def _git_state(root: Path) -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "unavailable", "working_tree_dirty": None}
    return {"commit": commit, "working_tree_dirty": dirty}


def _validate_source_stack(
    data_dir: Path, expected_resolution_m: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any], Path, Path]:
    provenance_path = data_dir / "provenance.json"
    stack_path = data_dir / "eo_stack.npz"
    if not provenance_path.is_file() or not stack_path.is_file():
        raise FileNotFoundError(
            "Spatial validation requires the validated real EO provenance.json and eo_stack.npz"
        )
    provenance = _read_json(provenance_path)
    require_real_provenance(provenance)
    if provenance.get("eo_stack_sha256") != sha256_file(stack_path):
        raise ValueError("EO stack does not match its real-source provenance hash")
    with np.load(stack_path, allow_pickle=False) as archive:
        missing = {"bands", "labels", "valid_mask"} - set(archive.files)
        if missing:
            raise ValueError(f"EO stack is missing required arrays: {sorted(missing)}")
        bands = np.asarray(archive["bands"])
        labels = np.asarray(archive["labels"])
        declared_valid = np.asarray(archive["valid_mask"], dtype=bool)
    if bands.ndim != 3 or bands.shape[0] != 5 or labels.shape != bands.shape[1:]:
        raise ValueError("EO stack must contain aligned five-band and label rasters")
    if declared_valid.shape != labels.shape:
        raise ValueError("EO valid_mask is not aligned with the source rasters")
    source_valid = np.all(np.isfinite(bands), axis=0) & np.isin(labels, sorted(WORLDCOVER_CLASSES))
    if not np.array_equal(declared_valid, source_valid):
        raise ValueError("EO valid_mask disagrees with finite bands and supported labels")
    if provenance.get("height") != labels.shape[0] or provenance.get("width") != labels.shape[1]:
        raise ValueError("EO provenance dimensions do not match the raster stack")
    transform = provenance["transform"]
    a, b, _, d, e, _ = _affine_coefficients(transform)
    column_resolution = float(np.hypot(a, d))
    row_resolution = float(np.hypot(b, e))
    if not (
        np.isclose(column_resolution, expected_resolution_m, atol=1e-9, rtol=0)
        and np.isclose(row_resolution, expected_resolution_m, atol=1e-9, rtol=0)
    ):
        raise ValueError("EO raster resolution differs from the frozen spatial protocol")
    features, y, _, derived_valid = geoai_train.construct_samples(bands, labels, block_pixels=20)
    if not np.array_equal(derived_valid, declared_valid):
        raise ValueError("Derived GeoAI validity differs from the EO source valid_mask")
    if not np.array_equal(y, labels[declared_valid]):
        raise AssertionError("GeoAI feature rows are not aligned with label pixels")
    return bands, labels, features, provenance, provenance_path, stack_path


def _feasibility_table(
    labels: np.ndarray,
    valid: np.ndarray,
    transform: Any,
    cfg: Mapping[str, Any],
) -> tuple[pd.DataFrame, list[SpatialFold]]:
    candidates = [float(value) for value in cfg["buffer_feasibility_candidates_m"]]
    folds = build_spatial_folds(
        labels,
        valid,
        transform,
        block_pixels=int(cfg["block_pixels"]),
        tile_rows=int(cfg["tile_rows"]),
        tile_columns=int(cfg["tile_cols"]),
        buffer_distances_m=sorted(set(candidates + [0.0])),
        fold_seeds=cfg["fold_seeds"],
    )
    rows: list[dict[str, Any]] = []
    for distance in candidates:
        for fold in folds:
            support = _support(labels, fold.train_masks[distance])
            eligible = sorted(
                class_id
                for class_id, count in support.items()
                if count >= int(cfg["min_train_per_class"])
            )
            rows.append(
                {
                    "buffer_m": distance,
                    "fold_index": fold.fold_index,
                    "fold_seed": fold.seed,
                    "eligible_class_count": len(eligible),
                    "eligible_class_ids": json.dumps(eligible, separators=(",", ":")),
                    "training_pixel_count": int(fold.train_masks[distance].sum()),
                    "minimum_train_test_center_distance_m": fold.minimum_separation_m[distance],
                    "selection_input": "geometry_and_training_support_only",
                }
            )
    frame = pd.DataFrame(rows)
    minimum_by_buffer = frame.groupby("buffer_m")["eligible_class_count"].min()
    feasible = [float(value) for value, count in minimum_by_buffer.items() if count >= 3]
    if not feasible or max(feasible) != float(cfg["selection_buffer_m"]):
        raise ValueError(
            "The prespecified 500 m buffer no longer satisfies its training-only feasibility rule"
        )
    return frame, folds


def _core_training_indices(
    mask: np.ndarray,
    labels: np.ndarray,
    valid: np.ndarray,
    core_classes: Sequence[int],
) -> np.ndarray:
    grid_indices = np.full(labels.shape, -1, dtype=int)
    grid_indices[valid] = np.arange(int(valid.sum()))
    selected = mask & np.isin(labels, core_classes)
    return grid_indices[selected]


def _metric_row(
    aggregate: Mapping[str, float | int],
    *,
    condition: str,
    buffer_m: float,
    method: str,
    features: str,
    fold: SpatialFold | None,
    train_n: int | None,
) -> dict[str, Any]:
    return {
        "condition": condition,
        "buffer_m": buffer_m,
        "method": method,
        "features": features,
        "fold_index": None if fold is None else fold.fold_index,
        "fold_seed": None if fold is None else fold.seed,
        "train_n": train_n,
        "test_n": int(aggregate["support"]),
        "accuracy_vs_worldcover": aggregate["accuracy"],
        "macro_f1": aggregate["macro_f1"],
        "weighted_f1": aggregate["weighted_f1"],
        "balanced_accuracy": aggregate["balanced_accuracy"],
        "reference_label_role": "historic_worldcover_2021_pseudo_labels",
    }


def _condition_name(distance: float) -> str:
    return "unbuffered_tiled_control" if distance == 0 else "strict_500m_buffer"


def _add_per_class_context(
    frame: pd.DataFrame,
    *,
    scope: str,
    condition: str,
    buffer_m: float,
    method: str,
    features: str,
    fold: SpatialFold | None,
) -> pd.DataFrame:
    result = frame.copy()
    result.insert(0, "scope", scope)
    result.insert(1, "condition", condition)
    result.insert(2, "buffer_m", buffer_m)
    result.insert(3, "method", method)
    result.insert(4, "features", features)
    result.insert(5, "fold_index", None if fold is None else fold.fold_index)
    result.insert(6, "fold_seed", None if fold is None else fold.seed)
    return result


def _run_fold_models(
    features: np.ndarray,
    labels: np.ndarray,
    valid: np.ndarray,
    folds: Sequence[SpatialFold],
    cfg: Mapping[str, Any],
    core_classes: Sequence[int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    fold_rows: list[dict[str, Any]] = []
    class_frames: list[pd.DataFrame] = []
    majority_rows: list[dict[str, Any]] = []
    details: dict[str, Any] = {"folds": [], "experiments": {}}
    aggregate_matrices: dict[tuple[float, str, str], np.ndarray] = {}

    for fold in folds:
        test_idx = _core_training_indices(fold.test_mask, labels, valid, core_classes)
        y_test = labels[fold.test_mask & np.isin(labels, core_classes)]
        if not np.array_equal(y_test, labels[valid][test_idx]):
            raise AssertionError("Held-out feature and label rows are misaligned")
        fold_record: dict[str, Any] = {
            "fold_index": fold.fold_index,
            "tile_row": fold.tile_row,
            "tile_col": fold.tile_col,
            "fold_seed": fold.seed,
            "tile_bounds_pixels": list(fold.tile_bounds_pixels),
            "test_group_ids": list(fold.test_group_ids),
            "test_core_support": _support(labels, fold.test_mask & np.isin(labels, core_classes)),
            "conditions": {},
        }
        for distance_value in cfg["buffer_distances_m"]:
            distance = float(distance_value)
            condition = _condition_name(distance)
            raw_train = _core_training_indices(
                fold.train_masks[distance], labels, valid, core_classes
            )
            training_support = {
                class_id: int(np.count_nonzero(labels[valid][raw_train] == class_id))
                for class_id in core_classes
            }
            if any(count < int(cfg["min_train_per_class"]) for count in training_support.values()):
                raise ValueError(
                    f"Fold {fold.fold_index} {condition} lacks prespecified core training support"
                )
            train_idx = geoai_train.subsample_training(
                raw_train,
                labels[valid],
                int(cfg["max_train_per_class"]),
                fold.seed,
            )
            fold_record["conditions"][condition] = {
                "buffer_m": distance,
                "minimum_train_test_center_distance_m": fold.minimum_separation_m[distance],
                "training_group_ids": list(fold.train_group_ids[distance]),
                "raw_core_training_support": training_support,
                "subsampled_core_training_support": _support(
                    labels[valid], np.isin(np.arange(len(labels[valid])), train_idx)
                ),
                "train_n": len(train_idx),
                "test_n": len(test_idx),
            }

            majority_id = min(
                core_classes, key=lambda class_id: (-training_support[int(class_id)], int(class_id))
            )
            majority_prediction = np.full(len(y_test), majority_id, dtype=labels.dtype)
            majority_class, majority_aggregate, majority_matrix = fold_classification_metrics(
                y_test,
                majority_prediction,
                core_classes,
                int(cfg["min_test_per_class"]),
            )
            majority_rows.append(
                {
                    **_metric_row(
                        majority_aggregate,
                        condition=condition,
                        buffer_m=distance,
                        method="training_majority",
                        features="training_label_counts_only",
                        fold=fold,
                        train_n=len(raw_train),
                    ),
                    "predicted_class_id": majority_id,
                    "selection_source": "training_partition_only",
                }
            )
            class_frames.append(
                _add_per_class_context(
                    majority_class,
                    scope="fold",
                    condition=condition,
                    buffer_m=distance,
                    method="training_majority",
                    features="training_label_counts_only",
                    fold=fold,
                )
            )
            key = (distance, "training_majority", "training_label_counts_only")
            aggregate_matrices.setdefault(key, np.zeros_like(majority_matrix))
            aggregate_matrices[key] += majority_matrix

            for method in cfg["classifiers"]:
                for ablation in cfg["ablation"]:
                    model = geoai_train._model(str(method), fold.seed)
                    columns = geoai_train._feature_columns(str(ablation))
                    model.fit(features[train_idx, columns], labels[valid][train_idx])
                    predictions = model.predict(features[test_idx, columns])
                    per_class, aggregate, matrix = fold_classification_metrics(
                        y_test,
                        predictions,
                        core_classes,
                        int(cfg["min_test_per_class"]),
                    )
                    row = _metric_row(
                        aggregate,
                        condition=condition,
                        buffer_m=distance,
                        method=str(method),
                        features=str(ablation),
                        fold=fold,
                        train_n=len(train_idx),
                    )
                    fold_rows.append(row)
                    class_frames.append(
                        _add_per_class_context(
                            per_class,
                            scope="fold",
                            condition=condition,
                            buffer_m=distance,
                            method=str(method),
                            features=str(ablation),
                            fold=fold,
                        )
                    )
                    key = (distance, str(method), str(ablation))
                    aggregate_matrices.setdefault(key, np.zeros_like(matrix))
                    aggregate_matrices[key] += matrix
                    experiment_key = f"{condition}__{method}__{ablation}"
                    details["experiments"].setdefault(experiment_key, {"runs": []})["runs"].append(
                        {
                            "fold_index": fold.fold_index,
                            "fold_seed": fold.seed,
                            "train_n": len(train_idx),
                            "test_n": len(test_idx),
                            "confusion_matrix": matrix.tolist(),
                        }
                    )
        details["folds"].append(fold_record)

    expected_test = int(np.count_nonzero(valid & np.isin(labels, core_classes)))
    aggregate_rows: list[dict[str, Any]] = []
    for (distance, method, ablation), matrix in sorted(aggregate_matrices.items()):
        per_class, aggregate = confusion_metrics(
            matrix, core_classes, eligible_class_ids=core_classes
        )
        if int(aggregate["support"]) != expected_test:
            raise AssertionError(
                "Aggregate out-of-fold confusion does not cover each core pixel once"
            )
        condition = _condition_name(distance)
        aggregate_rows.append(
            _metric_row(
                aggregate,
                condition=condition,
                buffer_m=distance,
                method=method,
                features=ablation,
                fold=None,
                train_n=None,
            )
        )
        per_class = per_class.rename(columns={"support": "test_support"})
        per_class["status"] = "ok"
        class_frames.append(
            _add_per_class_context(
                per_class,
                scope="aggregate_oof",
                condition=condition,
                buffer_m=distance,
                method=method,
                features=ablation,
                fold=None,
            )
        )
        experiment_key = f"{condition}__{method}__{ablation}"
        details["experiments"].setdefault(experiment_key, {})["aggregate"] = {
            "confusion_matrix": matrix.tolist(),
            "metrics": aggregate_rows[-1],
        }
    return (
        pd.DataFrame(fold_rows),
        pd.DataFrame(aggregate_rows),
        pd.concat(class_frames, ignore_index=True),
        {**details, "majority_baselines": majority_rows},
    )


def _fold_manifest(
    labels: np.ndarray,
    valid: np.ndarray,
    folds: Sequence[SpatialFold],
    cfg: Mapping[str, Any],
    core_classes: Sequence[int],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fold in folds:
        test_core = fold.test_mask & np.isin(labels, core_classes)
        for distance_value in cfg["buffer_distances_m"]:
            distance = float(distance_value)
            train = fold.train_masks[distance]
            train_core = train & np.isin(labels, core_classes)
            rows.append(
                {
                    "fold_index": fold.fold_index,
                    "tile_row": fold.tile_row,
                    "tile_col": fold.tile_col,
                    "fold_seed": fold.seed,
                    "condition": _condition_name(distance),
                    "buffer_m": distance,
                    "test_valid_pixels": int(fold.test_mask.sum()),
                    "test_core_pixels": int(test_core.sum()),
                    "train_valid_pixels": int(train.sum()),
                    "train_core_pixels": int(train_core.sum()),
                    "test_group_count": len(fold.test_group_ids),
                    "train_group_count": len(fold.train_group_ids[distance]),
                    "minimum_train_test_center_distance_m": fold.minimum_separation_m[distance],
                    "test_full_class_support": json.dumps(
                        _support(labels, fold.test_mask), sort_keys=True, separators=(",", ":")
                    ),
                    "train_full_class_support": json.dumps(
                        _support(labels, train), sort_keys=True, separators=(",", ":")
                    ),
                    "test_core_support": json.dumps(
                        _support(labels, test_core), sort_keys=True, separators=(",", ":")
                    ),
                    "train_core_support": json.dumps(
                        _support(labels, train_core), sort_keys=True, separators=(",", ":")
                    ),
                }
            )
    frame = pd.DataFrame(rows)
    if len(frame) != len(folds) * len(cfg["buffer_distances_m"]):
        raise AssertionError("Fold manifest is incomplete")
    if not (frame["test_core_pixels"] > 0).all():
        raise ValueError("At least one held-out tile contains no fixed-core evaluation pixels")
    return frame


def _baseline_spatial_diagnostics(
    features: np.ndarray,
    labels: np.ndarray,
    valid: np.ndarray,
    transform: Any,
    baseline: Mapping[str, Any],
    root: Path,
) -> pd.DataFrame:
    """Measure nearest train/test separation for each frozen original split."""

    details = baseline["details"]
    config_value = details["config_path"]
    baseline_config_path = Path(config_value)
    if not baseline_config_path.is_absolute():
        baseline_config_path = root / baseline_config_path
    baseline_config = load_yaml(baseline_config_path)
    geoai_config = baseline_config.get("geoai")
    if not isinstance(geoai_config, dict):
        raise TypeError("Frozen GeoAI configuration lacks its geoai mapping")
    group_grid = _group_grid(labels.shape, int(details["spatial_block_pixels"]))
    groups = group_grid[valid]
    y = labels[valid]
    centers = _pixel_centers(labels.shape, transform)[valid.ravel()]
    rows: list[dict[str, Any]] = []
    seeds = [int(value) for value in details["requested_split_seeds"]]
    for seed in seeds:
        train_idx, test_idx = geoai_train.split_groups(
            features, y, groups, float(geoai_config["test_fraction"]), seed
        )
        nearest = cKDTree(centers[train_idx]).query(centers[test_idx], k=1, workers=-1)[0]
        rows.append(
            {
                "split_seed": seed,
                "train_n": len(train_idx),
                "test_n": len(test_idx),
                "train_group_count": len(np.unique(groups[train_idx])),
                "test_group_count": len(np.unique(groups[test_idx])),
                "minimum_test_to_train_center_distance_m": float(nearest.min()),
                "median_test_to_train_center_distance_m": float(np.median(nearest)),
                "distance_direction": "held_out_test_pixel_to_nearest_training_pixel",
            }
        )
    return pd.DataFrame(rows)


def _write_figures(
    directory: Path,
    folds: Sequence[SpatialFold],
    aggregate_metrics: pd.DataFrame,
    details: Mapping[str, Any],
    core_classes: Sequence[int],
    baseline_analysis: Mapping[str, pd.DataFrame],
) -> list[Path]:
    paths: list[Path] = []
    fold = folds[4]
    design = np.zeros(fold.test_mask.shape, dtype=np.int8)
    design[fold.train_masks[500.0]] = 1
    design[fold.tile_mask] = 3
    excluded = ~fold.tile_mask & ~fold.train_masks[500.0]
    design[excluded] = 2
    figure, axis = plt.subplots(figsize=(7.2, 5.4), constrained_layout=True)
    image = axis.imshow(
        design,
        cmap=plt.matplotlib.colors.ListedColormap(["#f2f2f2", "#2c7fb8", "#fdae61", "#d7191c"]),
    )
    colorbar = figure.colorbar(image, ax=axis, ticks=[0, 1, 2, 3], shrink=0.85)
    colorbar.ax.set_yticklabels(["invalid", "training", "500 m exclusion", "test tile"])
    axis.set(
        title="Prespecified buffered spatial fold (central tile)",
        xlabel="Pixel column",
        ylabel="Pixel row",
    )
    path = directory / "spatial_fold_design.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    paths.append(path)

    baseline_class = baseline_analysis["per_class"].copy()
    baseline_class["model"] = (
        baseline_class["method"].replace(
            {"random_forest": "RF", "hist_gradient_boosting": "HistGB"}
        )
        + " / "
        + baseline_class["features"].replace(
            {"bands_only": "bands", "bands_plus_indices": "bands+indices"}
        )
    )
    class_order = (
        baseline_class[["class_id", "class_name"]].drop_duplicates().sort_values("class_id")
    )
    positions = np.arange(len(class_order))
    figure, axes = plt.subplots(2, 1, figsize=(10, 7.2), sharex=True, constrained_layout=True)
    for model, group in baseline_class.groupby("model", sort=False):
        mean_f1 = group.groupby("class_id")["f1"].mean().reindex(class_order["class_id"])
        axes[0].plot(positions, mean_f1, marker="o", linewidth=1.3, label=model)
    support = (
        baseline_class.drop_duplicates(["split_seed", "class_id"])
        .groupby("class_id")["support"]
        .mean()
        .reindex(class_order["class_id"])
    )
    axes[1].bar(positions, support, color="#607d8b")
    axes[0].set(
        ylabel="Mean per-split F1",
        ylim=(0, 1),
        title="Frozen benchmark: class imbalance and agreement by WorldCover class",
    )
    axes[0].legend(frameon=False, ncol=2, fontsize=8)
    axes[1].set(
        ylabel="Mean held-out support",
        xlabel="WorldCover pseudo-label class",
        xticks=positions,
        xticklabels=["10\nTree", "20\nShrub", "30\nGrass", "50\nBuilt", "60\nBare", "80\nWater"],
    )
    axes[1].set_yscale("log")
    path = directory / "baseline_per_class_f1_support.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    paths.append(path)

    models = aggregate_metrics.loc[aggregate_metrics["method"] != "training_majority"].copy()
    models["model"] = (
        models["method"].replace({"random_forest": "RF", "hist_gradient_boosting": "HistGB"})
        + " / "
        + models["features"].replace({"bands_only": "bands", "bands_plus_indices": "bands+indices"})
    )
    labels = list(dict.fromkeys(models["model"].tolist()))
    x = np.arange(len(labels))
    width = 0.35
    figure, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for offset, (condition, group) in zip(
        (-width / 2, width / 2),
        models.groupby("condition", sort=False),
        strict=True,
    ):
        values = group.set_index("model").reindex(labels)["macro_f1"].to_numpy()
        axis.bar(x + offset, values, width, label=condition.replace("_", " "))
    axis.set(
        ylabel="Out-of-fold macro-F1 vs WorldCover pseudo-labels",
        title="Unbuffered tiled control and strict 500 m separation",
        xticks=x,
        xticklabels=labels,
        ylim=(0, 1),
    )
    axis.tick_params(axis="x", labelrotation=20)
    axis.legend(frameon=False)
    path = directory / "spatial_robustness_comparison.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    paths.append(path)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    for axis, condition in zip(
        axes, ["unbuffered_tiled_control", "strict_500m_buffer"], strict=True
    ):
        key = f"{condition}__random_forest__bands_plus_indices"
        matrix = np.asarray(details["experiments"][key]["aggregate"]["confusion_matrix"])
        totals = matrix.sum(axis=1, keepdims=True)
        normalized = np.divide(
            matrix, totals, out=np.zeros_like(matrix, dtype=float), where=totals > 0
        )
        rendered = axis.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
        for row in range(len(core_classes)):
            for column in range(len(core_classes)):
                axis.text(column, row, f"{normalized[row, column]:.2f}", ha="center", va="center")
        axis.set(
            title=condition.replace("_", " "),
            xlabel="Predicted WorldCover class",
            ylabel="Reference WorldCover class",
            xticks=range(len(core_classes)),
            yticks=range(len(core_classes)),
            xticklabels=[
                f"{class_id} {WORLDCOVER_CLASSES[class_id].replace('Permanent water bodies', 'Water')}"
                for class_id in core_classes
            ],
            yticklabels=[
                f"{class_id} {WORLDCOVER_CLASSES[class_id].replace('Permanent water bodies', 'Water')}"
                for class_id in core_classes
            ],
        )
        axis.tick_params(axis="x", labelrotation=20, labelsize=8)
        axis.tick_params(axis="y", labelsize=8)
    figure.colorbar(rendered, ax=axes, label="Row-normalized proportion", shrink=0.8)
    path = directory / "spatial_confusion_random_forest_indices.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    paths.append(path)
    return paths


def render_spatial_validation_figures(
    bundle_dir: str | Path,
    output_dir: str | Path,
    config_file: str | Path,
    data_dir: str | Path,
    baseline_root: str | Path = ROOT,
) -> list[Path]:
    """Render publication figures from a validated bundle without refitting models."""

    bundle = Path(bundle_dir)
    destination = Path(output_dir)
    config_path = Path(config_file)
    data_path = Path(data_dir)
    root = Path(baseline_root)
    if destination.exists():
        raise FileExistsError(
            f"Spatial-validation figure output already exists and will not be overwritten: "
            f"{destination}"
        )
    validated = validate_spatial_validation_bundle(bundle, config_path, data_path, root)
    cfg = load_spatial_validation_config(config_path)
    bands, labels, _, source, _, _ = _validate_source_stack(
        data_path, float(cfg["expected_resolution_m"])
    )
    valid = np.isin(labels, sorted(WORLDCOVER_CLASSES)) & np.all(
        np.isfinite(bands), axis=0
    )
    _, folds = _feasibility_table(labels, valid, source["transform"], cfg)
    baseline = validate_geoai_artifacts(root)
    if baseline is None:
        raise FileNotFoundError("The frozen validated GeoAI result bundle is required")
    baseline_analysis = analyze_geoai_details(baseline["details"])

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir()
    try:
        figure_paths = _write_figures(
            temporary,
            folds,
            validated["aggregate_metrics"],
            validated["details"],
            validated["details"]["core_class_ids"],
            baseline_analysis,
        )
        publication_manifest = {
            "schema_version": "urbantwin.geoai-spatial-validation-figures/v1",
            "source_bundle_provenance_sha256": sha256_file(bundle / "provenance.json"),
            "source_bundle_artifact_hashes": validated["artifact_hashes"],
            "source_role": "WorldCover 2021 pseudo-label agreement; not field accuracy",
            "attribution": {
                "sentinel_2": "Contains modified Copernicus Sentinel data 2021.",
                "worldcover_2021": (
                    "© ESA WorldCover project 2021 / Contains modified Copernicus Sentinel "
                    "data (2021) processed by the ESA WorldCover consortium."
                ),
            },
            "worldcover_license": "CC BY 4.0",
            "figure_hashes": {path.name: sha256_file(path) for path in figure_paths},
        }
        _write_json(temporary / "publication_manifest.json", publication_manifest)
        names = [path.name for path in figure_paths]
        temporary.rename(destination)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return [destination / name for name in names]


def run_spatial_validation(
    data_dir: str | Path,
    out_dir: str | Path,
    config_file: str | Path,
    baseline_root: str | Path = ROOT,
) -> pd.DataFrame:
    """Execute the prespecified robustness study into a new atomic bundle.

    The destination must not exist.  The frozen GeoAI baseline and real EO
    source are validated before fitting, and a failed run leaves no partial
    research bundle at the requested destination.
    """

    data_path = Path(data_dir)
    destination = Path(out_dir)
    config_path = Path(config_file)
    root = Path(baseline_root)
    if destination.exists():
        raise FileExistsError(
            f"Spatial-validation output already exists and will not be overwritten: {destination}"
        )
    root_configuration = load_yaml(config_path)
    cfg = load_spatial_validation_config(config_path)
    baseline = validate_geoai_artifacts(root)
    if baseline is None:
        raise FileNotFoundError("The frozen validated GeoAI result bundle is required")
    bands, labels, features, source, source_provenance_path, stack_path = _validate_source_stack(
        data_path, float(cfg["expected_resolution_m"])
    )
    valid = np.asarray(
        np.isin(labels, sorted(WORLDCOVER_CLASSES)) & np.all(np.isfinite(bands), axis=0)
    )
    group_rows = int(np.ceil(labels.shape[0] / int(cfg["block_pixels"])))
    group_columns = int(np.ceil(labels.shape[1] / int(cfg["block_pixels"])))
    if (group_rows, group_columns) != (9, 11):
        raise ValueError(
            "The frozen Tahrir raster must form exactly 9 by 11 original 20-pixel groups"
        )
    feasibility, all_distance_folds = _feasibility_table(labels, valid, source["transform"], cfg)
    # Fold geometry is independent of the distance list. Reuse the audited
    # masks for the two comparison conditions instead of rebuilding them.
    folds = [
        SpatialFold(
            fold_index=fold.fold_index,
            tile_row=fold.tile_row,
            tile_col=fold.tile_col,
            seed=fold.seed,
            test_mask=fold.test_mask,
            train_masks={
                float(distance): fold.train_masks[float(distance)]
                for distance in cfg["buffer_distances_m"]
            },
            test_group_ids=fold.test_group_ids,
            train_group_ids={
                float(distance): fold.train_group_ids[float(distance)]
                for distance in cfg["buffer_distances_m"]
            },
            tile_mask=fold.tile_mask,
            tile_bounds_pixels=fold.tile_bounds_pixels,
            minimum_separation_m={
                float(distance): fold.minimum_separation_m[float(distance)]
                for distance in cfg["buffer_distances_m"]
            },
        )
        for fold in all_distance_folds
    ]
    core_classes, eligibility = select_core_classes_from_training(
        labels,
        folds,
        selection_buffer_m=float(cfg["selection_buffer_m"]),
        min_train_per_class=int(cfg["min_train_per_class"]),
        expected_core_class_ids=cfg["expected_core_class_ids"],
    )
    manifest = _fold_manifest(labels, valid, folds, cfg, core_classes)
    fold_metrics, aggregate_metrics, per_class, details = _run_fold_models(
        features, labels, valid, folds, cfg, core_classes
    )
    baseline_analysis = analyze_geoai_details(baseline["details"])
    baseline_spatial = _baseline_spatial_diagnostics(
        features, labels, valid, source["transform"], baseline, root
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir()
    try:
        _write_csv(temporary / "buffer_feasibility.csv", feasibility)
        _write_csv(temporary / "class_eligibility.csv", eligibility)
        _write_csv(temporary / "fold_manifest.csv", manifest)
        _write_csv(temporary / "fold_metrics.csv", fold_metrics)
        _write_csv(temporary / "aggregate_metrics.csv", aggregate_metrics)
        _write_csv(temporary / "per_class_metrics.csv", per_class)
        _write_csv(
            temporary / "majority_baselines.csv", pd.DataFrame(details["majority_baselines"])
        )
        for name, frame in baseline_analysis.items():
            _write_csv(temporary / f"baseline_{name}.csv", frame)
        _write_csv(temporary / "baseline_spatial_diagnostics.csv", baseline_spatial)
        details.update(
            {
                "schema_version": BUNDLE_SCHEMA,
                "data_kind": "real_eo_with_historic_worldcover_pseudo_labels",
                "class_selection_protocol": CLASS_SELECTION_PROTOCOL,
                "core_class_ids": core_classes,
                "core_class_names": [WORLDCOVER_CLASSES[value] for value in core_classes],
                "test_coverage_protocol": "all_valid_fixed_core_pixels_exactly_once_per_condition",
                "buffer_distance_protocol": "strict_euclidean_pixel_center_distance_from_entire_test_tile",
            }
        )
        _write_json(temporary / "spatial_details.json", details)
        _write_json(temporary / "configuration_snapshot.json", root_configuration)
        _write_json(
            temporary / "source_manifest.json",
            {
                "data_kind": "real",
                "eo_provenance_sha256": sha256_file(source_provenance_path),
                "eo_stack_sha256": sha256_file(stack_path),
                "sentinel_stac_item_ids": source["sentinel_stac_item_ids"],
                "worldcover_stac_item_ids": source["worldcover_stac_item_ids"],
                "crs": source["crs"],
                "transform": source["transform"],
                "source_role": {
                    "sentinel_2": "real_earth_observation_features",
                    "worldcover_2021": "historic_pseudo_labels_not_independent_ground_truth",
                },
                "attribution": {
                    "sentinel_2": "Contains modified Copernicus Sentinel data 2021.",
                    "worldcover_2021": (
                        "© ESA WorldCover project 2021 / Contains modified Copernicus "
                        "Sentinel data (2021) processed by the ESA WorldCover consortium."
                    ),
                },
                "worldcover_license": "CC BY 4.0",
                "signed_asset_urls_recorded": False,
            },
        )
        figure_paths = _write_figures(
            temporary,
            folds,
            aggregate_metrics,
            details,
            core_classes,
            baseline_analysis,
        )
        artifacts = sorted(
            path
            for path in temporary.iterdir()
            if path.is_file() and path.name != "provenance.json"
        )
        provenance = {
            "schema_version": BUNDLE_SCHEMA,
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "data_kind": "real_eo_with_historic_worldcover_pseudo_labels",
            "scientific_role": "secondary_spatial_robustness_study",
            "reference_label_limitation": (
                "Agreement with ESA WorldCover 2021 pseudo-labels is not independent field accuracy."
            ),
            "configuration": root_configuration,
            **configuration_hash_record(config_path, root_configuration),
            "source": {
                "eo_provenance_sha256": sha256_file(source_provenance_path),
                "eo_stack_sha256": sha256_file(stack_path),
            },
            "frozen_baseline_hashes": baseline["hashes"],
            "git": _git_state(root),
            "artifact_hashes": {path.name: sha256_file(path) for path in artifacts},
            "figure_files": [path.name for path in figure_paths],
        }
        _write_json(temporary / "provenance.json", provenance)
        validate_spatial_validation_bundle(temporary, config_path, data_path, root)
        temporary.rename(destination)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return aggregate_metrics


def _require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ArtifactIntegrityError(f"{name} is missing columns: {', '.join(missing)}")


def _require_reconstructed_table(
    recorded: pd.DataFrame,
    rebuilt: pd.DataFrame,
    columns: Sequence[str],
    message: str,
) -> None:
    """Compare CSV-loaded records with source reconstruction without dtype false alarms."""

    try:
        pd.testing.assert_frame_equal(
            recorded[list(columns)].reset_index(drop=True),
            rebuilt[list(columns)].reset_index(drop=True),
            check_dtype=False,
            check_exact=False,
            atol=1e-12,
            rtol=0,
        )
    except (KeyError, AssertionError) as exc:
        raise ArtifactIntegrityError(message) from exc


def _validate_artifact_hashes(
    directory: Path,
    artifact_hashes: Any,
    required: set[str],
) -> dict[str, str]:
    """Validate safe bundle-local paths and exact recorded file bytes."""

    if not isinstance(artifact_hashes, dict) or not artifact_hashes:
        raise ArtifactIntegrityError("Spatial-validation provenance lacks artifact hashes")
    if missing := sorted(required - set(artifact_hashes)):
        raise ArtifactIntegrityError(
            "Spatial-validation artifact hashes are missing: " + ", ".join(missing)
        )
    validated: dict[str, str] = {}
    for relative, expected_hash in artifact_hashes.items():
        if not isinstance(relative, str) or Path(relative).name != relative:
            raise ArtifactIntegrityError("Artifact hash paths must be safe bundle-local file names")
        path = directory / relative
        if not path.is_file():
            raise ArtifactIntegrityError(f"Spatial-validation artifact is missing: {relative}")
        if not isinstance(expected_hash, str) or sha256_file(path) != expected_hash:
            raise ArtifactIntegrityError(f"Spatial-validation artifact hash mismatch: {relative}")
        validated[relative] = expected_hash
    return validated


def validate_spatial_validation_bundle(
    out_dir: str | Path,
    config_file: str | Path,
    data_dir: str | Path,
    baseline_root: str | Path = ROOT,
) -> dict[str, Any]:
    """Strictly validate a completed spatial-study bundle and its live sources."""

    directory = Path(out_dir)
    config_path = Path(config_file)
    data_path = Path(data_dir)
    root = Path(baseline_root)
    provenance_path = directory / "provenance.json"
    if not directory.is_dir() or not provenance_path.is_file():
        raise ArtifactIntegrityError(f"Spatial-validation bundle is missing: {directory}")
    try:
        provenance = _read_json(provenance_path)
    except (TypeError, ValueError) as exc:
        raise ArtifactIntegrityError(str(exc)) from exc
    if provenance.get("schema_version") != BUNDLE_SCHEMA:
        raise ArtifactIntegrityError("Spatial-validation provenance schema is unsupported")
    if provenance.get("data_kind") != "real_eo_with_historic_worldcover_pseudo_labels":
        raise ArtifactIntegrityError("Spatial-validation bundle has an invalid source label")
    configuration = provenance.get("configuration")
    if not isinstance(configuration, dict):
        raise ArtifactIntegrityError("Spatial-validation provenance lacks a configuration snapshot")
    try:
        validate_configuration_hash(
            config_path,
            provenance,
            mode="canonical_v1",
            embedded_configuration=configuration,
        )
        cfg = load_spatial_validation_config(config_path)
    except (TypeError, ValueError) as exc:
        raise ArtifactIntegrityError(
            f"Invalid spatial-study configuration provenance: {exc}"
        ) from exc

    required = {
        "aggregate_metrics.csv",
        "baseline_majority_baselines.csv",
        "baseline_paired_ablations.csv",
        "baseline_per_class.csv",
        "baseline_spatial_diagnostics.csv",
        "baseline_split_metrics.csv",
        "buffer_feasibility.csv",
        "class_eligibility.csv",
        "configuration_snapshot.json",
        "fold_manifest.csv",
        "fold_metrics.csv",
        "majority_baselines.csv",
        "per_class_metrics.csv",
        "source_manifest.json",
        "spatial_details.json",
        "spatial_fold_design.png",
        "spatial_robustness_comparison.png",
        "spatial_confusion_random_forest_indices.png",
        "baseline_per_class_f1_support.png",
    }
    artifact_hashes = _validate_artifact_hashes(
        directory, provenance.get("artifact_hashes"), required
    )
    try:
        snapshot = _read_json(directory / "configuration_snapshot.json")
    except (TypeError, ValueError) as exc:
        raise ArtifactIntegrityError(str(exc)) from exc
    if snapshot != configuration:
        raise ArtifactIntegrityError("Configuration snapshot differs from provenance")

    source_provenance_path = data_path / "provenance.json"
    stack_path = data_path / "eo_stack.npz"
    source_record = provenance.get("source")
    if not isinstance(source_record, dict):
        raise ArtifactIntegrityError("Spatial-validation source record is missing")
    if (
        not source_provenance_path.is_file()
        or source_record.get("eo_provenance_sha256") != sha256_file(source_provenance_path)
        or not stack_path.is_file()
        or source_record.get("eo_stack_sha256") != sha256_file(stack_path)
    ):
        raise ArtifactIntegrityError("Spatial-validation EO source hashes do not match")
    try:
        bands, labels, source_features, source, _, _ = _validate_source_stack(
            data_path, float(cfg["expected_resolution_m"])
        )
    except (FileNotFoundError, TypeError, ValueError) as exc:
        raise ArtifactIntegrityError(f"Invalid spatial-validation EO source: {exc}") from exc
    source_manifest = _read_json(directory / "source_manifest.json")
    if (
        source_manifest.get("eo_stack_sha256") != sha256_file(stack_path)
        or source_manifest.get("eo_provenance_sha256") != sha256_file(source_provenance_path)
        or source_manifest.get("signed_asset_urls_recorded") is not False
        or source_manifest.get("crs") != source.get("crs")
        or source_manifest.get("worldcover_license") != "CC BY 4.0"
        or not isinstance(source_manifest.get("attribution"), dict)
        or "Copernicus Sentinel data 2021"
        not in source_manifest["attribution"].get("sentinel_2", "")
        or "ESA WorldCover project 2021"
        not in source_manifest["attribution"].get("worldcover_2021", "")
    ):
        raise ArtifactIntegrityError("Spatial-validation source manifest is inconsistent")

    baseline = validate_geoai_artifacts(root)
    if baseline is None or provenance.get("frozen_baseline_hashes") != baseline["hashes"]:
        raise ArtifactIntegrityError("Frozen GeoAI baseline is missing or differs from provenance")
    try:
        analyze_geoai_details(baseline["details"])
    except (TypeError, ValueError) as exc:
        raise ArtifactIntegrityError(f"Frozen GeoAI details are invalid: {exc}") from exc
    rebuilt_baseline_spatial = _baseline_spatial_diagnostics(
        source_features,
        labels,
        np.isin(labels, sorted(WORLDCOVER_CLASSES)) & np.all(np.isfinite(bands), axis=0),
        source["transform"],
        baseline,
        root,
    )

    try:
        details = _read_json(directory / "spatial_details.json")
        aggregate = pd.read_csv(directory / "aggregate_metrics.csv")
        fold_metrics = pd.read_csv(directory / "fold_metrics.csv")
        fold_manifest = pd.read_csv(directory / "fold_manifest.csv")
        feasibility = pd.read_csv(directory / "buffer_feasibility.csv")
        eligibility = pd.read_csv(directory / "class_eligibility.csv")
        majority = pd.read_csv(directory / "majority_baselines.csv")
        baseline_spatial = pd.read_csv(directory / "baseline_spatial_diagnostics.csv")
    except (OSError, TypeError, ValueError, pd.errors.ParserError) as exc:
        raise ArtifactIntegrityError(f"Spatial-validation table cannot be read: {exc}") from exc
    try:
        pd.testing.assert_frame_equal(
            baseline_spatial,
            rebuilt_baseline_spatial,
            check_dtype=False,
            check_exact=False,
            atol=1e-9,
            rtol=0,
        )
    except AssertionError as exc:
        raise ArtifactIntegrityError(
            "Frozen baseline spatial diagnostics differ from reconstruction"
        ) from exc
    if details.get("schema_version") != BUNDLE_SCHEMA:
        raise ArtifactIntegrityError("Spatial details have an unsupported schema")
    if details.get("class_selection_protocol") != CLASS_SELECTION_PROTOCOL:
        raise ArtifactIntegrityError("Spatial details have an invalid class-selection protocol")
    core_classes = [int(value) for value in details.get("core_class_ids", [])]
    if core_classes != cfg["expected_core_class_ids"]:
        raise ArtifactIntegrityError(
            "Spatial details do not retain the expected fixed core classes"
        )

    _require_columns(
        aggregate,
        {"condition", "buffer_m", "method", "features", "test_n", "macro_f1", "weighted_f1"},
        "aggregate_metrics.csv",
    )
    _require_columns(
        fold_metrics,
        {"condition", "buffer_m", "method", "features", "fold_index", "fold_seed", "test_n"},
        "fold_metrics.csv",
    )
    _require_columns(
        fold_manifest,
        {
            "fold_index",
            "condition",
            "buffer_m",
            "test_core_pixels",
            "train_core_pixels",
            "minimum_train_test_center_distance_m",
            "test_full_class_support",
            "train_full_class_support",
        },
        "fold_manifest.csv",
    )
    if len(fold_manifest) != 18 or len(fold_metrics) != 72 or len(majority) != 18:
        raise ArtifactIntegrityError("Spatial-validation fold tables have unexpected row counts")
    if len(aggregate) != 10 or aggregate.duplicated(["condition", "method", "features"]).any():
        raise ArtifactIntegrityError("Spatial-validation aggregate experiment axis is incomplete")
    bounded_columns = ["accuracy_vs_worldcover", "macro_f1", "weighted_f1", "balanced_accuracy"]
    for name, frame in (
        ("aggregate_metrics.csv", aggregate),
        ("fold_metrics.csv", fold_metrics),
        ("majority_baselines.csv", majority),
    ):
        _require_columns(frame, set(bounded_columns), name)
        for column in bounded_columns:
            values = pd.to_numeric(frame[column], errors="coerce")
            if values.isna().any() or not values.between(0, 1).all():
                raise ArtifactIntegrityError(f"{name} contains invalid {column} values")
    if set(fold_manifest["condition"]) != {"unbuffered_tiled_control", "strict_500m_buffer"}:
        raise ArtifactIntegrityError(
            "Spatial-validation comparison conditions differ from protocol"
        )
    buffered_manifest = fold_manifest.loc[fold_manifest["buffer_m"] == 500]
    if (
        len(buffered_manifest) != 9
        or not (buffered_manifest["minimum_train_test_center_distance_m"] > 500).all()
    ):
        raise ArtifactIntegrityError("Buffered fold manifest violates strict >500 m separation")

    valid = np.isin(labels, sorted(WORLDCOVER_CLASSES)) & np.all(np.isfinite(bands), axis=0)
    rebuilt_feasibility, all_distance_folds = _feasibility_table(
        labels, valid, source["transform"], cfg
    )
    if len(feasibility) != len(rebuilt_feasibility):
        raise ArtifactIntegrityError("Buffer feasibility table is incomplete")
    check_columns = [
        "buffer_m",
        "fold_index",
        "fold_seed",
        "eligible_class_count",
        "training_pixel_count",
    ]
    _require_reconstructed_table(
        feasibility,
        rebuilt_feasibility,
        check_columns,
        "Buffer feasibility table differs from source reconstruction",
    )
    folds = [
        SpatialFold(
            fold_index=fold.fold_index,
            tile_row=fold.tile_row,
            tile_col=fold.tile_col,
            seed=fold.seed,
            test_mask=fold.test_mask,
            train_masks={
                float(value): fold.train_masks[float(value)] for value in cfg["buffer_distances_m"]
            },
            test_group_ids=fold.test_group_ids,
            train_group_ids={
                float(value): fold.train_group_ids[float(value)]
                for value in cfg["buffer_distances_m"]
            },
            tile_mask=fold.tile_mask,
            tile_bounds_pixels=fold.tile_bounds_pixels,
            minimum_separation_m={
                float(value): fold.minimum_separation_m[float(value)]
                for value in cfg["buffer_distances_m"]
            },
        )
        for fold in all_distance_folds
    ]
    try:
        rebuilt_core, rebuilt_eligibility = select_core_classes_from_training(
            labels,
            folds,
            selection_buffer_m=float(cfg["selection_buffer_m"]),
            min_train_per_class=int(cfg["min_train_per_class"]),
            expected_core_class_ids=cfg["expected_core_class_ids"],
        )
    except ValueError as exc:
        raise ArtifactIntegrityError(
            f"Training-only core selection no longer validates: {exc}"
        ) from exc
    eligibility_columns = [
        "fold_index",
        "fold_seed",
        "buffer_m",
        "class_id",
        "training_support",
        "training_eligible",
        "selection_source",
    ]
    try:
        pd.testing.assert_frame_equal(
            eligibility[eligibility_columns].reset_index(drop=True),
            rebuilt_eligibility[eligibility_columns].reset_index(drop=True),
            check_dtype=False,
            check_exact=False,
            atol=1e-12,
            rtol=0,
        )
    except (KeyError, AssertionError) as exc:
        raise ArtifactIntegrityError(
            "Class eligibility table differs from source reconstruction"
        ) from exc
    if rebuilt_core != core_classes:
        raise ArtifactIntegrityError("Class eligibility table differs from source reconstruction")

    expected_test = int(np.count_nonzero(valid & np.isin(labels, core_classes)))
    if set(aggregate["test_n"].astype(int)) != {expected_test}:
        raise ArtifactIntegrityError("Aggregate metrics do not cover every fixed-core pixel once")
    experiments = details.get("experiments")
    if not isinstance(experiments, dict) or len(experiments) != 10:
        raise ArtifactIntegrityError("Spatial details have an incomplete experiment axis")
    for row in aggregate.itertuples(index=False):
        key = f"{row.condition}__{row.method}__{row.features}"
        experiment = experiments.get(key)
        if not isinstance(experiment, dict) or not isinstance(experiment.get("aggregate"), dict):
            raise ArtifactIntegrityError(f"Missing aggregate spatial details for {key}")
        matrix = np.asarray(experiment["aggregate"].get("confusion_matrix"))
        if (
            matrix.shape != (len(core_classes), len(core_classes))
            or int(matrix.sum()) != expected_test
        ):
            raise ArtifactIntegrityError(f"Aggregate confusion is invalid for {key}")
        _, calculated = confusion_metrics(matrix, core_classes, eligible_class_ids=core_classes)
        for column in ("macro_f1", "weighted_f1", "balanced_accuracy"):
            if not np.isclose(float(getattr(row, column)), calculated[column], atol=1e-11, rtol=0):
                raise ArtifactIntegrityError(
                    f"Aggregate {column} disagrees with confusion for {key}"
                )
        if row.method != "training_majority":
            runs = experiment.get("runs")
            if not isinstance(runs, list) or len(runs) != 9:
                raise ArtifactIntegrityError(f"Spatial model {key} does not contain all nine folds")
    return {
        "provenance": provenance,
        "details": details,
        "aggregate_metrics": aggregate,
        "fold_metrics": fold_metrics,
        "fold_manifest": fold_manifest,
        "artifact_hashes": dict(artifact_hashes),
    }


__all__ = [
    "SpatialFold",
    "build_spatial_folds",
    "fold_classification_metrics",
    "load_spatial_validation_config",
    "render_spatial_validation_figures",
    "run_spatial_validation",
    "select_core_classes_from_training",
    "validate_spatial_validation_bundle",
]
