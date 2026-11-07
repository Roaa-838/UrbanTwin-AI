"""Spatially grouped classification of ESA WorldCover pseudo-labels.

The scores produced here measure agreement with a raster label product. They are
not an independent land-cover accuracy assessment. In particular, class
eligibility is decided from each training partition only; held-out labels are
used for evaluation and support diagnostics, never to cherry-pick classes.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit

from urbantwin.common import ROOT, load_yaml, require_real_provenance, save_json, sha256_file
from urbantwin.data.earth_observation import WORLDCOVER_CLASSES, compute_indices


def construct_samples(bands: np.ndarray, labels: np.ndarray, block_pixels: int = 20):
    """Return finite known-class pixels and deterministic rectangular group IDs."""
    if bands.ndim != 3 or bands.shape[0] != 5 or labels.ndim != 2:
        raise ValueError("Expected bands (5,H,W) and labels (H,W)")
    if bands.shape[1:] != labels.shape:
        raise ValueError("Band and label grids must have identical height and width")
    if block_pixels < 1:
        raise ValueError("block_pixels must be a positive integer")
    ndvi, ndbi = compute_indices(bands)
    all_features = np.concatenate([bands, ndvi[None], ndbi[None]], axis=0)
    valid = np.isin(labels, sorted(WORLDCOVER_CLASSES)) & np.all(
        np.isfinite(all_features), axis=0
    )
    rows, cols = np.indices(labels.shape)
    nblock_col = int(np.ceil(labels.shape[1] / block_pixels))
    groups = (rows // block_pixels) * nblock_col + (cols // block_pixels)
    return all_features[:, valid].T, labels[valid], groups[valid], valid


def split_groups(X, y, groups, test_fraction: float = 0.25, seed: int = 42):
    """Create a reproducible holdout with no rectangular group leakage."""
    if not (0 < float(test_fraction) < 1):
        raise ValueError("test_fraction must be strictly between 0 and 1")
    if not (len(X) == len(y) == len(groups)):
        raise ValueError("X, y and groups must contain the same number of samples")
    if len(np.unique(groups)) < 2:
        raise ValueError("At least two spatial groups are required for a holdout")
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))
    if set(groups[train_idx]) & set(groups[test_idx]):
        raise AssertionError("Spatial group leakage")
    return train_idx, test_idx


def select_supported_classes(
    y: np.ndarray, train_idx: np.ndarray, min_train_per_class: int
) -> list[int]:
    """Select eligible labels using *only* the training partition."""
    if min_train_per_class < 1:
        raise ValueError("min_train_per_class must be positive")
    train_labels = np.asarray(y)[np.asarray(train_idx, dtype=int)]
    return [
        int(label)
        for label in sorted(np.unique(train_labels))
        if int(np.count_nonzero(train_labels == label)) >= min_train_per_class
    ]


def subsample_training(train_idx, y, max_per_class: int, seed: int):
    if max_per_class < 1:
        raise ValueError("max_per_class must be positive")
    rng = np.random.default_rng(seed)
    chosen: list[int] = []
    for label in np.unique(y[train_idx]):
        ids = train_idx[y[train_idx] == label]
        chosen.extend(rng.choice(ids, size=min(max_per_class, len(ids)), replace=False).tolist())
    return np.array(sorted(chosen), dtype=int)


def _configured_split_seeds(config: dict[str, Any], root_config: dict[str, Any]) -> list[int]:
    raw = config.get("spatial_split_seeds", [root_config.get("random_seed", 42)])
    if isinstance(raw, (int, np.integer)):
        raw = [raw]
    if not isinstance(raw, list) or not raw:
        raise ValueError("geoai.spatial_split_seeds must be a non-empty list of integers")
    seeds = [int(value) for value in raw]
    if len(seeds) != len(set(seeds)):
        raise ValueError("geoai.spatial_split_seeds must not contain duplicates")
    return seeds


def _model(name: str, seed: int):
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=120, min_samples_leaf=2, n_jobs=-1, random_state=seed
        )
    if name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            max_iter=100, max_leaf_nodes=15, learning_rate=0.08, random_state=seed
        )
    raise ValueError(f"Unknown model {name!r}")


def _feature_columns(name: str) -> slice:
    if name == "bands_only":
        return slice(0, 5)
    if name == "bands_plus_indices":
        return slice(0, 7)
    raise ValueError(f"Unknown feature ablation {name!r}")


def _class_support(labels: np.ndarray) -> dict[str, int]:
    return {str(int(label)): int(np.count_nonzero(labels == label)) for label in np.unique(labels)}


def run_classification(
    data_dir: str | Path,
    out_dir: str | Path,
    config_file: str | Path,
    allow_synthetic: bool = False,
) -> pd.DataFrame:
    """Run one or more grouped holdouts and write auditable aggregate results."""
    directory = Path(data_dir)
    provenance_path = directory / "provenance.json"
    stack_path = directory / "eo_stack.npz"
    if not provenance_path.is_file():
        raise FileNotFoundError(f"Missing EO provenance: {provenance_path}")
    if not stack_path.is_file():
        raise FileNotFoundError(f"Missing aligned EO stack: {stack_path}")
    try:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"Invalid EO provenance JSON: {provenance_path}") from exc
    if not isinstance(provenance, dict) or not provenance.get("data_kind"):
        raise ValueError("EO provenance must be an object with a data_kind field")
    if not allow_synthetic:
        require_real_provenance(provenance)
    stack_sha256 = sha256_file(stack_path)
    declared_stack_sha256 = provenance.get("eo_stack_sha256")
    if declared_stack_sha256 and declared_stack_sha256 != stack_sha256:
        raise ValueError(
            "EO stack does not match provenance: eo_stack_sha256 differs from the file on disk"
        )

    config_path = Path(config_file)
    root_config = load_yaml(config_path)
    if "geoai" not in root_config or not isinstance(root_config["geoai"], dict):
        raise ValueError("Configuration must contain a geoai mapping")
    cfg = root_config["geoai"]
    split_seeds = _configured_split_seeds(cfg, root_config)

    with np.load(stack_path, allow_pickle=False) as arrays:
        missing_arrays = {"bands", "labels"} - set(arrays.files)
        if missing_arrays:
            raise ValueError(f"EO stack is missing arrays: {sorted(missing_arrays)}")
        bands = np.asarray(arrays["bands"])
        labels = np.asarray(arrays["labels"])
        declared_valid_mask = (
            np.asarray(arrays["valid_mask"], dtype=bool)
            if "valid_mask" in arrays.files
            else None
        )
    if provenance.get("band_order") not in (None, ["B02", "B03", "B04", "B08", "B11"]):
        raise ValueError("EO provenance band_order does not match the required feature order")
    if provenance.get("height") not in (None, labels.shape[0]) or provenance.get("width") not in (
        None,
        labels.shape[1],
    ):
        raise ValueError("EO stack shape does not match the provenance grid dimensions")
    X, y, groups, derived_valid_mask = construct_samples(
        bands, labels, int(cfg["spatial_block_pixels"])
    )
    if declared_valid_mask is not None:
        if declared_valid_mask.shape != labels.shape:
            raise ValueError("EO valid_mask is not aligned with bands and labels")
        source_valid_mask = np.all(np.isfinite(bands), axis=0) & np.isin(
            labels, sorted(WORLDCOVER_CLASSES)
        )
        if not np.array_equal(declared_valid_mask, source_valid_mask):
            raise ValueError(
                "EO valid_mask disagrees with finite-band and supported-label source validity"
            )
        if np.any(derived_valid_mask & ~declared_valid_mask):
            raise AssertionError("Feature-valid pixels cannot fall outside the EO valid_mask")
    if len(y) == 0:
        raise RuntimeError("No finite pixels with supported WorldCover labels are available")

    min_train = int(cfg["min_train_per_class"])
    min_test = int(cfg.get("min_test_per_class", 1))
    max_train = int(cfg["max_train_per_class"])
    split_contexts: list[dict[str, Any]] = []
    split_diagnostics: list[dict[str, Any]] = []
    for split_seed in split_seeds:
        raw_train, raw_test = split_groups(X, y, groups, float(cfg["test_fraction"]), split_seed)
        eligible = select_supported_classes(y, raw_train, min_train)
        train_support = _class_support(y[raw_train])
        test_support = _class_support(y[raw_test])
        unsupported_test = {
            label: count for label, count in test_support.items() if int(label) not in eligible
        }
        low_test = {
            str(label): int(np.count_nonzero(y[raw_test] == label))
            for label in eligible
            if int(np.count_nonzero(y[raw_test] == label)) < min_test
        }
        diagnostic: dict[str, Any] = {
            "seed": split_seed,
            "class_selection_source": "training_partition_only",
            "eligible_class_ids": eligible,
            "raw_train_class_support": train_support,
            "raw_test_class_support": test_support,
            "unsupported_test_class_support": unsupported_test,
            "low_test_support_diagnostic": low_test,
        }
        if len(eligible) < 2:
            diagnostic.update(
                status="skipped",
                reason="fewer than two classes meet training-only support threshold",
            )
            split_diagnostics.append(diagnostic)
            continue
        train_idx = raw_train[np.isin(y[raw_train], eligible)]
        test_idx = raw_test[np.isin(y[raw_test], eligible)]
        train_idx = subsample_training(train_idx, y, max_train, seed=split_seed)
        if len(test_idx) == 0:
            diagnostic.update(
                status="skipped",
                reason="held-out partition has no samples from training-eligible classes",
            )
            split_diagnostics.append(diagnostic)
            continue
        diagnostic.update(
            status="evaluated",
            train_n=len(train_idx),
            test_n=len(test_idx),
            train_group_count=len(set(groups[train_idx])),
            test_group_count=len(set(groups[test_idx])),
        )
        split_diagnostics.append(diagnostic)
        split_contexts.append(
            {"seed": split_seed, "train": train_idx, "test": test_idx, "eligible": eligible}
        )
    if not split_contexts:
        raise RuntimeError(
            "No spatial split supports evaluation under the training-only class protocol; "
            "enlarge the AOI or explicitly revise training thresholds"
        )

    report_class_ids = sorted(
        {label for context in split_contexts for label in context["eligible"]}
    )
    resolved_config_path = config_path.resolve()
    try:
        config_reference = resolved_config_path.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        config_reference = str(resolved_config_path)
    generated_utc = datetime.now(timezone.utc).isoformat()
    detailed: dict[str, Any] = {
        "schema_version": 2,
        "generated_utc": generated_utc,
        "provenance_kind": provenance["data_kind"],
        "evaluation_reference": "ESA WorldCover pseudo-label agreement",
        "interpretation_warning": (
            "Scores are agreement with WorldCover, not independent surveyed land-cover accuracy."
        ),
        "class_selection_protocol": "training_partition_only",
        "min_test_per_class_role": "diagnostic_only; never used to select eligible classes",
        "spatial_block_pixels": int(cfg["spatial_block_pixels"]),
        "requested_split_seeds": split_seeds,
        "evaluated_split_count": len(split_contexts),
        "split_diagnostics": split_diagnostics,
        "class_ids": report_class_ids,
        "class_names": [WORLDCOVER_CLASSES[class_id] for class_id in report_class_ids],
        "input_provenance_sha256": sha256_file(provenance_path),
        "input_stack_sha256": stack_sha256,
        "config_path": config_reference,
        "config_sha256": sha256_file(config_path),
        "experiments": {},
    }
    aggregate_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    for ablation in cfg["ablation"]:
        subset = _feature_columns(str(ablation))
        for classifier in cfg["classifiers"]:
            key = f"{classifier}__{ablation}"
            runs: list[dict[str, Any]] = []
            aggregate_matrix = np.zeros(
                (len(report_class_ids), len(report_class_ids)), dtype=np.int64
            )
            for context in split_contexts:
                train_idx = context["train"]
                test_idx = context["test"]
                eligible = context["eligible"]
                split_seed = int(context["seed"])
                model = _model(str(classifier), split_seed)
                model.fit(X[train_idx, subset], y[train_idx])
                prediction = model.predict(X[test_idx, subset])
                run_row = {
                    "method": str(classifier),
                    "features": str(ablation),
                    "data_kind": provenance["data_kind"],
                    "split_seed": split_seed,
                    "train_n": len(train_idx),
                    "test_n": len(test_idx),
                    "macro_f1": float(
                        f1_score(
                            y[test_idx], prediction, labels=eligible, average="macro", zero_division=0
                        )
                    ),
                    "weighted_f1": float(
                        f1_score(
                            y[test_idx],
                            prediction,
                            labels=eligible,
                            average="weighted",
                            zero_division=0,
                        )
                    ),
                    "accuracy_vs_worldcover": float(accuracy_score(y[test_idx], prediction)),
                }
                matrix = confusion_matrix(
                    y[test_idx], prediction, labels=report_class_ids
                ).astype(np.int64)
                aggregate_matrix += matrix
                report = classification_report(
                    y[test_idx],
                    prediction,
                    labels=eligible,
                    target_names=[WORLDCOVER_CLASSES[label] for label in eligible],
                    output_dict=True,
                    zero_division=0,
                )
                runs.append(
                    {
                        "seed": split_seed,
                        "row": run_row,
                        "eligible_class_ids": eligible,
                        "per_class": report,
                        "confusion_matrix": matrix.tolist(),
                    }
                )
                split_rows.append(run_row)
            macro = np.asarray([run["row"]["macro_f1"] for run in runs], dtype=float)
            weighted = np.asarray([run["row"]["weighted_f1"] for run in runs], dtype=float)
            accuracy = np.asarray(
                [run["row"]["accuracy_vs_worldcover"] for run in runs], dtype=float
            )
            row = {
                "method": str(classifier),
                "features": str(ablation),
                "data_kind": provenance["data_kind"],
                "split_count": len(runs),
                "train_n": float(np.mean([run["row"]["train_n"] for run in runs])),
                "test_n": float(np.mean([run["row"]["test_n"] for run in runs])),
                "macro_f1": float(np.mean(macro)),
                "macro_f1_std": float(np.std(macro, ddof=0)),
                "weighted_f1": float(np.mean(weighted)),
                "weighted_f1_std": float(np.std(weighted, ddof=0)),
                "accuracy_vs_worldcover": float(np.mean(accuracy)),
                "accuracy_vs_worldcover_std": float(np.std(accuracy, ddof=0)),
            }
            aggregate_rows.append(row)
            detailed["experiments"][key] = {
                "row": row,
                "runs": runs,
                "confusion_matrix": aggregate_matrix.tolist(),
                "confusion_matrix_note": (
                    "Sum of per-split matrices on the union of training-eligible classes."
                ),
            }

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(aggregate_rows)
    metrics_path = output / "geoai_metrics.csv"
    split_metrics_path = output / "geoai_split_metrics.csv"
    table.to_csv(metrics_path, index=False)
    pd.DataFrame(split_rows).to_csv(split_metrics_path, index=False)
    detailed["metrics_sha256"] = sha256_file(metrics_path)
    detailed["split_metrics_sha256"] = sha256_file(split_metrics_path)
    save_json(detailed, output / "geoai_details.json")
    return table
