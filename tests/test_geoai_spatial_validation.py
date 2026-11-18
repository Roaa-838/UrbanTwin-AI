"""Deterministic scientific invariants for the buffered GeoAI robustness study."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
from rasterio.transform import from_origin

from urbantwin.common import sha256_file
from urbantwin.geoai.spatial_validation import (
    _validate_artifact_hashes,
    build_spatial_folds,
    fold_classification_metrics,
    load_spatial_validation_config,
    run_spatial_validation,
    select_core_classes_from_training,
)
from urbantwin.visualization.report_writer import ArtifactIntegrityError


def _synthetic_grid() -> tuple[np.ndarray, np.ndarray]:
    """Return exactly 9x11 original blocks, with balanced repeating labels."""
    rows, cols = np.indices((18, 22))
    labels = np.choose((rows + 2 * cols) % 3, [10, 50, 80]).astype(np.int16)
    return labels, np.ones(labels.shape, dtype=bool)


def _folds(labels: np.ndarray, valid: np.ndarray):
    return build_spatial_folds(
        labels,
        valid,
        from_origin(500_000, 3_320_000, 100, 100),
        block_pixels=2,
        tile_rows=3,
        tile_columns=3,
        buffer_distances_m=[0, 500],
        fold_seeds=[42, 142, 242, 342, 442, 542, 642, 742, 842],
    )


def _valid_config() -> dict:
    return {
        "schema_version": "urbantwin.geoai-spatial-validation-config/v1",
        "random_seed": 42,
        "spatial_validation": {
            "block_pixels": 20,
            "tile_rows": 3,
            "tile_cols": 3,
            "expected_resolution_m": 10,
            "buffer_distances_m": [0, 500],
            "selection_buffer_m": 500,
            "buffer_feasibility_candidates_m": [100, 200, 300, 400, 500, 600],
            "buffer_selection_rule": (
                "Largest prespecified candidate retaining at least three "
                "training-supported classes in every fold."
            ),
            "fold_seeds": [42, 142, 242, 342, 442, 542, 642, 742, 842],
            "min_train_per_class": 20,
            "min_test_per_class": 5,
            "max_train_per_class": 1200,
            "expected_core_class_ids": [10, 50, 80],
            "classifiers": ["random_forest", "hist_gradient_boosting"],
            "ablation": ["bands_only", "bands_plus_indices"],
        },
    }


def test_group_aligned_tiles_cover_valid_pixels_once_and_keep_groups_disjoint():
    labels, valid = _synthetic_grid()
    folds = _folds(labels, valid)

    assert len(folds) == 9
    coverage = np.sum([fold.test_mask for fold in folds], axis=0)
    assert np.array_equal(coverage[valid], np.ones(np.count_nonzero(valid), dtype=int))

    original_groups = (np.indices(labels.shape)[0] // 2) * 11 + (np.indices(labels.shape)[1] // 2)
    for fold in folds:
        test_groups = set(original_groups[fold.test_mask])
        assert test_groups == set(fold.test_group_ids)
        for distance, train_mask in fold.train_masks.items():
            train_groups = set(original_groups[train_mask])
            assert test_groups.isdisjoint(train_groups), distance
            assert train_groups == set(fold.train_group_ids[distance])


def test_buffer_requires_strict_pixel_center_separation_from_entire_test_tile():
    labels, valid = _synthetic_grid()
    folds = _folds(labels, valid)

    for fold in folds:
        train_coords = np.argwhere(fold.train_masks[500.0])
        test_coords = np.argwhere(fold.test_mask)
        # Pixel size is 100 m in both axes. Every retained training center must
        # be farther than 500 m from every held-out center; equality is excluded.
        squared = np.sum(
            ((train_coords[:, None, :] - test_coords[None, :, :]) * 100.0) ** 2,
            axis=2,
        )
        assert np.all(np.sqrt(np.min(squared, axis=1)) > 500.0)


def test_core_class_selection_uses_buffered_training_labels_not_held_out_labels():
    labels, valid = _synthetic_grid()
    folds = _folds(labels, valid)
    core, support = select_core_classes_from_training(
        labels,
        folds,
        selection_buffer_m=500,
        min_train_per_class=3,
        expected_core_class_ids=[10, 50, 80],
    )

    changed = labels.copy()
    changed[folds[0].test_mask] = 99
    changed_core, changed_support = select_core_classes_from_training(
        changed,
        folds,
        selection_buffer_m=500,
        min_train_per_class=3,
        expected_core_class_ids=[10, 50, 80],
    )

    assert core == changed_core == [10, 50, 80]
    original_fold = support[
        (support["fold_index"] == 0) & support["class_id"].isin(core)
    ].reset_index(drop=True)
    mutated_fold = changed_support[
        (changed_support["fold_index"] == 0) & changed_support["class_id"].isin(core)
    ].reset_index(drop=True)
    assert original_fold.equals(mutated_fold)
    assert 99 not in core


def test_core_class_selection_rejects_unexpected_training_intersection():
    labels, valid = _synthetic_grid()
    with pytest.raises(ValueError, match="expected core"):
        select_core_classes_from_training(
            labels,
            _folds(labels, valid),
            selection_buffer_m=500,
            min_train_per_class=3,
            expected_core_class_ids=[10, 50, 80, 90],
        )


def test_per_fold_low_and_absent_class_support_is_diagnostic_not_a_score():
    metrics, aggregate, confusion = fold_classification_metrics(
        np.array([10, 10, 50]),
        np.array([10, 50, 50]),
        core_classes=[10, 50, 80],
        min_test_per_class=2,
    )
    metrics = metrics.set_index("class_id")

    assert metrics.loc[10, "status"] == "ok"
    assert metrics.loc[10, "test_support"] == 2
    assert np.isfinite(metrics.loc[10, "f1"])
    assert metrics.loc[50, "status"] == "low_support"
    assert metrics.loc[50, "test_support"] == 1
    assert np.isnan(metrics.loc[50, "precision"])
    assert np.isnan(metrics.loc[50, "recall"])
    assert np.isnan(metrics.loc[50, "f1"])
    assert metrics.loc[80, "status"] == "absent"
    assert metrics.loc[80, "test_support"] == 0
    assert np.isnan(metrics.loc[80, "f1"])
    assert aggregate["evaluated_samples"] == 3
    assert confusion.shape == (3, 3)


def test_config_loader_accepts_frozen_protocol_and_rejects_protocol_drift(tmp_path):
    import yaml

    path = tmp_path / "spatial.yaml"
    path.write_text(yaml.safe_dump(_valid_config(), sort_keys=False), encoding="utf-8")
    loaded = load_spatial_validation_config(path)
    assert loaded["expected_core_class_ids"] == [10, 50, 80]
    assert loaded["selection_buffer_m"] == 500.0

    invalid_cases = []
    wrong_fold_count = deepcopy(_valid_config())
    wrong_fold_count["spatial_validation"]["fold_seeds"] = [42]
    invalid_cases.append(wrong_fold_count)
    wrong_buffer = deepcopy(_valid_config())
    wrong_buffer["spatial_validation"]["buffer_distances_m"] = [0, 400]
    invalid_cases.append(wrong_buffer)
    missing_rule = deepcopy(_valid_config())
    del missing_rule["spatial_validation"]["buffer_selection_rule"]
    invalid_cases.append(missing_rule)
    wrong_classes = deepcopy(_valid_config())
    wrong_classes["spatial_validation"]["expected_core_class_ids"] = [10, 50]
    invalid_cases.append(wrong_classes)

    for index, invalid in enumerate(invalid_cases):
        invalid_path = tmp_path / f"invalid-{index}.yaml"
        invalid_path.write_text(yaml.safe_dump(invalid, sort_keys=False), encoding="utf-8")
        with pytest.raises(ValueError):
            load_spatial_validation_config(invalid_path)


def test_runner_refuses_to_overwrite_existing_bundle_before_reading_inputs(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()

    with pytest.raises(FileExistsError, match="will not be overwritten"):
        run_spatial_validation(
            tmp_path / "missing-eo",
            output,
            tmp_path / "missing-config.yaml",
            tmp_path / "missing-baseline",
        )


def test_artifact_hash_validation_rejects_corruption_and_unsafe_paths(tmp_path):
    artifact = tmp_path / "metrics.csv"
    artifact.write_text("metric,value\nmacro_f1,0.5\n", encoding="utf-8")
    record = {artifact.name: sha256_file(artifact)}

    assert _validate_artifact_hashes(tmp_path, record, {artifact.name}) == record

    artifact.write_text("metric,value\nmacro_f1,0.9\n", encoding="utf-8")
    with pytest.raises(ArtifactIntegrityError, match="hash mismatch"):
        _validate_artifact_hashes(tmp_path, record, {artifact.name})

    artifact.write_text("metric,value\nmacro_f1,0.5\n", encoding="utf-8")
    unsafe = {**record, "../outside.csv": "0" * 64}
    with pytest.raises(ArtifactIntegrityError, match="safe bundle-local"):
        _validate_artifact_hashes(tmp_path, unsafe, {artifact.name})
