"""Boundary and source-contract tests for the secondary GeoAI spatial study."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml

from urbantwin.common import ROOT, load_yaml, sha256_file
from urbantwin.geoai.spatial_validation import (
    SpatialFold,
    _affine_coefficients,
    _baseline_spatial_diagnostics,
    _git_state,
    _read_json,
    _validate_source_stack,
    build_spatial_folds,
    fold_classification_metrics,
    load_spatial_validation_config,
    select_core_classes_from_training,
)


def _configuration() -> dict:
    return deepcopy(load_yaml(ROOT / "configs/geoai_spatial_validation_v1.yaml"))


def _write_configuration(path: Path, value: dict) -> Path:
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda root: root.update(schema_version="wrong"), "Expected configuration schema"),
        (lambda root: root.update(spatial_validation=[]), "must contain a spatial_validation"),
        (
            lambda root: root["spatial_validation"].pop("block_pixels"),
            "configuration is missing",
        ),
        (
            lambda root: root["spatial_validation"].update(block_pixels=True),
            "must be a positive integer",
        ),
        (
            lambda root: root["spatial_validation"].update(tile_rows=2),
            "requires 3x3 tiles",
        ),
        (
            lambda root: root["spatial_validation"].update(expected_resolution_m=0),
            "positive and finite",
        ),
        (
            lambda root: root["spatial_validation"].update(buffer_distances_m=[0, -1]),
            "finite and non-negative",
        ),
        (
            lambda root: root["spatial_validation"].update(selection_buffer_m=0),
            "prespecified 500 m",
        ),
        (
            lambda root: root["spatial_validation"].update(
                buffer_feasibility_candidates_m=[100, 500]
            ),
            "frozen 100-600 m",
        ),
        (
            lambda root: root["spatial_validation"].update(buffer_selection_rule=""),
            "must document",
        ),
        (
            lambda root: root["spatial_validation"].update(fold_seeds=list(range(9))),
            "frozen sequence",
        ),
        (
            lambda root: root["spatial_validation"].update(classifiers=["random_forest"]),
            "frozen benchmark",
        ),
        (
            lambda root: root["spatial_validation"].update(ablation=["bands_only"]),
            "feature ablations",
        ),
        (
            lambda root: root["spatial_validation"].update(max_train_per_class=100),
            "support/subsampling",
        ),
    ],
)
def test_configuration_rejects_each_frozen_protocol_drift(
    tmp_path: Path, mutate, message: str
) -> None:
    configuration = _configuration()
    mutate(configuration)

    with pytest.raises((TypeError, ValueError), match=message):
        load_spatial_validation_config(
            _write_configuration(tmp_path / "spatial.yaml", configuration)
        )


@pytest.mark.parametrize(
    "transform, message",
    [([1, 0, 0], "six affine"), (object(), "must expose six affine")],
)
def test_affine_contract_rejects_invalid_transforms(transform, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _affine_coefficients(transform)


@pytest.mark.parametrize(
    "labels, valid, kwargs, message",
    [
        (np.ones(4), np.ones(4, dtype=bool), {}, "aligned two-dimensional"),
        (np.ones((2, 2)), np.ones((2, 3), dtype=bool), {}, "aligned two-dimensional"),
        (
            np.ones((2, 2)),
            np.ones((2, 2), dtype=bool),
            {"block_pixels": 0},
            "dimensions must be positive",
        ),
        (
            np.ones((2, 2)),
            np.ones((2, 2), dtype=bool),
            {"buffer_distances_m": [0, 0]},
            "non-empty and unique",
        ),
        (
            np.ones((2, 2)),
            np.ones((2, 2), dtype=bool),
            {"fold_seeds": [1]},
            "one unique seed per tile",
        ),
        (
            np.ones((2, 2)),
            np.ones((2, 2), dtype=bool),
            {"block_pixels": 2},
            "fewer base groups",
        ),
    ],
)
def test_fold_builder_rejects_invalid_geometry_contracts(
    labels: np.ndarray,
    valid: np.ndarray,
    kwargs: dict,
    message: str,
) -> None:
    arguments = {
        "block_pixels": 1,
        "tile_rows": 2,
        "tile_columns": 2,
        "buffer_distances_m": [0],
        "fold_seeds": [1, 2, 3, 4],
    }
    arguments.update(kwargs)
    with pytest.raises(ValueError, match=message):
        build_spatial_folds(labels, valid, [10, 0, 0, 0, -10, 0], **arguments)


def _one_fold(labels: np.ndarray) -> SpatialFold:
    mask = np.ones(labels.shape, dtype=bool)
    return SpatialFold(
        fold_index=0,
        tile_row=0,
        tile_col=0,
        seed=42,
        test_mask=np.zeros(labels.shape, dtype=bool),
        train_masks={500.0: mask},
        test_group_ids=(),
        train_group_ids={500.0: (0,)},
        tile_mask=np.zeros(labels.shape, dtype=bool),
        tile_bounds_pixels=(0, 0, 0, 0),
        minimum_separation_m={500.0: 501.0},
    )


def test_training_class_selection_rejects_missing_or_misaligned_partitions() -> None:
    labels = np.array([[10, 50], [80, 10]])

    with pytest.raises(ValueError, match="At least one"):
        select_core_classes_from_training(
            labels,
            [],
            selection_buffer_m=500,
            min_train_per_class=1,
            expected_core_class_ids=[10, 50, 80],
        )

    fold = _one_fold(labels)
    with pytest.raises(ValueError, match="no 400 m"):
        select_core_classes_from_training(
            labels,
            [fold],
            selection_buffer_m=400,
            min_train_per_class=1,
            expected_core_class_ids=[10, 50, 80],
        )

    misaligned = SpatialFold(
        **{**fold.__dict__, "train_masks": {500.0: np.ones((3, 3), dtype=bool)}}
    )
    with pytest.raises(ValueError, match="not aligned"):
        select_core_classes_from_training(
            labels,
            [misaligned],
            selection_buffer_m=500,
            min_train_per_class=1,
            expected_core_class_ids=[10, 50, 80],
        )


def test_empty_fold_metrics_are_explicitly_unsupported() -> None:
    per_class, aggregate, matrix = fold_classification_metrics(
        np.array([], dtype=int),
        np.array([], dtype=int),
        [10, 50, 80],
        min_test_per_class=1,
    )

    assert aggregate["support"] == 0
    assert aggregate["evaluated_samples"] == 0
    assert np.isnan(aggregate["macro_f1"])
    assert per_class["status"].tolist() == ["absent", "absent", "absent"]
    assert matrix.tolist() == [[0, 0, 0], [0, 0, 0], [0, 0, 0]]


@pytest.mark.parametrize(
    "truth, predictions, classes, minimum, message",
    [
        ([[10]], [10], [10], 1, "one-dimensional"),
        ([10], [10], [], 1, "non-empty unique"),
        ([10], [10], [10], 0, "must be positive"),
        ([10], [50], [10], 1, "fixed core axis"),
    ],
)
def test_fold_metrics_reject_invalid_evaluation_axes(
    truth, predictions, classes, minimum: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        fold_classification_metrics(
            np.asarray(truth), np.asarray(predictions), classes, minimum
        )


def _write_real_source(directory: Path) -> tuple[Path, Path]:
    directory.mkdir()
    rows, columns = 4, 5
    bands = np.ones((5, rows, columns), dtype=np.float32)
    bands[3] = 2
    labels = np.resize(np.array([10, 50, 80], dtype=np.int16), (rows, columns))
    valid = np.ones((rows, columns), dtype=bool)
    stack_path = directory / "eo_stack.npz"
    np.savez_compressed(stack_path, bands=bands, labels=labels, valid_mask=valid)
    provenance = {
        "schema_version": "urbantwin.eo-provenance/v1",
        "data_kind": "real",
        "sentinel_stac_collection": "sentinel-2-l2a",
        "worldcover_stac_collection": "esa-worldcover",
        "sentinel_stac_item_ids": ["S2-fixture"],
        "worldcover_stac_item_ids": ["WC-fixture"],
        "band_order": ["B02", "B03", "B04", "B08", "B11"],
        "crs": "EPSG:32636",
        "transform": [10, 0, 500000, 0, -10, 3320000],
        "width": columns,
        "height": rows,
        "processing": {
            "continuous_resampling": "bilinear",
            "categorical_resampling": "nearest",
            "clear_scl_classes": [4, 5, 6, 7],
        },
        "eo_stack_sha256": sha256_file(stack_path),
        "signed_asset_urls_recorded": False,
    }
    provenance_path = directory / "provenance.json"
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    return provenance_path, stack_path


def test_source_stack_accepts_aligned_real_projected_fixture(tmp_path: Path) -> None:
    data = tmp_path / "real"
    provenance_path, stack_path = _write_real_source(data)

    bands, labels, features, provenance, actual_provenance, actual_stack = (
        _validate_source_stack(data, 10)
    )

    assert bands.shape == (5, 4, 5)
    assert labels.shape == (4, 5)
    assert features.shape == (20, 7)
    np.testing.assert_allclose(features[:, :5], bands.reshape(5, -1).T)
    assert not np.array_equal(features[:, 0], labels.ravel())
    assert provenance["crs"] == "EPSG:32636"
    assert actual_provenance == provenance_path
    assert actual_stack == stack_path


def test_source_stack_rejects_missing_corrupt_and_inconsistent_inputs(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="validated real EO"):
        _validate_source_stack(tmp_path / "missing", 10)

    data = tmp_path / "real"
    provenance_path, stack_path = _write_real_source(data)
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["eo_stack_sha256"] = "0" * 64
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match"):
        _validate_source_stack(data, 10)

    provenance["eo_stack_sha256"] = sha256_file(stack_path)
    provenance["height"] = 99
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    with pytest.raises(ValueError, match="dimensions"):
        _validate_source_stack(data, 10)


def test_baseline_spatial_diagnostics_reconstruct_grouped_separation(tmp_path: Path) -> None:
    baseline_config = tmp_path / "baseline.yaml"
    baseline_config.write_text(
        yaml.safe_dump({"geoai": {"test_fraction": 0.25}}), encoding="utf-8"
    )
    rows, columns = np.indices((6, 6))
    labels = np.choose((rows + columns) % 3, [10, 50, 80]).astype(np.int16)
    valid = np.ones(labels.shape, dtype=bool)
    features = np.column_stack(
        [rows.ravel(), columns.ravel(), np.zeros((labels.size, 5))]
    )
    baseline = {
        "details": {
            "config_path": baseline_config.name,
            "spatial_block_pixels": 2,
            "requested_split_seeds": [7, 107],
        }
    }

    diagnostics = _baseline_spatial_diagnostics(
        features,
        labels,
        valid,
        [100, 0, 500000, 0, -100, 3320000],
        baseline,
        tmp_path,
    )

    assert diagnostics["split_seed"].tolist() == [7, 107]
    assert (diagnostics["train_n"] + diagnostics["test_n"]).eq(labels.size).all()
    assert (diagnostics["minimum_test_to_train_center_distance_m"] >= 100).all()
    assert diagnostics["distance_direction"].eq(
        "held_out_test_pixel_to_nearest_training_pixel"
    ).all()

    baseline_config.write_text("{}\n", encoding="utf-8")
    with pytest.raises(TypeError, match="lacks its geoai mapping"):
        _baseline_spatial_diagnostics(
            features,
            labels,
            valid,
            [100, 0, 500000, 0, -100, 3320000],
            baseline,
            tmp_path,
        )


def test_json_and_git_metadata_fail_closed(tmp_path: Path, monkeypatch) -> None:
    invalid = tmp_path / "invalid.json"
    invalid.write_text("not-json", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid JSON artifact"):
        _read_json(invalid)

    array = tmp_path / "array.json"
    array.write_text("[]", encoding="utf-8")
    with pytest.raises(TypeError, match="JSON object"):
        _read_json(array)

    def unavailable(*_args, **_kwargs):
        raise OSError("git is unavailable")

    monkeypatch.setattr("urbantwin.geoai.spatial_validation.subprocess.run", unavailable)
    assert _git_state(tmp_path) == {"commit": "unavailable", "working_tree_dirty": None}
