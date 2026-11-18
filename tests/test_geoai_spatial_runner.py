"""Runner-level tests for the atomic GeoAI spatial robustness workflow."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from urbantwin.common import ROOT, sha256_file
from urbantwin.geoai import spatial_validation as spatial

FIGURE_NAMES = (
    "spatial_fold_design.png",
    "baseline_per_class_f1_support.png",
    "spatial_robustness_comparison.png",
    "spatial_confusion_random_forest_indices.png",
)


def _copy_config(tmp_path: Path) -> Path:
    source = ROOT / "configs/geoai_spatial_validation_v1.yaml"
    destination = tmp_path / "spatial-validation.yaml"
    destination.write_bytes(source.read_bytes())
    return destination


def _synthetic_source(tmp_path: Path, shape: tuple[int, int] = (180, 220)):
    """Build an in-memory raster with the frozen 9x11 base-group geometry.

    Class 80 occupies the 500--600 m ring around the central tile. This makes
    500 m the largest candidate retaining all three training-supported classes,
    so the real score-independent feasibility rule is exercised.
    """

    rows, columns = np.indices(shape)
    labels = np.where((rows + columns) % 2, 10, 50).astype(np.int16)
    if shape == (180, 220):
        row_distance = np.maximum.reduce([60 - rows, rows - 119, np.zeros_like(rows)])
        column_distance = np.maximum.reduce(
            [80 - columns, columns - 159, np.zeros_like(columns)]
        )
        central_distance_m = 10 * np.sqrt(row_distance**2 + column_distance**2)
        labels[(central_distance_m > 500) & (central_distance_m <= 600)] = 80
    else:
        labels[(rows + 2 * columns) % 3 == 2] = 80

    bands = np.ones((5, *shape), dtype=np.float32)
    features = np.zeros((labels.size, 7), dtype=np.float64)
    # A unique, non-target row identifier lets the test prove that every
    # estimator sees disjoint fit and prediction rows.
    features[:, 0] = np.arange(labels.size)
    features[:, 1] = rows.ravel()
    features[:, 2] = columns.ravel()

    data_dir = tmp_path / "synthetic-eo"
    data_dir.mkdir()
    provenance_path = data_dir / "provenance.json"
    stack_path = data_dir / "eo_stack.npz"
    provenance_path.write_text('{"data_kind":"synthetic_test_fixture"}\n', encoding="utf-8")
    stack_path.write_bytes(b"synthetic test fixture; not a research raster")
    source = {
        "sentinel_stac_item_ids": ["synthetic-test-item"],
        "worldcover_stac_item_ids": ["synthetic-test-labels"],
        "crs": "EPSG:32636",
        "transform": [10.0, 0.0, 500_000.0, 0.0, -10.0, 3_320_000.0],
    }
    return data_dir, bands, labels, features, source, provenance_path, stack_path


def _baseline_analysis() -> dict[str, pd.DataFrame]:
    return {
        "per_class": pd.DataFrame(
            {
                "method": ["random_forest"],
                "features": ["bands_only"],
                "class_id": [10],
                "class_name": ["Tree cover"],
                "f1": [0.5],
                "support": [100],
                "split_seed": [42],
            }
        ),
        "split_metrics": pd.DataFrame({"split_seed": [42], "macro_f1": [0.5]}),
        "paired_ablations": pd.DataFrame({"method": ["random_forest"], "delta": [0.0]}),
        "majority_baselines": pd.DataFrame({"split_seed": [42], "macro_f1": [0.2]}),
    }


def _install_fast_dependencies(monkeypatch, source_tuple):
    """Replace only expensive/external trust boundaries, not the runner."""

    fitted_models = []

    class TrainingMajorityEstimator:
        def fit(self, features, labels):
            values, counts = np.unique(labels, return_counts=True)
            self.prediction = int(values[np.argmax(counts)])
            self.fit_ids = set(np.asarray(features[:, 0], dtype=int))
            fitted_models.append(self)
            return self

        def predict(self, features):
            self.predict_ids = set(np.asarray(features[:, 0], dtype=int))
            return np.full(len(features), self.prediction, dtype=np.int16)

    baseline = {"hashes": {"details.json": "a" * 64}, "details": {"fixture": True}}
    monkeypatch.setattr(spatial, "validate_geoai_artifacts", lambda _root: baseline)
    monkeypatch.setattr(spatial, "_validate_source_stack", lambda *_args: source_tuple[1:])
    monkeypatch.setattr(spatial, "analyze_geoai_details", lambda _details: _baseline_analysis())
    monkeypatch.setattr(
        spatial,
        "_baseline_spatial_diagnostics",
        lambda *_args: pd.DataFrame(
            {
                "split_seed": [42],
                "minimum_test_to_train_center_distance_m": [10.0],
            }
        ),
    )
    monkeypatch.setattr(
        spatial.geoai_train,
        "_model",
        lambda _method, _seed: TrainingMajorityEstimator(),
    )
    return fitted_models


def _write_fixture_figures(directory, *_args):
    paths = []
    for name in FIGURE_NAMES:
        path = directory / name
        path.write_bytes(b"synthetic test figure")
        paths.append(path)
    return paths


def test_runner_publishes_validated_bundle_atomically_with_disjoint_fit_rows(
    tmp_path, monkeypatch
):
    config_path = _copy_config(tmp_path)
    source_tuple = _synthetic_source(tmp_path)
    data_dir = source_tuple[0]
    fitted_models = _install_fast_dependencies(monkeypatch, source_tuple)
    destination = tmp_path / "spatial-bundle"
    validation_calls = []

    monkeypatch.setattr(spatial, "_write_figures", _write_fixture_figures)

    def validate_before_publish(directory, config_file, source_dir, baseline_root):
        directory = Path(directory)
        assert directory.name.startswith(".spatial-bundle.tmp-")
        assert not destination.exists()
        assert Path(config_file) == config_path
        assert Path(source_dir) == data_dir
        assert Path(baseline_root) == tmp_path / "baseline"
        provenance = json.loads((directory / "provenance.json").read_text(encoding="utf-8"))
        for relative, expected in provenance["artifact_hashes"].items():
            assert sha256_file(directory / relative) == expected
        assert provenance["figure_files"] == list(FIGURE_NAMES)
        validation_calls.append(directory.name)
        return {"provenance": provenance}

    monkeypatch.setattr(spatial, "validate_spatial_validation_bundle", validate_before_publish)

    aggregate = spatial.run_spatial_validation(
        data_dir,
        destination,
        config_path,
        tmp_path / "baseline",
    )

    assert len(validation_calls) == 1
    assert destination.is_dir()
    assert not list(tmp_path.glob(".spatial-bundle.tmp-*"))
    assert len(aggregate) == 10
    assert set(aggregate["test_n"]) == {39_600}

    fold_metrics = pd.read_csv(destination / "fold_metrics.csv")
    majority = pd.read_csv(destination / "majority_baselines.csv")
    manifest = pd.read_csv(destination / "fold_manifest.csv")
    eligibility = pd.read_csv(destination / "class_eligibility.csv")
    details = json.loads((destination / "spatial_details.json").read_text(encoding="utf-8"))
    assert len(fold_metrics) == 72
    assert len(majority) == 18
    assert eligibility["selection_source"].eq("training_partition_only").all()
    assert set(manifest["condition"]) == {
        "unbuffered_tiled_control",
        "strict_500m_buffer",
    }
    assert manifest.groupby("fold_index")["test_core_pixels"].nunique().eq(1).all()
    assert (
        manifest.loc[
            manifest["condition"] == "strict_500m_buffer",
            "minimum_train_test_center_distance_m",
        ]
        > 500
    ).all()
    for fold in details["folds"]:
        unbuffered = fold["conditions"]["unbuffered_tiled_control"]
        buffered = fold["conditions"]["strict_500m_buffer"]
        assert unbuffered["test_n"] == buffered["test_n"]
        assert set(fold["test_group_ids"]).isdisjoint(unbuffered["training_group_ids"])
        assert set(fold["test_group_ids"]).isdisjoint(buffered["training_group_ids"])

    # Nine folds x two conditions x two models x two feature sets.
    assert len(fitted_models) == 72
    assert all(model.fit_ids.isdisjoint(model.predict_ids) for model in fitted_models)


def test_runner_rejects_missing_frozen_baseline_before_reading_source(tmp_path, monkeypatch):
    config_path = _copy_config(tmp_path)
    destination = tmp_path / "spatial-bundle"
    monkeypatch.setattr(spatial, "validate_geoai_artifacts", lambda _root: None)
    monkeypatch.setattr(
        spatial,
        "_validate_source_stack",
        lambda *_args: pytest.fail("source must not be read without the frozen baseline"),
    )

    with pytest.raises(FileNotFoundError, match="frozen validated GeoAI result bundle"):
        spatial.run_spatial_validation(
            tmp_path / "missing-source",
            destination,
            config_path,
            tmp_path / "missing-baseline",
        )

    assert not destination.exists()


def test_runner_rejects_raster_with_wrong_frozen_group_geometry(tmp_path, monkeypatch):
    config_path = _copy_config(tmp_path)
    source_tuple = _synthetic_source(tmp_path, shape=(20, 20))
    _install_fast_dependencies(monkeypatch, source_tuple)
    destination = tmp_path / "spatial-bundle"

    with pytest.raises(ValueError, match="exactly 9 by 11"):
        spatial.run_spatial_validation(
            source_tuple[0],
            destination,
            config_path,
            tmp_path / "baseline",
        )

    assert not destination.exists()


def test_runner_removes_temporary_bundle_after_keyboard_interrupt(tmp_path, monkeypatch):
    config_path = _copy_config(tmp_path)
    source_tuple = _synthetic_source(tmp_path)
    _install_fast_dependencies(monkeypatch, source_tuple)
    destination = tmp_path / "spatial-bundle"
    interrupted_directories = []

    def two_condition_feasibility(labels, valid, transform, cfg):
        folds = spatial.build_spatial_folds(
            labels,
            valid,
            transform,
            block_pixels=cfg["block_pixels"],
            tile_rows=cfg["tile_rows"],
            tile_columns=cfg["tile_cols"],
            buffer_distances_m=cfg["buffer_distances_m"],
            fold_seeds=cfg["fold_seeds"],
        )
        rows = [
            {
                "buffer_m": distance,
                "fold_index": fold.fold_index,
                "fold_seed": fold.seed,
                "eligible_class_count": 3,
                "training_pixel_count": int(fold.train_masks[distance].sum()),
                "minimum_train_test_center_distance_m": fold.minimum_separation_m[distance],
                "selection_input": "geometry_and_training_support_only",
            }
            for fold in folds
            for distance in (0.0, 500.0)
        ]
        return pd.DataFrame(rows), folds

    def interrupt_during_figures(directory, *_args):
        interrupted_directories.append(Path(directory))
        assert (Path(directory) / "aggregate_metrics.csv").is_file()
        raise KeyboardInterrupt("simulated user interruption")

    monkeypatch.setattr(spatial, "_feasibility_table", two_condition_feasibility)
    monkeypatch.setattr(spatial, "_write_figures", interrupt_during_figures)

    with pytest.raises(KeyboardInterrupt, match="simulated user interruption"):
        spatial.run_spatial_validation(
            source_tuple[0],
            destination,
            config_path,
            tmp_path / "baseline",
        )

    assert len(interrupted_directories) == 1
    assert not interrupted_directories[0].exists()
    assert not destination.exists()
    assert not list(tmp_path.glob(".spatial-bundle.tmp-*"))
