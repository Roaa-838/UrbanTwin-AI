"""Bundle-integrity and rendering tests for the GeoAI spatial study."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

from urbantwin.common import sha256_file
from urbantwin.data.earth_observation import WORLDCOVER_CLASSES
from urbantwin.geoai import spatial_validation as spatial
from urbantwin.visualization.report_writer import ArtifactIntegrityError


def _configuration() -> dict:
    return {
        "schema_version": spatial.CONFIG_SCHEMA,
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
                "Largest prespecified candidate retaining three training-supported "
                "classes in every fold, selected without model scores."
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


def _baseline_analysis() -> dict[str, pd.DataFrame]:
    rows = []
    for method in ("random_forest", "hist_gradient_boosting"):
        for features in ("bands_only", "bands_plus_indices"):
            for class_id in (10, 20, 30, 50, 60, 80):
                rows.append(
                    {
                        "split_seed": 42,
                        "method": method,
                        "features": features,
                        "class_id": class_id,
                        "class_name": WORLDCOVER_CLASSES[class_id],
                        "f1": 0.5,
                        "support": 10,
                    }
                )
    return {
        "per_class": pd.DataFrame(rows),
        "majority_baselines": pd.DataFrame({"split_seed": [42], "macro_f1": [0.2]}),
        "paired_ablations": pd.DataFrame({"method": ["random_forest"], "delta": [0.0]}),
        "split_metrics": pd.DataFrame({"split_seed": [42], "macro_f1": [0.5]}),
    }


class _TrainingMajorityEstimator:
    """Fast estimator using training labels only and no held-out feature signal."""

    def fit(self, features: np.ndarray, labels: np.ndarray) -> _TrainingMajorityEstimator:
        assert len(features) == len(labels)
        values, counts = np.unique(labels, return_counts=True)
        self.prediction = int(values[np.argmax(counts)])
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.full(len(features), self.prediction, dtype=np.int16)


@pytest.fixture(scope="module")
def synthetic_bundle(tmp_path_factory: pytest.TempPathFactory):
    """Create one internally consistent bundle through production orchestration."""

    root = tmp_path_factory.mktemp("spatial-bundle")
    config_path = root / "spatial.yaml"
    config_path.write_text(
        yaml.safe_dump(_configuration(), sort_keys=False), encoding="utf-8", newline="\n"
    )
    data_dir = root / "eo"
    data_dir.mkdir()
    provenance_path = data_dir / "provenance.json"
    provenance_path.write_text('{"fixture":"synthetic-test-only"}\n', encoding="utf-8")
    stack_path = data_dir / "eo_stack.npz"
    stack_path.write_bytes(b"synthetic-test-stack")

    rows, columns = np.indices((180, 220))
    labels = np.choose((rows + columns) % 2, [10, 50]).astype(np.int16)
    transform = [100.0, 0.0, 500_000.0, 0.0, -100.0, 3_320_000.0]
    # Put class 80 in the 500--600 m annulus around fold zero. Every 500 m
    # training partition retains >=20 examples, whereas fold zero retains none
    # at 600 m. This satisfies the frozen, score-independent feasibility rule.
    geometry = spatial.build_spatial_folds(
        np.full(labels.shape, 10, dtype=np.int16),
        np.ones(labels.shape, dtype=bool),
        transform,
        block_pixels=20,
        tile_rows=3,
        tile_columns=3,
        buffer_distances_m=[500, 600],
        fold_seeds=[42, 142, 242, 342, 442, 542, 642, 742, 842],
    )
    class_80 = geometry[0].train_masks[500.0] & ~geometry[0].train_masks[600.0]
    labels[class_80] = 80
    bands = np.ones((5, *labels.shape), dtype=np.float32)
    features = np.zeros((labels.size, 7), dtype=np.float32)
    features[:, 0] = rows.ravel()
    features[:, 1] = columns.ravel()
    source = {
        "height": 180,
        "width": 220,
        "crs": "EPSG:32636",
        "transform": transform,
        "sentinel_stac_item_ids": ["synthetic-sentinel-test-fixture"],
        "worldcover_stac_item_ids": ["synthetic-worldcover-test-fixture"],
    }
    baseline = {"details": {}, "hashes": {"details": "synthetic-baseline-hash"}}
    baseline_analysis = _baseline_analysis()
    baseline_spatial = pd.DataFrame(
        {
            "split_seed": [42],
            "train_n": [100],
            "test_n": [20],
            "minimum_test_to_train_center_distance_m": [10.0],
        }
    )

    patches = pytest.MonkeyPatch()
    patches.setattr(
        spatial,
        "_validate_source_stack",
        lambda *_args, **_kwargs: (
            bands,
            labels,
            features,
            source,
            provenance_path,
            stack_path,
        ),
    )
    patches.setattr(spatial, "validate_geoai_artifacts", lambda _root: baseline)
    patches.setattr(spatial, "analyze_geoai_details", lambda _details: baseline_analysis)
    patches.setattr(
        spatial,
        "_baseline_spatial_diagnostics",
        lambda *_args, **_kwargs: baseline_spatial.copy(),
    )
    patches.setattr(
        spatial.geoai_train,
        "_model",
        lambda _method, _seed: _TrainingMajorityEstimator(),
    )

    bundle = root / "bundle"
    spatial.run_spatial_validation(data_dir, bundle, config_path, root)
    try:
        yield SimpleNamespace(root=root, bundle=bundle, config=config_path, data=data_dir)
    finally:
        patches.undo()


def _copy_bundle(synthetic_bundle: SimpleNamespace, tmp_path: Path) -> Path:
    destination = tmp_path / "bundle"
    shutil.copytree(synthetic_bundle.bundle, destination)
    return destination


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8", newline="\n")


def _rehash(bundle: Path, artifact_name: str) -> None:
    provenance_path = bundle / "provenance.json"
    provenance = _read_json(provenance_path)
    provenance["artifact_hashes"][artifact_name] = sha256_file(bundle / artifact_name)
    _write_json(provenance_path, provenance)


def test_valid_synthetic_bundle_reconstructs_scientific_tables(synthetic_bundle):
    validated = spatial.validate_spatial_validation_bundle(
        synthetic_bundle.bundle,
        synthetic_bundle.config,
        synthetic_bundle.data,
        synthetic_bundle.root,
    )

    assert len(validated["fold_manifest"]) == 18
    assert len(validated["fold_metrics"]) == 72
    assert len(validated["aggregate_metrics"]) == 10
    assert set(validated["aggregate_metrics"]["test_n"]) == {180 * 220}
    assert validated["details"]["core_class_ids"] == [10, 50, 80]
    buffered = validated["fold_manifest"].loc[
        validated["fold_manifest"]["buffer_m"] == 500,
        "minimum_train_test_center_distance_m",
    ]
    assert (buffered > 500).all()


@pytest.mark.parametrize("artifact_name", ["fold_metrics.csv", "spatial_details.json"])
def test_validator_rejects_missing_required_artifacts(
    synthetic_bundle, tmp_path, artifact_name
):
    bundle = _copy_bundle(synthetic_bundle, tmp_path)
    (bundle / artifact_name).unlink()

    with pytest.raises(ArtifactIntegrityError, match="artifact is missing"):
        spatial.validate_spatial_validation_bundle(
            bundle, synthetic_bundle.config, synthetic_bundle.data, synthetic_bundle.root
        )


def test_validator_rejects_corrupted_artifact_bytes(synthetic_bundle, tmp_path):
    bundle = _copy_bundle(synthetic_bundle, tmp_path)
    with (bundle / "aggregate_metrics.csv").open("a", encoding="utf-8") as stream:
        stream.write("corrupted\n")

    with pytest.raises(ArtifactIntegrityError, match="hash mismatch"):
        spatial.validate_spatial_validation_bundle(
            bundle, synthetic_bundle.config, synthetic_bundle.data, synthetic_bundle.root
        )


def test_validator_rejects_semantically_changed_configuration(synthetic_bundle, tmp_path):
    changed = _configuration()
    changed["random_seed"] = 99
    changed_config = tmp_path / "changed.yaml"
    changed_config.write_text(
        yaml.safe_dump(changed, sort_keys=False), encoding="utf-8", newline="\n"
    )

    with pytest.raises(ArtifactIntegrityError, match="configuration provenance"):
        spatial.validate_spatial_validation_bundle(
            synthetic_bundle.bundle,
            changed_config,
            synthetic_bundle.data,
            synthetic_bundle.root,
        )


def test_validator_rejects_changed_live_source_bytes(synthetic_bundle, tmp_path):
    copied_data = tmp_path / "eo"
    shutil.copytree(synthetic_bundle.data, copied_data)
    (copied_data / "eo_stack.npz").write_bytes(b"different-source-bytes")

    with pytest.raises(ArtifactIntegrityError, match="EO source hashes do not match"):
        spatial.validate_spatial_validation_bundle(
            synthetic_bundle.bundle,
            synthetic_bundle.config,
            copied_data,
            synthetic_bundle.root,
        )


def test_validator_rejects_rehashed_class_eligibility_mismatch(synthetic_bundle, tmp_path):
    bundle = _copy_bundle(synthetic_bundle, tmp_path)
    path = bundle / "class_eligibility.csv"
    eligibility = pd.read_csv(path)
    eligibility.loc[0, "training_support"] += 1
    eligibility.to_csv(path, index=False, lineterminator="\n")
    _rehash(bundle, path.name)

    with pytest.raises(ArtifactIntegrityError, match="eligibility table differs"):
        spatial.validate_spatial_validation_bundle(
            bundle, synthetic_bundle.config, synthetic_bundle.data, synthetic_bundle.root
        )


def test_validator_rejects_rehashed_incomplete_fold_table(synthetic_bundle, tmp_path):
    bundle = _copy_bundle(synthetic_bundle, tmp_path)
    path = bundle / "fold_manifest.csv"
    pd.read_csv(path).iloc[:-1].to_csv(path, index=False, lineterminator="\n")
    _rehash(bundle, path.name)

    with pytest.raises(ArtifactIntegrityError, match="unexpected row counts"):
        spatial.validate_spatial_validation_bundle(
            bundle, synthetic_bundle.config, synthetic_bundle.data, synthetic_bundle.root
        )


def test_validator_rejects_missing_experiment_record(synthetic_bundle, tmp_path):
    bundle = _copy_bundle(synthetic_bundle, tmp_path)
    path = bundle / "spatial_details.json"
    details = _read_json(path)
    details["experiments"].pop(next(iter(details["experiments"])))
    _write_json(path, details)
    _rehash(bundle, path.name)

    with pytest.raises(ArtifactIntegrityError, match="incomplete experiment axis"):
        spatial.validate_spatial_validation_bundle(
            bundle, synthetic_bundle.config, synthetic_bundle.data, synthetic_bundle.root
        )


def test_validator_rejects_confusion_total_mismatch(synthetic_bundle, tmp_path):
    bundle = _copy_bundle(synthetic_bundle, tmp_path)
    path = bundle / "spatial_details.json"
    details = _read_json(path)
    experiment = next(iter(details["experiments"].values()))
    experiment["aggregate"]["confusion_matrix"][0][0] -= 1
    _write_json(path, details)
    _rehash(bundle, path.name)

    with pytest.raises(ArtifactIntegrityError, match="Aggregate confusion is invalid"):
        spatial.validate_spatial_validation_bundle(
            bundle, synthetic_bundle.config, synthetic_bundle.data, synthetic_bundle.root
        )


def test_validator_rejects_metric_disagreement_with_confusion(synthetic_bundle, tmp_path):
    bundle = _copy_bundle(synthetic_bundle, tmp_path)
    path = bundle / "aggregate_metrics.csv"
    aggregate = pd.read_csv(path)
    aggregate.loc[0, "macro_f1"] = 0.25
    aggregate.to_csv(path, index=False, lineterminator="\n")
    _rehash(bundle, path.name)

    with pytest.raises(ArtifactIntegrityError, match="disagrees with confusion"):
        spatial.validate_spatial_validation_bundle(
            bundle, synthetic_bundle.config, synthetic_bundle.data, synthetic_bundle.root
        )


def test_renderer_validates_bundle_and_records_figure_hashes(synthetic_bundle, tmp_path):
    destination = tmp_path / "figures"
    figures = spatial.render_spatial_validation_figures(
        synthetic_bundle.bundle,
        destination,
        synthetic_bundle.config,
        synthetic_bundle.data,
        synthetic_bundle.root,
    )

    manifest = _read_json(destination / "publication_manifest.json")
    assert {path.name for path in figures} == set(manifest["figure_hashes"])
    assert all(sha256_file(path) == manifest["figure_hashes"][path.name] for path in figures)
    assert manifest["source_bundle_provenance_sha256"] == sha256_file(
        synthetic_bundle.bundle / "provenance.json"
    )
    assert manifest["source_role"].endswith("not field accuracy")


def test_renderer_rejects_invalid_source_bundle_before_publication(
    synthetic_bundle, tmp_path
):
    bundle = _copy_bundle(synthetic_bundle, tmp_path)
    (bundle / "aggregate_metrics.csv").write_text("corrupt\n", encoding="utf-8")
    destination = tmp_path / "figures"

    with pytest.raises(ArtifactIntegrityError, match="hash mismatch"):
        spatial.render_spatial_validation_figures(
            bundle,
            destination,
            synthetic_bundle.config,
            synthetic_bundle.data,
            synthetic_bundle.root,
        )
    assert not destination.exists()


def test_renderer_cleans_atomic_temporary_directory_after_failure(
    synthetic_bundle, tmp_path, monkeypatch
):
    destination = tmp_path / "failed-figures"

    def fail_render(*_args, **_kwargs):
        raise RuntimeError("synthetic rendering failure")

    monkeypatch.setattr(spatial, "_write_figures", fail_render)
    with pytest.raises(RuntimeError, match="synthetic rendering failure"):
        spatial.render_spatial_validation_figures(
            synthetic_bundle.bundle,
            destination,
            synthetic_bundle.config,
            synthetic_bundle.data,
            synthetic_bundle.root,
        )

    assert not destination.exists()
    assert not list(tmp_path.glob(".failed-figures.tmp-*"))
