from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from urbantwin.common import sha256_file
from urbantwin.observed.artifacts import (
    ObservedArtifactIntegrityError,
    validate_observed_artifacts,
)


def _write_csv(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False, lineterminator="\n")


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _metric(
    model: str,
    feature: str,
    budget: str,
    protocol: str,
    rows: int,
    *,
    mae: float,
    rmse: float,
) -> dict:
    return {
        "evaluation": "frozen_holdout",
        "protocol": protocol,
        "model": model,
        "feature_set": feature,
        "information_budget": budget,
        "rows": rows,
        "inferred_sites": 1,
        "mae": mae,
        "rmse": rmse,
        "median_absolute_error": 1.0,
        "mean_bias": 0.5,
        "mean_poisson_deviance": 0.4,
        "mae_ci95_low": 1.0,
        "mae_ci95_high": 3.0,
        "rmse_ci95_low": 2.0,
        "rmse_ci95_high": 4.0,
    }


def _observed_bundle(root: Path) -> Path:
    permission = {
        "status": "author_reported_academic_research_reuse_authorized",
        "public_redistribution_status": "requires_exact_written_terms",
    }
    configuration = {
        "schema_version": "synthetic-test-config",
        "permission": permission,
    }
    config = root / "configs" / "observed_traffic.yaml"
    config.parent.mkdir(parents=True)
    config.write_text(json.dumps(configuration), encoding="utf-8")
    source = root / "data" / "observed" / "tfc" / "normalized_v1"
    source.mkdir(parents=True)
    source_manifest = {
        "data_kind": "real_observed_traffic",
        "license": {"status": "unspecified_by_publisher", "redistribution_permitted": False},
        "sources": {
            "counts": {"sha256": "a" * 64},
            "locations": {"sha256": "b" * 64},
        },
    }
    source_path = source / "source_manifest.json"
    source_path.write_text(json.dumps(source_manifest), encoding="utf-8")
    output = root / "results" / "observed_traffic"
    (output / "figures").mkdir(parents=True)
    training_observations = [f"train-{index}" for index in range(5)]
    training_sites = [f"site-{index}" for index in range(5)]
    selection = {
        "selection_partition": "training_only",
        "selection_protocol": "five_fold_grouped_cv_by_inferred_site",
        "criterion": "lowest_mean_mae_then_mean_rmse",
        "model": "random_forest",
        "feature_set": "combined",
        "mean_mae": 2.0,
        "mean_rmse": 3.0,
        "candidate_count": 2,
        "holdout_scores_used_for_selection": False,
    }
    lowest_holdout = {
        "selection_role": "descriptive_only_not_used_for_model_selection",
        "model": "histogram_gradient_boosting",
        "feature_set": "combined",
        "mae": 1.0,
        "rmse": 2.0,
    }
    split = {
        "schema_version": "urbantwin.observed-split/v1",
        "protocol": "future_date_and_unseen_inferred_site_holdout",
        "train_dates": ["2024-07-01"],
        "test_dates": ["2024-11-01"],
        "train_rows": 5,
        "test_rows": 2,
        "train_inferred_sites": 5,
        "test_inferred_sites": 1,
        "site_overlap": [],
        "excluded_ambiguous_rows": 0,
        "inferred_site_definition": "SYNTHETIC TEST grouping",
        "training_observation_ids_sha256": _canonical_hash(training_observations),
        "test_observation_ids_sha256": _canonical_hash(["test-0", "test-1"]),
        "model_selection": selection,
        "lowest_observed_holdout_point_estimate": lowest_holdout,
    }
    (output / "split_manifest.json").write_text(json.dumps(split), encoding="utf-8")
    _write_csv(
        output / "observed_metrics.csv",
        [
            _metric(
                "random_forest",
                "combined",
                "measurement_free_estimation",
                "future_date_and_unseen_inferred_site_holdout",
                2,
                mae=5.0,
                rmse=6.0,
            ),
            _metric(
                "histogram_gradient_boosting",
                "combined",
                "measurement_free_estimation",
                "future_date_and_unseen_inferred_site_holdout",
                2,
                mae=1.0,
                rmse=2.0,
            ),
            _metric(
                "persistence",
                "previous_observed_interval",
                "online_forecast_with_previous_measurement",
                "online_one_step_exactly_adjacent_intervals",
                1,
                mae=2.0,
                rmse=3.0,
            ),
        ],
    )
    _write_csv(
        output / "observed_cv_metrics.csv",
        [
            record
            for fold in range(1, 6)
            for record in (
                {
                    "fold": fold,
                    "model": "random_forest",
                    "feature_set": "combined",
                    "information_budget": "measurement_free_estimation",
                    "train_rows": 4,
                    "validation_rows": 1,
                    "train_inferred_sites": 4,
                    "validation_inferred_sites": 1,
                    "mae": 2.0,
                    "rmse": 3.0,
                },
                {
                    "fold": fold,
                    "model": "histogram_gradient_boosting",
                    "feature_set": "combined",
                    "information_budget": "measurement_free_estimation",
                    "train_rows": 4,
                    "validation_rows": 1,
                    "train_inferred_sites": 4,
                    "validation_inferred_sites": 1,
                    "mae": 4.0,
                    "rmse": 5.0,
                },
            )
        ],
    )
    _write_csv(
        output / "observed_cv_predictions.csv",
        [
            {
                "fold": fold,
                "observation_id": training_observations[fold - 1],
                "inferred_site_id": training_sites[fold - 1],
                "model": model,
                "feature_set": "combined",
                "protocol": "training_only_five_fold_grouped_oof",
                "information_budget": "measurement_free_estimation",
                "observed_vehicle_total": 9 + fold,
                "prediction": 8 + fold + prediction_offset,
            }
            for fold in range(1, 6)
            for model, prediction_offset in (
                ("random_forest", 0),
                ("histogram_gradient_boosting", 1),
            )
        ],
    )
    _write_csv(
        output / "observed_predictions.csv",
        [
            {
                "observation_id": f"test-{index}",
                "model": model,
                "feature_set": "combined",
                "protocol": "future_date_and_unseen_inferred_site_holdout",
                "information_budget": "measurement_free_estimation",
                "observed_vehicle_total": 10 + index,
                "prediction": 9 + index + prediction_offset,
            }
            for model, prediction_offset in (
                ("random_forest", 0),
                ("histogram_gradient_boosting", 1),
            )
            for index in range(2)
        ]
        + [
            {
                "observation_id": "test-1",
                "model": "persistence",
                "feature_set": "previous_observed_interval",
                "protocol": "online_one_step_exactly_adjacent_intervals",
                "information_budget": "online_forecast_with_previous_measurement",
                "observed_vehicle_total": 11,
                "prediction": 10,
            }
        ],
    )
    _write_csv(
        output / "observed_site_errors.csv",
        [{"model": "random_forest", "inferred_site_id": "synthetic", "mae": 5}],
    )
    _write_csv(
        output / "observed_vehicle_classes.csv",
        [
            {
                "vehicle_class": f"synthetic_class_{index}",
                "observed_vehicles": index + 1,
                "share_of_observed_vehicles": (index + 1) / 78,
            }
            for index in range(12)
        ],
    )
    _write_csv(
        output / "observed_characterization.csv",
        [{"observed_date": "2024-11-01", "observed_vehicles": 21}],
    )
    (output / "segment_features.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": []}), encoding="utf-8"
    )
    (output / "historic_eo_context.json").write_text("[]", encoding="utf-8")
    (output / "figures" / "synthetic_chart.png").write_bytes(b"SYNTHETIC TEST IMAGE")
    cv_folds = []
    for fold in range(1, 6):
        validation_index = fold - 1
        fit_observations = [
            value for index, value in enumerate(training_observations) if index != validation_index
        ]
        fit_sites = [
            value for index, value in enumerate(training_sites) if index != validation_index
        ]
        cv_folds.append(
            {
                "fold": fold,
                "fit_rows": 4,
                "validation_rows": 1,
                "fit_inferred_sites": 4,
                "validation_inferred_sites": 1,
                "site_overlap": [],
                "fit_observation_ids_sha256": _canonical_hash(fit_observations),
                "validation_observation_ids_sha256": _canonical_hash(
                    [training_observations[validation_index]]
                ),
                "fit_inferred_site_ids_sha256": _canonical_hash(fit_sites),
                "validation_inferred_site_ids_sha256": _canonical_hash(
                    [training_sites[validation_index]]
                ),
            }
        )
    cv_split = {
        "schema_version": "urbantwin.observed-cv-split/v1",
        "protocol": "training_only_five_fold_grouped_oof",
        "group_key": "inferred_site_id",
        "n_splits": 5,
        "training_rows": 5,
        "training_inferred_sites": 5,
        "training_observation_ids_sha256": _canonical_hash(training_observations),
        "source_manifest_sha256": sha256_file(source_path),
        "source_counts_sha256": "a" * 64,
        "source_locations_sha256": "b" * 64,
        "config_sha256": sha256_file(config),
        "model_selection": selection,
        "folds": cv_folds,
        "leakage_controls": ["SYNTHETIC TEST folds are site-disjoint"],
    }
    (output / "cv_split_manifest.json").write_text(json.dumps(cv_split), encoding="utf-8")
    artifact_files = [
        path
        for path in output.rglob("*")
        if path.is_file() and path.name != "provenance.json"
    ]
    provenance = {
        "schema_version": "urbantwin.observed-benchmark/v2",
        "data_kind": "real_observed_traffic_analysis",
        "source_counts_sha256": "a" * 64,
        "source_locations_sha256": "b" * 64,
        "source_manifest_sha256": sha256_file(source_path),
        "config_sha256": sha256_file(config),
        "configuration": configuration,
        "split_manifest_sha256": sha256_file(output / "split_manifest.json"),
        "cv_split_manifest_sha256": sha256_file(output / "cv_split_manifest.json"),
        "git": {"commit": "c" * 40, "dirty": False},
        "license": {"status": "unspecified_by_publisher", "redistribution_permitted": False},
        "portal_license": {
            "status": "unspecified_by_publisher",
            "redistribution_permitted": False,
        },
        "author_reported_permission": permission,
        "model_selection": selection,
        "lowest_observed_holdout_point_estimate": lowest_holdout,
        "source_labels": {
            "traffic": "measured TfC directional link counts (SYNTHETIC TEST BUNDLE)",
            "geometry": "SYNTHETIC TEST",
            "eo": "not included",
            "simulation": "not run; scientific gate failed",
        },
        "study_gate": {
            "allow_measured_demand_sumo": False,
            "blockers": ["SYNTHETIC TEST has no turning movements"],
        },
        "artifact_hashes": {
            path.relative_to(output).as_posix(): sha256_file(path) for path in artifact_files
        },
        "figure_files": ["figures/synthetic_chart.png"],
    }
    (output / "provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
    return output


def test_observed_validator_accepts_bound_bundle(tmp_path: Path) -> None:
    output = _observed_bundle(tmp_path)

    verified = validate_observed_artifacts(tmp_path, output)

    assert verified is not None
    assert len(verified["metrics"]) == 3
    assert len(verified["predictions"]) == 5
    assert len(verified["cv_predictions"]) == 10
    assert verified["provenance"]["model_selection"]["model"] == "random_forest"
    assert verified["provenance"]["source_labels"]["simulation"].startswith("not run")


def test_observed_validator_rejects_corruption_and_stale_config(tmp_path: Path) -> None:
    output = _observed_bundle(tmp_path)
    metrics = output / "observed_metrics.csv"
    metrics.write_text(metrics.read_text(encoding="utf-8") + "corrupt", encoding="utf-8")
    with pytest.raises(ObservedArtifactIntegrityError, match="hash-mismatched"):
        validate_observed_artifacts(tmp_path, output)

    other_root = tmp_path / "stale"
    other_output = _observed_bundle(other_root)
    (other_root / "configs" / "observed_traffic.yaml").write_text(
        "changed: true\n", encoding="utf-8"
    )
    with pytest.raises(ObservedArtifactIntegrityError, match="stale or mismatched"):
        validate_observed_artifacts(other_root, other_output)


def test_observed_mlflow_logs_and_reuses_completed_run(tmp_path: Path) -> None:
    mlflow = pytest.importorskip("mlflow")
    from mlflow.tracking import MlflowClient

    from urbantwin.observed.mlflow_tracking import EXPERIMENT_NAME, track_observed_study
    from urbantwin.traffic.mlflow_tracking import sqlite_tracking_uri

    output = _observed_bundle(tmp_path)
    uri = sqlite_tracking_uri(tmp_path / "tracking" / "mlflow.db")
    first = track_observed_study(output, project_root=tmp_path, tracking_uri=uri, resume=True)
    second = track_observed_study(output, project_root=tmp_path, tracking_uri=uri, resume=True)

    assert first["run_id"] == second["run_id"]
    assert first["already_completed"] is False
    assert second["already_completed"] is True
    client = MlflowClient(tracking_uri=uri)
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    assert experiment is not None
    runs = client.search_runs([experiment.experiment_id])
    assert len(runs) == 1
    run = runs[0]
    assert run.data.params["test_rows"] == "2"
    assert run.data.params["eo_stack_sha256"] == "not_included"
    assert run.data.params["road_manifest_sha256"] == "not_included"
    assert run.data.tags["urbantwin.simulation_status"] == "not_run_scientific_gate_failed"
    assert run.data.tags["urbantwin.eo_role"] == (
        "sentinel_2024_context_worldcover_2021_historic_reference"
    )
    assert run.data.tags["urbantwin.permission_status"] == (
        "author_reported_academic_research_reuse_authorized"
    )
    assert run.data.tags["urbantwin.public_redistribution_status"] == (
        "requires_exact_written_terms"
    )
    assert run.data.params["selected_model"] == "random_forest"
    assert run.data.params["lowest_holdout_point_model"] == (
        "histogram_gradient_boosting"
    )
    assert "random_forest.combined.mae" in run.data.metrics
    artifacts = client.list_artifacts(run.info.run_id, "verified_aggregates")
    assert artifacts
    assert mlflow.get_tracking_uri() == uri
