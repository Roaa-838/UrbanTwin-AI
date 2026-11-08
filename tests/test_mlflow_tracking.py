from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from urbantwin.common import sha256_file
from urbantwin.traffic.mlflow_tracking import (
    EXPERIMENT_NAME,
    MlflowTracker,
    MlflowTrackingError,
    MlflowUnavailableError,
    default_tracking_uri,
    sqlite_tracking_uri,
    track_benchmark,
)


def test_missing_optional_mlflow_dependency_is_reported_lazily(monkeypatch):
    import urbantwin.traffic.mlflow_tracking as tracking

    real_import = tracking.importlib.import_module

    def import_without_mlflow(name: str):
        if name == "mlflow" or name.startswith("mlflow."):
            raise ImportError("simulated optional dependency absence")
        return real_import(name)

    monkeypatch.setattr(tracking.importlib, "import_module", import_without_mlflow)
    with pytest.raises(MlflowUnavailableError, match="optional dependency|not installed"):
        tracking._load_mlflow()


def test_default_tracking_uri_is_local_sqlite(tmp_path: Path):
    uri = default_tracking_uri(tmp_path)
    assert uri.startswith("sqlite:///")
    assert uri.endswith("/.urbantwin/mlflow/mlflow.db")


def test_corrupt_mlflow_resume_state_is_rejected_before_duplicate_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import urbantwin.traffic.mlflow_tracking as tracking

    state = tmp_path / "state.json"
    state.write_text("not-json", encoding="utf-8")

    class FakeClient:
        def __init__(self, tracking_uri: str):
            self.tracking_uri = tracking_uri

        def get_experiment_by_name(self, _name: str):
            return type("Experiment", (), {"experiment_id": "1"})()

    class FakeMlflow:
        @staticmethod
        def set_tracking_uri(_uri: str):
            return None

    monkeypatch.setattr(tracking, "_load_mlflow", lambda: (FakeMlflow(), FakeClient))
    with pytest.raises(MlflowTrackingError, match="unreadable"):
        MlflowTracker.start(
            state_path=state,
            compatibility={"config_sha256": "a" * 64},
            resume=True,
            tracking_uri="sqlite:///ignored.db",
        )


def test_mlflow_parent_children_metrics_artifact_and_resume(tmp_path: Path):
    mlflow = pytest.importorskip("mlflow")
    from mlflow.tracking import MlflowClient

    database = tmp_path / "tracking" / "mlflow.db"
    uri = sqlite_tracking_uri(database)
    state = tmp_path / "checkpoints" / "mlflow_state.json"
    artifact_root = tmp_path / "artifacts"
    compatibility = {
        "config_sha256": "a" * 64,
        "network_sha256": "b" * 64,
        "sumo_version": "SUMO 1.26.0",
    }
    tracker = MlflowTracker.start(
        state_path=state,
        compatibility=compatibility,
        resume=True,
        tracking_uri=uri,
        artifact_root=artifact_root,
        run_name="test benchmark",
        params={"git_commit": "abc123", "evaluation_budget": 18},
        tags={"urbantwin.data_kind": "simulation_abstract"},
    )
    scenario_id = tracker.ensure_scenario(
        "low", params={"training_seeds": [101], "evaluation_seeds": [201]}
    )
    method_id = tracker.ensure_method(
        "low", "GA", params={"signal_plan": [30, 40], "optimizer_seed": 42}
    )
    tracker.log_metrics(
        method_id,
        {
            "mean_time_loss_s_completed": 4.5,
            "completed_vehicles": 10,
            "not_a_metric": "simulator-only",
        },
    )
    artifact = tmp_path / "traffic_summary.csv"
    artifact.write_text("scenario,method\nlow,GA\n", encoding="utf-8")
    tracker.log_artifacts(method_id, [artifact], artifact_path="verified")
    tracker.finish_method("low", "GA")
    tracker.finish_scenario("low")
    tracker.finish()

    payload = json.loads(state.read_text(encoding="utf-8"))
    assert payload["completed"] is True
    assert payload["parent_run_id"] == tracker.parent_run_id
    assert payload["scenario_runs"]["low"] == scenario_id
    assert payload["method_runs"]["low"]["GA"] == method_id

    client = MlflowClient(tracking_uri=uri)
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    assert experiment is not None
    parent = client.get_run(tracker.parent_run_id)
    scenario = client.get_run(scenario_id)
    method = client.get_run(method_id)
    assert parent.data.params["git_commit"] == "abc123"
    assert parent.data.tags["urbantwin.completed"] == "true"
    assert scenario.data.tags["mlflow.parentRunId"] == tracker.parent_run_id
    assert method.data.tags["mlflow.parentRunId"] == scenario_id
    assert method.data.metrics["mean_time_loss_s_completed"] == 4.5
    assert "not_a_metric" not in method.data.metrics
    assert any(
        item.path.rsplit("/", 1)[-1] == "traffic_summary.csv"
        for item in client.list_artifacts(method_id, "verified")
    )

    resumed = MlflowTracker.start(
        state_path=state,
        compatibility=compatibility,
        resume=True,
        tracking_uri=uri,
        artifact_root=artifact_root,
    )
    assert resumed.resumed is True
    assert resumed.completed is True
    assert resumed.parent_run_id == tracker.parent_run_id
    assert resumed.ensure_scenario("low") == scenario_id
    assert resumed.ensure_method("low", "GA") == method_id
    assert resumed.method_completed("low", "GA")
    assert len(mlflow.search_runs([experiment.experiment_id])) == 3

    state.unlink()
    rediscovered = MlflowTracker.start(
        state_path=state,
        compatibility=compatibility,
        resume=True,
        tracking_uri=uri,
        artifact_root=artifact_root,
    )
    assert rediscovered.parent_run_id == tracker.parent_run_id
    assert rediscovered.ensure_scenario("low") == scenario_id
    assert rediscovered.ensure_method("low", "GA") == method_id
    assert len(mlflow.search_runs([experiment.experiment_id])) == 3


def test_mlflow_resume_rejects_incompatible_inputs(tmp_path: Path):
    pytest.importorskip("mlflow")
    uri = sqlite_tracking_uri(tmp_path / "mlflow.db")
    state = tmp_path / "state.json"
    tracker = MlflowTracker.start(
        state_path=state,
        compatibility={"config_sha256": "a" * 64},
        resume=True,
        tracking_uri=uri,
        artifact_root=tmp_path / "artifacts",
    )
    tracker.finish()
    with pytest.raises(MlflowTrackingError, match="incompatible"):
        MlflowTracker.start(
            state_path=state,
            compatibility={"config_sha256": "b" * 64},
            resume=True,
            tracking_uri=uri,
            artifact_root=tmp_path / "artifacts",
        )


def test_track_verified_benchmark_hierarchy_artifacts_and_completed_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    pytest.importorskip("mlflow")
    from mlflow.tracking import MlflowClient

    import urbantwin.traffic.mlflow_tracking as tracking

    monkeypatch.setattr(tracking, "ROOT", tmp_path)
    output = tmp_path / "results" / "traffic"
    output.mkdir(parents=True)
    config = tmp_path / "configs" / "experiments.yaml"
    config.parent.mkdir()
    config.write_text("random_seed: 42\ntraffic:\n  track_emissions: true\n", encoding="utf-8")
    config_hash = sha256_file(config)
    csv_names = (
        "traffic_metrics.csv",
        "traffic_summary.csv",
        "traffic_paired_deltas.csv",
        "traffic_paired_summary.csv",
    )
    for name in csv_names:
        (output / name).write_text("verified,test\n1,1\n", encoding="utf-8")
    figure = output / "traffic_time_loss.png"
    figure.write_bytes(b"verified-figure")

    methods = ["fixed", "GA", "PSO", "random_search"]
    metrics = pd.DataFrame(
        [
            {
                "scenario": "low",
                "method": method,
                "seed": 101,
                "data_kind": "simulation_abstract",
                "green_ns_s": 30 + index,
                "green_ew_s": 40 + index,
                "optimization_wall_time_s": float(index),
            }
            for index, method in enumerate(methods)
        ]
    )
    summary = pd.DataFrame(
        [
            {
                "scenario": "low",
                "method": method,
                "data_kind": "simulation_abstract",
                "mean_time_loss_s_completed": 10.0 + index,
                "mean_waiting_time_s_completed": 5.0 + index,
                "mean_approach_queue_vehicles": 2.0 + index,
                "mean_completed_throughput_vph": 100.0 + index,
                "mean_generated_vehicles": 20.0,
                "mean_inserted_vehicles": 19.0,
                "mean_completed_vehicles": 18.0,
                "mean_unfinished_vehicles": 1.0,
                "mean_pending_vehicles": 1.0,
                "mean_simulated_co2_g_inserted_fleet": 42.0 + index,
                "runs": 1,
            }
            for index, method in enumerate(methods)
        ]
    )
    paired_summary = pd.DataFrame(
        [
            {
                "scenario": "low",
                "method": method,
                "data_kind": "simulation_abstract",
                "runs": 1,
                "mean_delta_mean_time_loss_s_completed_method_minus_fixed": float(index),
            }
            for index, method in enumerate(methods[1:], start=1)
        ]
    )
    optimization = {
        "low": {
            "training_seeds": [17],
            "evaluation_seeds": [101],
            "objective_evaluation_budget_per_method": 4,
            "GA": {"optimizer_seed": 42, "objective_evaluations": 4},
            "PSO": {"optimizer_seed": 43, "objective_evaluations": 4},
            "random_search": {"optimizer_seed": 44, "objective_evaluations": 4},
        }
    }
    provenance = {
        "schema_version": "urbantwin.traffic-benchmark/v2",
        "data_kind": "simulation_abstract",
        "generated_utc": "2026-11-07T00:00:00+00:00",
        "config_sha256": config_hash,
        "network_sha256": "b" * 64,
        "network_source": "generated_abstract_junction",
        "sumo_version": "Eclipse SUMO 1.26.0",
        "optimizer_source_sha256": "e" * 64,
        "equal_objective_evaluation_budget": 4,
        "track_emissions": True,
        "emission_class_assumptions": ["HBEFA3/PC_G_EU4"],
        "co2_accounting": "SUMO model integration",
        "optimization": optimization,
        "optimization_runtime_interpretation": (
            "Current invocation runtime; cache hits reduce resumed wall time."
        ),
        "route_artifacts_by_scenario_seed": {
            "low__17": {"path": "redacted", "sha256": "c" * 64},
            "low__101": {"path": "redacted", "sha256": "d" * 64},
        },
    }
    verified = {
        "metrics": metrics,
        "summary": summary,
        "paired_deltas": pd.DataFrame(),
        "paired_summary": paired_summary,
        "provenance": provenance,
        "hashes": {
            "config": config_hash,
            "network": "b" * 64,
            "routes": {"low__17": "c" * 64, "low__101": "d" * 64},
        },
    }
    calls = []

    def validate(root: Path, results_dir: Path):
        calls.append((root, results_dir))
        return verified

    monkeypatch.setattr(tracking, "validate_traffic_artifacts", validate)
    uri = sqlite_tracking_uri(tmp_path / ".urbantwin" / "mlflow" / "mlflow.db")
    first = track_benchmark(
        output,
        config_path=config,
        tracking_uri=uri,
        resume=True,
        figure_paths=[figure],
    )
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    second = track_benchmark(
        output,
        config_path=config,
        resume=True,
        figure_paths=[figure],
    )
    assert len(calls) == 2
    assert first["already_completed"] is False
    assert second["already_completed"] is True
    assert second["resumed"] is True
    assert second["run_id"] == first["run_id"]

    client = MlflowClient(tracking_uri=uri)
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    runs = client.search_runs([experiment.experiment_id])
    assert len(runs) == 6  # benchmark + scenario + four methods
    parent = client.get_run(first["run_id"])
    assert parent.data.params["sumo_version"] == "Eclipse SUMO 1.26.0"
    assert parent.data.params["optimizer_source_sha256"] == "e" * 64
    assert parent.data.tags["urbantwin.co2_source"] == "simulation_only_not_observed"
    artifacts = client.list_artifacts(first["run_id"], "verified")
    artifact_names = {artifact.path.rsplit("/", 1)[-1] for artifact in artifacts}
    assert set(csv_names) <= artifact_names
    assert config.name in artifact_names
    assert figure.name in artifact_names
    assert any(name.startswith("safe_traffic_provenance_") for name in artifact_names)

    ga_run = next(run for run in runs if run.data.tags.get("urbantwin.method") == "GA")
    assert ga_run.data.params["optimizer_seed"] == "42"
    assert ga_run.data.params["training_seeds"] == "[17]"
    assert ga_run.data.metrics["completed_trip_time_loss_s"] == 11.0
    assert ga_run.data.metrics["optimizer_runtime_s"] == 1.0
    assert (
        ga_run.data.metrics[
            "paired_vs_fixed.mean_time_loss_s_completed.method_minus_fixed"
        ]
        == 1.0
    )
