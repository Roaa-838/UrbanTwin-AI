"""Optional, resumable MLflow tracking for traffic benchmarks.

The local SUMO checkpoint store is authoritative.  This module deliberately
contains no simulation or cache logic: MLflow can be unavailable without
preventing a benchmark from running or resuming from its local checkpoints.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import subprocess
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from urbantwin.common import ROOT, sha256_file
from urbantwin.visualization.report_writer import validate_traffic_artifacts

EXPERIMENT_NAME = "UrbanTwin-Traffic-Optimization"
STATE_SCHEMA = "urbantwin.mlflow-run-state/v1"
_COMPATIBILITY_TAG = "urbantwin.compatibility_sha256"
_COMPLETED_TAG = "urbantwin.completed"
_PARENT_TAG = "mlflow.parentRunId"
_RUN_NAME_TAG = "mlflow.runName"


class MlflowTrackingError(RuntimeError):
    """An explicitly reported MLflow setup, validation, or logging failure."""


class MlflowUnavailableError(MlflowTrackingError):
    """Raised only when tracking was requested but MLflow is not installed."""


def sqlite_tracking_uri(database: str | Path) -> str:
    """Return a SQLAlchemy SQLite URI that is also valid on Windows."""
    path = Path(database).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{path.as_posix()}"


def default_tracking_uri(project_root: str | Path = ROOT) -> str:
    """Default to a repository-local, ignored SQLite tracking database."""
    return sqlite_tracking_uri(Path(project_root) / ".urbantwin" / "mlflow" / "mlflow.db")


def default_artifact_root(project_root: str | Path = ROOT) -> Path:
    """Return the default local MLflow artifact directory."""
    return Path(project_root).resolve() / ".urbantwin" / "mlflow" / "artifacts"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def compatibility_sha256(compatibility: Mapping[str, Any]) -> str:
    """Hash caller-supplied inputs that identify one compatible benchmark."""
    return hashlib.sha256(_canonical_json(dict(compatibility)).encode("utf-8")).hexdigest()


def _string_value(value: Any) -> str:
    if isinstance(value, (dict, list, tuple)):
        return _canonical_json(value)
    if value is None:
        return "null"
    return str(value)


def _strings(values: Mapping[str, Any] | None) -> dict[str, str]:
    return {str(key): _string_value(value) for key, value in (values or {}).items()}


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        if os.name != "nt":
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_mlflow() -> tuple[Any, Any]:
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    try:
        mlflow = importlib.import_module("mlflow")
        tracking = importlib.import_module("mlflow.tracking")
    except ImportError as exc:
        raise MlflowUnavailableError(
            "MLflow tracking was requested but MLflow is not installed. "
            "Install it with `python -m pip install -e \".[mlflow]\"`."
        ) from exc
    return mlflow, tracking.MlflowClient


def _run_status(run: Any) -> str:
    return str(run.info.status).upper()


def _is_complete(run: Any) -> bool:
    return _run_status(run) == "FINISHED" and run.data.tags.get(_COMPLETED_TAG) == "true"


@dataclass
class MlflowTracker:
    """One logical benchmark and its scenario/method child-run hierarchy."""

    client: Any
    mlflow: Any
    state_path: Path
    tracking_uri: str
    experiment_name: str
    experiment_id: str
    compatibility_hash: str
    parent_run_id: str
    scenario_runs: dict[str, str] = field(default_factory=dict)
    method_runs: dict[str, dict[str, str]] = field(default_factory=dict)
    completed: bool = False
    resumed: bool = False

    @classmethod
    def start(
        cls,
        *,
        state_path: str | Path,
        compatibility: Mapping[str, Any],
        resume: bool,
        tracking_uri: str | None = None,
        experiment_name: str = EXPERIMENT_NAME,
        artifact_root: str | Path | None = None,
        run_name: str = "UrbanTwin traffic benchmark",
        params: Mapping[str, Any] | None = None,
        tags: Mapping[str, Any] | None = None,
    ) -> MlflowTracker:
        """Create or resume a benchmark run after validating local association state.

        A valid completed run is associated without being reopened.  Corrupt or
        incompatible state raises instead of silently creating a duplicate run.
        """
        mlflow, client_type = _load_mlflow()
        uri = tracking_uri or default_tracking_uri()
        state = Path(state_path)
        compatibility_hash = compatibility_sha256(compatibility)
        try:
            mlflow.set_tracking_uri(uri)
            client = client_type(tracking_uri=uri)
            experiment = client.get_experiment_by_name(experiment_name)
            if experiment is None:
                create_kwargs: dict[str, str] = {}
                selected_artifact_root = artifact_root
                if selected_artifact_root is None and uri.lower().startswith("sqlite:"):
                    selected_artifact_root = default_artifact_root()
                if selected_artifact_root is not None:
                    artifacts = Path(selected_artifact_root).resolve()
                    artifacts.mkdir(parents=True, exist_ok=True)
                    create_kwargs["artifact_location"] = artifacts.as_uri()
                experiment_id = str(client.create_experiment(experiment_name, **create_kwargs))
            else:
                experiment_id = str(experiment.experiment_id)

            if resume and not state.exists():
                compatible_runs = list(
                    client.search_runs(
                        [experiment_id],
                        filter_string=(
                            f"tags.`{_COMPATIBILITY_TAG}` = '{compatibility_hash}'"
                        ),
                    )
                )
                parents = [
                    run
                    for run in compatible_runs
                    if run.data.tags.get("urbantwin.run_level") == "benchmark"
                ]
                if len(parents) > 1:
                    raise MlflowTrackingError(
                        "MLflow contains multiple compatible benchmark parents; refusing "
                        "to guess a resume association."
                    )
                if parents:
                    parent = parents[0]
                    parent_id = str(parent.info.run_id)
                    scenarios: dict[str, str] = {}
                    scenario_by_id: dict[str, str] = {}
                    for run in compatible_runs:
                        tags_for_run = run.data.tags
                        if (
                            tags_for_run.get("urbantwin.run_level") == "scenario"
                            and tags_for_run.get(_PARENT_TAG) == parent_id
                        ):
                            scenario = tags_for_run.get("urbantwin.scenario")
                            if not scenario or scenario in scenarios:
                                raise MlflowTrackingError(
                                    "MLflow has invalid or duplicate compatible scenario runs."
                                )
                            run_id = str(run.info.run_id)
                            scenarios[scenario] = run_id
                            scenario_by_id[run_id] = scenario
                    methods: dict[str, dict[str, str]] = {
                        scenario: {} for scenario in scenarios
                    }
                    for run in compatible_runs:
                        tags_for_run = run.data.tags
                        parent_for_run = tags_for_run.get(_PARENT_TAG)
                        if (
                            tags_for_run.get("urbantwin.run_level") == "method"
                            and parent_for_run in scenario_by_id
                        ):
                            scenario = scenario_by_id[parent_for_run]
                            method = tags_for_run.get("urbantwin.method")
                            if not method or method in methods[scenario]:
                                raise MlflowTrackingError(
                                    "MLflow has invalid or duplicate compatible method runs."
                                )
                            methods[scenario][method] = str(run.info.run_id)
                    tracker = cls(
                        client=client,
                        mlflow=mlflow,
                        state_path=state,
                        tracking_uri=uri,
                        experiment_name=experiment_name,
                        experiment_id=experiment_id,
                        compatibility_hash=compatibility_hash,
                        parent_run_id=parent_id,
                        scenario_runs=scenarios,
                        method_runs=methods,
                        completed=_is_complete(parent),
                        resumed=True,
                    )
                    tracker._validate_associated_children()
                    if not tracker.completed:
                        tracker._reactivate(parent_id)
                    tracker._save_state()
                    return tracker

            if resume and state.exists():
                payload = cls._read_state(state)
                cls._validate_state(
                    payload,
                    tracking_uri=uri,
                    experiment_name=experiment_name,
                    experiment_id=experiment_id,
                    compatibility_hash=compatibility_hash,
                )
                parent_id = str(payload["parent_run_id"])
                parent = client.get_run(parent_id)
                if str(parent.info.experiment_id) != experiment_id:
                    raise MlflowTrackingError(
                        "Stored MLflow parent run belongs to a different experiment."
                    )
                if parent.data.tags.get(_COMPATIBILITY_TAG) != compatibility_hash:
                    raise MlflowTrackingError(
                        "Stored MLflow parent run is incompatible with the benchmark inputs."
                    )
                tracker = cls(
                    client=client,
                    mlflow=mlflow,
                    state_path=state,
                    tracking_uri=uri,
                    experiment_name=experiment_name,
                    experiment_id=experiment_id,
                    compatibility_hash=compatibility_hash,
                    parent_run_id=parent_id,
                    scenario_runs={
                        str(key): str(value)
                        for key, value in payload["scenario_runs"].items()
                    },
                    method_runs={
                        str(scenario): {
                            str(method): str(run_id) for method, run_id in methods.items()
                        }
                        for scenario, methods in payload["method_runs"].items()
                    },
                    completed=_is_complete(parent),
                    resumed=True,
                )
                tracker._validate_associated_children()
                if not tracker.completed:
                    tracker._reactivate(parent_id)
                    client.set_tag(parent_id, "urbantwin.interrupted", "false")
                tracker._save_state()
                return tracker

            parent_tags = {
                _RUN_NAME_TAG: run_name,
                _COMPATIBILITY_TAG: compatibility_hash,
                _COMPLETED_TAG: "false",
                "urbantwin.run_level": "benchmark",
                **_strings(tags),
            }
            parent = client.create_run(experiment_id, tags=parent_tags)
            tracker = cls(
                client=client,
                mlflow=mlflow,
                state_path=state,
                tracking_uri=uri,
                experiment_name=experiment_name,
                experiment_id=experiment_id,
                compatibility_hash=compatibility_hash,
                parent_run_id=str(parent.info.run_id),
            )
            tracker.log_params(
                tracker.parent_run_id,
                {"compatibility_sha256": compatibility_hash, **dict(params or {})},
            )
            tracker._save_state()
            return tracker
        except MlflowTrackingError:
            raise
        except Exception as exc:
            raise MlflowTrackingError(f"Could not initialize MLflow tracking: {exc}") from exc

    @staticmethod
    def _read_state(path: Path) -> dict[str, Any]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise MlflowTrackingError(f"MLflow resume state is unreadable: {path}: {exc}") from exc
        if not isinstance(payload, dict):
            raise MlflowTrackingError(f"MLflow resume state must be a JSON object: {path}")
        return payload

    @staticmethod
    def _validate_state(
        payload: Mapping[str, Any],
        *,
        tracking_uri: str,
        experiment_name: str,
        experiment_id: str,
        compatibility_hash: str,
    ) -> None:
        required = {
            "schema_version",
            "tracking_uri_sha256",
            "experiment_name",
            "experiment_id",
            "compatibility_sha256",
            "parent_run_id",
            "scenario_runs",
            "method_runs",
            "completed",
        }
        missing = sorted(required - payload.keys())
        if missing:
            raise MlflowTrackingError(
                f"MLflow resume state is incomplete; missing: {', '.join(missing)}"
            )
        if payload["schema_version"] != STATE_SCHEMA:
            raise MlflowTrackingError("MLflow resume state has an unsupported schema version.")
        uri_hash = hashlib.sha256(tracking_uri.encode("utf-8")).hexdigest()
        expected = {
            "tracking_uri_sha256": uri_hash,
            "experiment_name": experiment_name,
            "experiment_id": experiment_id,
            "compatibility_sha256": compatibility_hash,
        }
        mismatched = [key for key, value in expected.items() if payload.get(key) != value]
        if mismatched:
            raise MlflowTrackingError(
                "MLflow resume state is incompatible with the requested benchmark: "
                + ", ".join(mismatched)
            )
        if not isinstance(payload["scenario_runs"], dict) or not isinstance(
            payload["method_runs"], dict
        ):
            raise MlflowTrackingError("MLflow resume state has invalid child-run mappings.")

    def _state_payload(self) -> dict[str, Any]:
        return {
            "schema_version": STATE_SCHEMA,
            "tracking_uri_sha256": hashlib.sha256(
                self.tracking_uri.encode("utf-8")
            ).hexdigest(),
            "experiment_name": self.experiment_name,
            "experiment_id": self.experiment_id,
            "compatibility_sha256": self.compatibility_hash,
            "parent_run_id": self.parent_run_id,
            "scenario_runs": self.scenario_runs,
            "method_runs": self.method_runs,
            "completed": self.completed,
        }

    def _save_state(self) -> None:
        _atomic_json(self._state_payload(), self.state_path)

    def _reactivate(self, run_id: str) -> None:
        run = self.client.get_run(run_id)
        if _run_status(run) != "RUNNING":
            self.client.update_run(run_id, status="RUNNING")
        self.client.set_tag(run_id, _COMPLETED_TAG, "false")

    def _validate_associated_children(self) -> None:
        for scenario, run_id in self.scenario_runs.items():
            self._validate_child(run_id, self.parent_run_id, "scenario", scenario)
        for scenario, methods in self.method_runs.items():
            if scenario not in self.scenario_runs:
                raise MlflowTrackingError(
                    f"MLflow state has methods without a scenario run: {scenario}"
                )
            for method, run_id in methods.items():
                self._validate_child(
                    run_id,
                    self.scenario_runs[scenario],
                    "method",
                    f"{scenario}/{method}",
                )

    def _validate_child(self, run_id: str, parent_id: str, level: str, identity: str) -> Any:
        try:
            run = self.client.get_run(run_id)
        except Exception as exc:
            raise MlflowTrackingError(
                f"Stored MLflow {level} run is unavailable for {identity}: {exc}"
            ) from exc
        tags = run.data.tags
        if str(run.info.experiment_id) != self.experiment_id:
            raise MlflowTrackingError(f"Stored MLflow {level} run is in another experiment.")
        if tags.get(_PARENT_TAG) != parent_id:
            raise MlflowTrackingError(f"Stored MLflow {level} run has the wrong parent.")
        if tags.get(_COMPATIBILITY_TAG) != self.compatibility_hash:
            raise MlflowTrackingError(f"Stored MLflow {level} run is incompatible.")
        if tags.get("urbantwin.run_level") != level:
            raise MlflowTrackingError(f"Stored MLflow {level} run has invalid hierarchy tags.")
        return run

    def _create_child(
        self,
        *,
        parent_id: str,
        level: str,
        run_name: str,
        tags: Mapping[str, Any] | None,
    ) -> str:
        child = self.client.create_run(
            self.experiment_id,
            tags={
                _RUN_NAME_TAG: run_name,
                _PARENT_TAG: parent_id,
                _COMPATIBILITY_TAG: self.compatibility_hash,
                _COMPLETED_TAG: "false",
                "urbantwin.run_level": level,
                **_strings(tags),
            },
        )
        return str(child.info.run_id)

    def ensure_scenario(
        self,
        scenario: str,
        *,
        params: Mapping[str, Any] | None = None,
        tags: Mapping[str, Any] | None = None,
    ) -> str:
        """Return one stable child run for a scenario."""
        key = str(scenario)
        if key in self.scenario_runs:
            run_id = self.scenario_runs[key]
            run = self._validate_child(run_id, self.parent_run_id, "scenario", key)
            if not _is_complete(run):
                self._reactivate(run_id)
            return run_id
        if self.completed:
            raise MlflowTrackingError("Cannot add a scenario to a completed benchmark run.")
        run_id = self._create_child(
            parent_id=self.parent_run_id,
            level="scenario",
            run_name=f"scenario:{key}",
            tags={"urbantwin.scenario": key, **dict(tags or {})},
        )
        self.scenario_runs[key] = run_id
        self.method_runs.setdefault(key, {})
        self.log_params(run_id, params or {})
        self._save_state()
        return run_id

    def ensure_method(
        self,
        scenario: str,
        method: str,
        *,
        params: Mapping[str, Any] | None = None,
        tags: Mapping[str, Any] | None = None,
    ) -> str:
        """Return one stable method child beneath its scenario run."""
        scenario_key = str(scenario)
        method_key = str(method)
        scenario_id = self.ensure_scenario(scenario_key)
        methods = self.method_runs.setdefault(scenario_key, {})
        if method_key in methods:
            run_id = methods[method_key]
            run = self._validate_child(
                run_id,
                scenario_id,
                "method",
                f"{scenario_key}/{method_key}",
            )
            if not _is_complete(run):
                self._reactivate(run_id)
            return run_id
        if self.completed:
            raise MlflowTrackingError("Cannot add a method to a completed benchmark run.")
        run_id = self._create_child(
            parent_id=scenario_id,
            level="method",
            run_name=f"{scenario_key}:{method_key}",
            tags={
                "urbantwin.scenario": scenario_key,
                "urbantwin.method": method_key,
                **dict(tags or {}),
            },
        )
        methods[method_key] = run_id
        self.log_params(run_id, params or {})
        self._save_state()
        return run_id

    def log_params(self, run_id: str, params: Mapping[str, Any]) -> None:
        """Log immutable parameters, validating resumed values before skipping them."""
        if not params:
            return
        try:
            existing = self.client.get_run(run_id).data.params
            for key, value in _strings(params).items():
                if key in existing and existing[key] != value:
                    raise MlflowTrackingError(
                        f"MLflow parameter {key!r} changed on resumed run {run_id}."
                    )
                if key not in existing:
                    self.client.log_param(run_id, key, value)
        except MlflowTrackingError:
            raise
        except Exception as exc:
            raise MlflowTrackingError(f"Could not log MLflow parameters: {exc}") from exc

    def log_metrics(
        self,
        run_id: str,
        metrics: Mapping[str, Any],
        *,
        step: int = 0,
    ) -> None:
        """Log finite numeric metrics only; labels and assumptions belong in tags."""
        try:
            for key, value in metrics.items():
                if value is None or isinstance(value, bool):
                    continue
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(numeric):
                    self.client.log_metric(run_id, str(key), numeric, step=int(step))
        except Exception as exc:
            raise MlflowTrackingError(f"Could not log MLflow metrics: {exc}") from exc

    def log_artifacts(
        self,
        run_id: str,
        paths: Sequence[str | Path],
        *,
        artifact_path: str | None = None,
    ) -> None:
        """Log caller-validated files; directories and missing paths are rejected."""
        try:
            for value in paths:
                path = Path(value)
                if not path.is_file():
                    raise MlflowTrackingError(f"MLflow artifact is not a file: {path}")
                self.client.log_artifact(run_id, str(path.resolve()), artifact_path)
        except MlflowTrackingError:
            raise
        except Exception as exc:
            raise MlflowTrackingError(f"Could not log MLflow artifacts: {exc}") from exc

    def method_completed(self, scenario: str, method: str) -> bool:
        run_id = self.method_runs.get(str(scenario), {}).get(str(method))
        return bool(run_id and _is_complete(self.client.get_run(run_id)))

    def scenario_completed(self, scenario: str) -> bool:
        run_id = self.scenario_runs.get(str(scenario))
        return bool(run_id and _is_complete(self.client.get_run(run_id)))

    def finish_method(self, scenario: str, method: str) -> None:
        run_id = self.method_runs[str(scenario)][str(method)]
        if not _is_complete(self.client.get_run(run_id)):
            self.client.set_tag(run_id, _COMPLETED_TAG, "true")
            self.client.set_terminated(run_id, status="FINISHED")

    def finish_scenario(self, scenario: str) -> None:
        run_id = self.scenario_runs[str(scenario)]
        if not _is_complete(self.client.get_run(run_id)):
            self.client.set_tag(run_id, _COMPLETED_TAG, "true")
            self.client.set_terminated(run_id, status="FINISHED")

    def mark_interrupted(self) -> None:
        """Record interruption without terminating the resumable parent run."""
        if not self.completed:
            self.client.set_tag(self.parent_run_id, "urbantwin.interrupted", "true")
            self._save_state()

    def finish(self) -> None:
        """Mark the parent complete only after all local artifacts were validated/logged."""
        if not self.completed:
            self.client.set_tag(self.parent_run_id, _COMPLETED_TAG, "true")
            self.client.set_tag(self.parent_run_id, "urbantwin.interrupted", "false")
            self.client.set_terminated(self.parent_run_id, status="FINISHED")
            self.completed = True
            self._save_state()


_AGGREGATE_METRICS = {
    "mean_time_loss_s_completed": "completed_trip_time_loss_s",
    "mean_waiting_time_s_completed": "completed_trip_waiting_time_s",
    "mean_approach_queue_vehicles": "mean_approach_queue_vehicles",
    "mean_completed_throughput_vph": "completed_throughput_vehicles_per_sim_hour",
    "mean_generated_vehicles": "generated_vehicles",
    "mean_inserted_vehicles": "inserted_vehicles",
    "mean_completed_vehicles": "completed_vehicles",
    "mean_unfinished_vehicles": "unfinished_vehicles",
    "mean_pending_vehicles": "pending_vehicles",
    "mean_simulated_co2_g_inserted_fleet": "modeled_co2_g_inserted_fleet",
    "runs": "evaluation_runs",
}


def _git_commit() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise MlflowTrackingError(f"Could not determine the Git commit for MLflow: {exc}") from exc
    commit = completed.stdout.strip()
    if len(commit) != 40:
        raise MlflowTrackingError("Git did not return a full 40-character commit hash.")
    return commit


def _git_state() -> dict[str, Any]:
    try:
        commit = _git_commit()
        status = subprocess.run(
            ["git", "status", "--porcelain=v1"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (MlflowTrackingError, OSError, subprocess.CalledProcessError):
        return {"commit": "unavailable", "dirty": "unknown"}
    return {"commit": commit, "dirty": bool(status.stdout.strip())}


def _require_provenance_value(provenance: Mapping[str, Any], key: str) -> Any:
    value = provenance.get(key)
    if value is None or value == "" or value == [] or value == {}:
        raise MlflowTrackingError(f"Verified traffic provenance is missing {key!r}.")
    return value


def _safe_provenance(provenance: Mapping[str, Any]) -> dict[str, Any]:
    """Create a path-redacted provenance snapshot suitable for MLflow artifacts."""
    allowed = {
        "schema_version",
        "data_kind",
        "generated_utc",
        "config_sha256",
        "network_sha256",
        "network_source",
        "warning",
        "objective",
        "equal_objective_evaluation_budget",
        "optimization",
        "evaluation_signal_programs",
        "track_emissions",
        "emission_class_assumptions",
        "co2_accounting",
        "co2_interpretation",
        "sumo_version",
        "optimizer_source_sha256",
        "checkpointing",
        "optimization_runtime_interpretation",
        "metrics_sha256",
        "summary_sha256",
        "paired_deltas_sha256",
        "paired_summary_sha256",
    }
    safe = {key: provenance[key] for key in sorted(allowed) if key in provenance}
    if isinstance(safe.get("checkpointing"), Mapping):
        safe["checkpointing"] = {
            key: value
            for key, value in safe["checkpointing"].items()
            if key != "checkpoint_root"
        }
        safe["checkpointing"]["checkpoint_root_redacted"] = True
    routes = provenance.get("route_artifacts_by_scenario_seed")
    if isinstance(routes, Mapping):
        safe["route_hashes_by_scenario_seed"] = {
            str(key): record.get("sha256")
            for key, record in sorted(routes.items())
            if isinstance(record, Mapping)
        }
    safe["path_fields_redacted"] = True
    safe["co2_source_label"] = "SUMO model estimate; simulator-only, not observed data"
    return safe


def _write_safe_provenance(
    provenance: Mapping[str, Any], association_dir: Path, fingerprint: str
) -> Path:
    path = association_dir / f"safe_traffic_provenance_{fingerprint}.json"
    _atomic_json(_safe_provenance(provenance), path)
    return path


def _method_signal_plan(metrics: pd.DataFrame, scenario: str, method: str) -> list[int]:
    rows = metrics.loc[
        (metrics["scenario"].astype(str) == scenario)
        & (metrics["method"].astype(str) == method)
    ]
    required = {"green_ns_s", "green_ew_s"}
    missing = sorted(required - set(rows.columns))
    if missing:
        raise MlflowTrackingError(
            "Verified traffic metrics cannot identify the signal plan; missing: "
            + ", ".join(missing)
        )
    plans = {
        (int(row.green_ns_s), int(row.green_ew_s))
        for row in rows[["green_ns_s", "green_ew_s"]].itertuples(index=False)
    }
    if len(plans) != 1:
        raise MlflowTrackingError(
            f"Traffic signal plan is not unique for {scenario}/{method}: {sorted(plans)}"
        )
    return list(next(iter(plans)))


def _optimizer_runtime(metrics: pd.DataFrame, scenario: str, method: str) -> float:
    if "optimization_wall_time_s" not in metrics.columns:
        raise MlflowTrackingError("Verified traffic metrics lack optimizer runtime.")
    values = pd.to_numeric(
        metrics.loc[
            (metrics["scenario"].astype(str) == scenario)
            & (metrics["method"].astype(str) == method),
            "optimization_wall_time_s",
        ],
        errors="coerce",
    ).dropna()
    if values.empty or not math.isfinite(float(values.iloc[0])):
        raise MlflowTrackingError(f"Optimizer runtime is invalid for {scenario}/{method}.")
    if not (values - values.iloc[0]).abs().le(1e-9).all():
        raise MlflowTrackingError(f"Optimizer runtime is inconsistent for {scenario}/{method}.")
    return float(values.iloc[0])


def _paired_metrics(row: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in row.items():
        if not str(key).startswith("mean_delta_"):
            continue
        metric = str(key)[len("mean_delta_") :]
        metric = metric.removesuffix("_method_minus_fixed")
        result[f"paired_vs_fixed.{metric}.method_minus_fixed"] = value
    return result


def track_benchmark(
    output_dir: str | Path,
    *,
    config_path: str | Path,
    tracking_uri: str | None = None,
    resume: bool = False,
    figure_paths: Sequence[str | Path] = (),
) -> dict[str, Any]:
    """Validate and log one completed local benchmark to optional MLflow.

    The strict local artifact validator runs before MLflow is imported or any
    tracking run is created.  Consequently, MLflow never upgrades an invalid or
    incomplete local result bundle into an apparently successful experiment.
    """
    output = Path(output_dir)
    if not output.is_absolute():
        output = (ROOT / output).resolve()
    configuration = Path(config_path)
    if not configuration.is_absolute():
        configuration = (ROOT / configuration).resolve()
    if not configuration.is_file():
        raise MlflowTrackingError(f"MLflow configuration snapshot is missing: {configuration}")

    try:
        verified = validate_traffic_artifacts(ROOT, output)
    except Exception as exc:
        raise MlflowTrackingError(
            f"Traffic artifacts failed strict validation before MLflow logging: {exc}"
        ) from exc
    if verified is None:
        raise MlflowTrackingError(f"No completed traffic benchmark exists in {output}.")

    provenance = verified["provenance"]
    config_hash = sha256_file(configuration)
    if config_hash != provenance.get("config_sha256") or config_hash != verified["hashes"][
        "config"
    ]:
        raise MlflowTrackingError(
            "The requested MLflow configuration snapshot does not match traffic provenance."
        )
    sumo_version = str(_require_provenance_value(provenance, "sumo_version"))
    source = str(_require_provenance_value(provenance, "data_kind"))
    budget = int(_require_provenance_value(provenance, "equal_objective_evaluation_budget"))
    network_hash = str(_require_provenance_value(provenance, "network_sha256"))
    optimizer_source_hash = str(
        _require_provenance_value(provenance, "optimizer_source_sha256")
    )
    git_state = _git_state()
    checkpointing = provenance.get("checkpointing")
    checkpoint_compatibility = {}
    if isinstance(checkpointing, Mapping):
        checkpoint_compatibility = {
            key: checkpointing.get(key)
            for key in (
                "schema_version",
                "runner_contract",
                "runner_source_sha256",
                "signal_adapter_sha256",
            )
        }
    route_hashes = {
        str(key): str(value) for key, value in sorted(verified["hashes"]["routes"].items())
    }
    compatibility = {
        "config_sha256": config_hash,
        "network_sha256": network_hash,
        "route_hashes_by_scenario_seed": route_hashes,
        "sumo_version": sumo_version,
        "data_kind": source,
        "objective_evaluation_budget": budget,
        "optimizer_source_sha256": optimizer_source_hash,
        "track_emissions": provenance.get("track_emissions") is True,
        "emission_class_assumptions": provenance.get("emission_class_assumptions", []),
        "co2_accounting": provenance.get("co2_accounting"),
        "git": git_state,
        "simulation_runner": checkpoint_compatibility,
    }
    fingerprint = compatibility_sha256(compatibility)
    uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI") or default_tracking_uri(ROOT)
    uri_fingerprint = hashlib.sha256(uri.encode("utf-8")).hexdigest()[:16]
    association_dir = ROOT / ".urbantwin" / "mlflow" / "run_associations"
    state_path = association_dir / f"{fingerprint}_{uri_fingerprint}.json"
    tracker: MlflowTracker | None = None
    try:
        tracker = MlflowTracker.start(
            state_path=state_path,
            compatibility=compatibility,
            resume=resume,
            tracking_uri=uri,
            artifact_root=(
                default_artifact_root(ROOT) if uri.lower().startswith("sqlite:") else None
            ),
            run_name=f"traffic-benchmark:{output.name}",
            params={
                "config_sha256": config_hash,
                "git_commit": git_state["commit"],
                "git_dirty": git_state["dirty"],
                "sumo_version": sumo_version,
                "data_kind": source,
                "objective_evaluation_budget_per_method": budget,
                "network_sha256": network_hash,
                "optimizer_source_sha256": optimizer_source_hash,
                "route_hashes_by_scenario_seed": route_hashes,
                "track_emissions": provenance.get("track_emissions") is True,
                "emission_class_assumptions": provenance.get(
                    "emission_class_assumptions", []
                ),
                "optimizer_runtime_interpretation": provenance.get(
                    "optimization_runtime_interpretation"
                ),
            },
            tags={
                "urbantwin.data_kind": source,
                "urbantwin.co2_source": "simulation_only_not_observed",
                "urbantwin.authoritative_store": "validated_local_artifacts",
                "urbantwin.git_dirty": git_state["dirty"],
                "urbantwin.optimizer_runtime_scope": "current_invocation_cache_sensitive",
            },
        )
        if tracker.completed:
            return {
                "run_id": tracker.parent_run_id,
                "tracking_uri": uri,
                "resumed": tracker.resumed,
                "already_completed": True,
            }

        metrics: pd.DataFrame = verified["metrics"]
        summary: pd.DataFrame = verified["summary"]
        paired_summary: pd.DataFrame = verified["paired_summary"]
        required_summary = set(_AGGREGATE_METRICS) | {"scenario", "method", "data_kind"}
        missing = sorted(required_summary - set(summary.columns))
        if missing:
            raise MlflowTrackingError(
                "Verified traffic summary lacks required MLflow metrics: " + ", ".join(missing)
            )

        optimization = provenance.get("optimization")
        if not isinstance(optimization, Mapping):
            raise MlflowTrackingError("Traffic provenance lacks optimizer seed metadata.")
        for scenario, scenario_rows in summary.groupby("scenario", sort=True):
            scenario_key = str(scenario)
            scenario_optimization = optimization.get(scenario_key)
            if not isinstance(scenario_optimization, Mapping):
                raise MlflowTrackingError(
                    f"Traffic provenance lacks optimization metadata for {scenario_key}."
                )
            training_seeds = scenario_optimization.get("training_seeds")
            evaluation_seeds = scenario_optimization.get("evaluation_seeds")
            scenario_was_complete = tracker.scenario_completed(scenario_key)
            scenario_route_hashes = {
                key: value
                for key, value in route_hashes.items()
                if key.startswith(f"{scenario_key}__")
            }
            scenario_id = tracker.ensure_scenario(
                scenario_key,
                params={
                    "training_seeds": training_seeds,
                    "evaluation_seeds": evaluation_seeds,
                    "objective_evaluation_budget_per_method": budget,
                    "route_hashes_by_seed": scenario_route_hashes,
                },
                tags={"urbantwin.data_kind": source},
            )
            for row in scenario_rows.to_dict(orient="records"):
                method = str(row["method"])
                signal_plan = _method_signal_plan(metrics, scenario_key, method)
                method_record = scenario_optimization.get(method, {})
                if not isinstance(method_record, Mapping):
                    method_record = {}
                method_id = tracker.ensure_method(
                    scenario_key,
                    method,
                    params={
                        "signal_plan_ns_ew_s": signal_plan,
                        "optimizer_seed": method_record.get("optimizer_seed"),
                        "objective_evaluations": (
                            method_record.get("objective_evaluations")
                            if method != "fixed"
                            else 0
                        ),
                        "training_seeds": training_seeds,
                        "evaluation_seeds": evaluation_seeds,
                        "network_sha256": network_hash,
                    },
                    tags={
                        "urbantwin.data_kind": source,
                        "urbantwin.co2_source": "simulation_only_not_observed",
                        "urbantwin.optimizer_runtime_scope": (
                            "current_invocation_cache_sensitive"
                        ),
                    },
                )
                if tracker.method_completed(scenario_key, method):
                    continue
                aggregate = {
                    target: row[source_column]
                    for source_column, target in _AGGREGATE_METRICS.items()
                }
                aggregate["optimizer_runtime_s"] = _optimizer_runtime(
                    metrics, scenario_key, method
                )
                paired_rows = paired_summary.loc[
                    (paired_summary["scenario"].astype(str) == scenario_key)
                    & (paired_summary["method"].astype(str) == method)
                ]
                if method != "fixed":
                    if len(paired_rows) != 1:
                        raise MlflowTrackingError(
                            f"Expected one paired summary for {scenario_key}/{method}."
                        )
                    aggregate.update(_paired_metrics(paired_rows.iloc[0].to_dict()))
                tracker.log_metrics(method_id, aggregate)
                tracker.finish_method(scenario_key, method)
            if not scenario_was_complete:
                tracker.log_metrics(scenario_id, {"methods": len(scenario_rows)})
                tracker.finish_scenario(scenario_key)

        safe_provenance = _write_safe_provenance(provenance, association_dir, fingerprint)
        artifact_files = [
            output / "traffic_metrics.csv",
            output / "traffic_summary.csv",
            output / "traffic_paired_deltas.csv",
            output / "traffic_paired_summary.csv",
            configuration,
            safe_provenance,
            *(Path(path) for path in figure_paths),
        ]
        tracker.log_artifacts(tracker.parent_run_id, artifact_files, artifact_path="verified")
        tracker.log_metrics(
            tracker.parent_run_id,
            {
                "scenarios": summary["scenario"].nunique(),
                "scenario_method_aggregates": len(summary),
                "evaluation_rows": len(metrics),
            },
        )
        tracker.finish()
        return {
            "run_id": tracker.parent_run_id,
            "tracking_uri": uri,
            "resumed": tracker.resumed,
            "already_completed": False,
        }
    except MlflowTrackingError as exc:
        if tracker is not None and not tracker.completed:
            try:
                tracker.mark_interrupted()
            except Exception as interruption_exc:  # noqa: BLE001
                raise MlflowTrackingError(
                    f"{exc} Additionally, MLflow interruption tagging failed: "
                    f"{interruption_exc}"
                ) from exc
        raise
    except Exception as exc:
        if tracker is not None and not tracker.completed:
            try:
                tracker.mark_interrupted()
            except Exception as interruption_exc:  # noqa: BLE001
                raise MlflowTrackingError(
                    f"MLflow traffic tracking failed: {exc}. Additionally, interruption "
                    f"tagging failed: {interruption_exc}"
                ) from exc
        raise MlflowTrackingError(f"MLflow traffic tracking failed: {exc}") from exc


__all__ = [
    "EXPERIMENT_NAME",
    "MlflowTracker",
    "MlflowTrackingError",
    "MlflowUnavailableError",
    "compatibility_sha256",
    "default_artifact_root",
    "default_tracking_uri",
    "sqlite_tracking_uri",
    "track_benchmark",
]
