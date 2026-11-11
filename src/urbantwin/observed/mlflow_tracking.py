"""Optional MLflow logging for the validated observed-traffic study."""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import uuid
from pathlib import Path
from typing import Any

import pandas as pd

from urbantwin.common import ROOT
from urbantwin.observed.artifacts import (
    ObservedArtifactIntegrityError,
    validate_observed_artifacts,
)
from urbantwin.traffic.mlflow_tracking import default_artifact_root, default_tracking_uri

EXPERIMENT_NAME = "UrbanTwin-Observed-Traffic"
STATE_SCHEMA = "urbantwin.observed-mlflow-association/v1"
_COMPATIBILITY_TAG = "urbantwin.compatibility_sha256"
_COMPLETED_TAG = "urbantwin.completed"


class ObservedMlflowTrackingError(RuntimeError):
    """Raised when explicitly requested observed-study tracking fails."""


class ObservedMlflowUnavailableError(ObservedMlflowTrackingError):
    """Raised when the optional MLflow dependency is unavailable."""


def _load_mlflow() -> tuple[Any, Any]:
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    try:
        mlflow = importlib.import_module("mlflow")
        client_module = importlib.import_module("mlflow.tracking")
    except ImportError as exc:
        raise ObservedMlflowUnavailableError(
            "MLflow is optional and not installed. Install it with "
            "`python -m pip install -e \".[mlflow]\"`."
        ) from exc
    return mlflow, client_module.MlflowClient


def _canonical_hash(value: Any) -> str:
    serialized = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _atomic_json(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _string(value: Any) -> str:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return str(value)


def _safe_metric_key(*parts: str) -> str:
    return ".".join(
        str(part).replace(" ", "_").replace("/", "_").replace(">", "gt")
        for part in parts
    )


def _experiment(client: Any, artifact_root: Path) -> Any:
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        artifact_root.mkdir(parents=True, exist_ok=True)
        experiment_id = client.create_experiment(
            EXPERIMENT_NAME, artifact_location=artifact_root.resolve().as_uri()
        )
        experiment = client.get_experiment(experiment_id)
    return experiment


def _association_path(root: Path, compatibility: str, tracking_uri: str) -> Path:
    uri_hash = hashlib.sha256(tracking_uri.encode("utf-8")).hexdigest()[:16]
    return (
        root
        / ".urbantwin"
        / "mlflow"
        / "run_associations"
        / "observed"
        / f"{compatibility}_{uri_hash}.json"
    )


def _compatibility_record(verified: dict[str, Any]) -> dict[str, Any]:
    provenance = verified["provenance"]
    return {
        "source_counts_sha256": provenance["source_counts_sha256"],
        "source_locations_sha256": provenance["source_locations_sha256"],
        "source_manifest_sha256": provenance["source_manifest_sha256"],
        "config_sha256": provenance["config_sha256"],
        "split_manifest_sha256": provenance["split_manifest_sha256"],
        "cv_split_manifest_sha256": provenance["cv_split_manifest_sha256"],
        "artifact_hashes": provenance["artifact_hashes"],
        "eo_context": provenance.get("eo_context"),
        "road_context": provenance.get("road_context"),
        "git": provenance["git"],
    }


def observed_run_association(
    output_dir: str | Path,
    *,
    project_root: str | Path = ROOT,
    tracking_uri: str | None = None,
) -> dict[str, Any] | None:
    """Return the compatible local run association without importing MLflow."""
    root = Path(project_root).resolve()
    verified = validate_observed_artifacts(root, Path(output_dir).resolve())
    if verified is None:
        return None
    uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI") or default_tracking_uri(root)
    compatibility = _canonical_hash(_compatibility_record(verified))
    path = _association_path(root, compatibility, uri)
    if not path.is_file():
        return None
    return _read_association(path, compatibility, uri)


def _read_association(path: Path, compatibility: str, tracking_uri: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ObservedMlflowTrackingError(f"Invalid observed MLflow association: {path}") from exc
    if not isinstance(value, dict) or value.get("schema_version") != STATE_SCHEMA:
        raise ObservedMlflowTrackingError("Observed MLflow association schema is incompatible")
    if value.get("compatibility_sha256") != compatibility:
        raise ObservedMlflowTrackingError("Observed MLflow association is incompatible")
    if value.get("tracking_uri") != tracking_uri:
        raise ObservedMlflowTrackingError("Observed MLflow tracking URI changed")
    run_id = value.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ObservedMlflowTrackingError("Observed MLflow association has no run ID")
    return value


def track_observed_study(
    output_dir: str | Path,
    *,
    project_root: str | Path = ROOT,
    tracking_uri: str | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    """Validate and log one completed observed study without changing local evidence."""
    root = Path(project_root).resolve()
    output = Path(output_dir).resolve()
    try:
        verified = validate_observed_artifacts(root, output)
    except ObservedArtifactIntegrityError as exc:
        raise ObservedMlflowTrackingError(
            f"Observed artifacts failed strict validation before MLflow logging: {exc}"
        ) from exc
    if verified is None:
        raise ObservedMlflowTrackingError(f"No observed study exists in {output}")
    provenance = verified["provenance"]
    split = verified["split_manifest"]
    compatibility_record = _compatibility_record(verified)
    compatibility = _canonical_hash(compatibility_record)
    uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI") or default_tracking_uri(root)
    state_path = _association_path(root, compatibility, uri)
    mlflow, client_type = _load_mlflow()
    try:
        mlflow.set_tracking_uri(uri)
        client = client_type(tracking_uri=uri)
        experiment = _experiment(client, default_artifact_root(root) / "observed")
        run = None
        association = None
        was_resumed = False
        if state_path.is_file():
            association = _read_association(state_path, compatibility, uri)
            try:
                run = client.get_run(association["run_id"])
            except Exception as exc:
                raise ObservedMlflowTrackingError(
                    "Associated observed MLflow run cannot be retrieved"
                ) from exc
            complete = (
                str(run.info.status).upper() == "FINISHED"
                and run.data.tags.get(_COMPLETED_TAG) == "true"
            )
            if complete:
                return {
                    "run_id": run.info.run_id,
                    "experiment_id": experiment.experiment_id,
                    "tracking_uri": uri,
                    "already_completed": True,
                    "resumed": False,
                }
            if not resume:
                raise ObservedMlflowTrackingError(
                    "A compatible incomplete observed MLflow run exists; rerun with --resume."
                )
            was_resumed = True
        if run is None:
            run = client.create_run(
                experiment.experiment_id,
                tags={
                    "mlflow.runName": "observed-traffic:frozen-study",
                    _COMPATIBILITY_TAG: compatibility,
                    _COMPLETED_TAG: "false",
                    "urbantwin.data_kind": "real_observed_traffic_analysis",
                    "urbantwin.simulation_status": "not_run_scientific_gate_failed",
                    "urbantwin.license_status": str(provenance["license"]["status"]),
                    "urbantwin.permission_status": str(
                        provenance["author_reported_permission"]["status"]
                    ),
                    "urbantwin.public_redistribution_status": str(
                        provenance["author_reported_permission"][
                            "public_redistribution_status"
                        ]
                    ),
                    "urbantwin.eo_role": (
                        "sentinel_2024_context_worldcover_2021_historic_reference"
                    ),
                    "urbantwin.current_osm_role": "third_party_map_context_not_observation",
                },
            )
            association = {
                "schema_version": STATE_SCHEMA,
                "compatibility_sha256": compatibility,
                "tracking_uri": uri,
                "experiment_id": experiment.experiment_id,
                "run_id": run.info.run_id,
                "completed": False,
            }
            _atomic_json(association, state_path)
        run_id = run.info.run_id
        params = {
            "source_counts_sha256": provenance["source_counts_sha256"],
            "source_locations_sha256": provenance["source_locations_sha256"],
            "source_manifest_sha256": provenance["source_manifest_sha256"],
            "config_sha256": provenance["config_sha256"],
            "split_manifest_sha256": provenance["split_manifest_sha256"],
            "cv_split_manifest_sha256": provenance["cv_split_manifest_sha256"],
            "git_commit": provenance["git"]["commit"],
            "git_dirty": provenance["git"]["dirty"],
            "train_dates": split["train_dates"],
            "test_dates": split["test_dates"],
            "train_rows": split["train_rows"],
            "test_rows": split["test_rows"],
            "train_inferred_sites": split["train_inferred_sites"],
            "test_inferred_sites": split["test_inferred_sites"],
            "excluded_ambiguous_rows": split["excluded_ambiguous_rows"],
            "inferred_site_definition": split["inferred_site_definition"],
            "license_status": provenance["license"]["status"],
            "permission_status": provenance["author_reported_permission"]["status"],
            "public_redistribution_status": provenance[
                "author_reported_permission"
            ]["public_redistribution_status"],
            "selected_model": provenance["model_selection"]["model"],
            "selected_feature_set": provenance["model_selection"]["feature_set"],
            "selected_grouped_cv_mean_mae": provenance["model_selection"]["mean_mae"],
            "lowest_holdout_point_model": provenance[
                "lowest_observed_holdout_point_estimate"
            ]["model"],
            "lowest_holdout_point_feature_set": provenance[
                "lowest_observed_holdout_point_estimate"
            ]["feature_set"],
            "measured_demand_sumo": "not_run",
            "sumo_blockers": provenance["study_gate"]["blockers"],
        }
        eo_context = provenance.get("eo_context") or {}
        road_context = provenance.get("road_context") or {}
        params.update(
            {
                "eo_stack_sha256": eo_context.get("eo_stack_sha256", "not_included"),
                "eo_provenance_sha256": eo_context.get(
                    "provenance_sha256", "not_included"
                ),
                "sentinel_item_ids": eo_context.get("sentinel_item_ids", []),
                "worldcover_item_ids": eo_context.get("worldcover_item_ids", []),
                "road_manifest_sha256": road_context.get(
                    "manifest_sha256", "not_included"
                ),
                "road_source_kind": road_context.get("source_kind", "not_included"),
                "road_nodes": road_context.get("nodes", "not_included"),
                "road_directed_edges": road_context.get(
                    "directed_edges", "not_included"
                ),
                "road_simulation_readiness": road_context.get(
                    "simulation_readiness", "not_included"
                ),
            }
        )
        for key, value in params.items():
            client.log_param(run_id, key, _string(value))

        metrics: pd.DataFrame = verified["metrics"]
        for row in metrics.to_dict(orient="records"):
            prefix = _safe_metric_key(str(row["model"]), str(row["feature_set"]))
            for metric_name in (
                "mae",
                "rmse",
                "median_absolute_error",
                "mean_bias",
                "mean_poisson_deviance",
                "mae_ci95_low",
                "mae_ci95_high",
                "rmse_ci95_low",
                "rmse_ci95_high",
                "rows",
            ):
                client.log_metric(
                    run_id,
                    _safe_metric_key(prefix, metric_name),
                    float(row[metric_name]),
                )
        cv = verified["cv_metrics"]
        cv_summary = cv.groupby(["model", "feature_set"], observed=True)[["mae", "rmse"]].mean()
        for (model, feature_set), row in cv_summary.iterrows():
            prefix = _safe_metric_key("grouped_cv", str(model), str(feature_set))
            client.log_metric(run_id, f"{prefix}.mean_mae", float(row["mae"]))
            client.log_metric(run_id, f"{prefix}.mean_rmse", float(row["rmse"]))
        client.log_metric(
            run_id,
            "measured_observed_vehicle_total",
            float(verified["vehicle_classes"]["observed_vehicles"].sum()),
        )

        safe_artifacts = [
            output / "observed_metrics.csv",
            output / "observed_cv_metrics.csv",
            output / "observed_characterization.csv",
            output / "observed_vehicle_classes.csv",
            output / "observed_site_errors.csv",
            output / "historic_eo_context.json",
            output / "split_manifest.json",
            output / "cv_split_manifest.json",
            output / "provenance.json",
            *(output / relative for relative in provenance["figure_files"]),
        ]
        for path in safe_artifacts:
            client.log_artifact(run_id, str(path), artifact_path="verified_aggregates")
        client.set_tag(run_id, _COMPLETED_TAG, "true")
        client.set_terminated(run_id, status="FINISHED")
        association = {
            **(association or {}),
            "schema_version": STATE_SCHEMA,
            "compatibility_sha256": compatibility,
            "tracking_uri": uri,
            "experiment_id": experiment.experiment_id,
            "run_id": run_id,
            "completed": True,
        }
        _atomic_json(association, state_path)
        return {
            "run_id": run_id,
            "experiment_id": experiment.experiment_id,
            "tracking_uri": uri,
            "already_completed": False,
            "resumed": was_resumed,
        }
    except ObservedMlflowTrackingError:
        raise
    except Exception as exc:
        raise ObservedMlflowTrackingError(
            f"MLflow logging failed; validated local observed artifacts remain intact: {exc}"
        ) from exc


__all__ = [
    "EXPERIMENT_NAME",
    "ObservedMlflowTrackingError",
    "ObservedMlflowUnavailableError",
    "observed_run_association",
    "track_observed_study",
]
