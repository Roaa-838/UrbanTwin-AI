"""Strict validation for the local measured-traffic research bundle."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from urbantwin.common import require_real_provenance, sha256_file


class ObservedArtifactIntegrityError(ValueError):
    """Raised when a measured-traffic result bundle is stale or inconsistent."""


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ObservedArtifactIntegrityError(f"Invalid JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise ObservedArtifactIntegrityError(f"Expected a JSON object: {path}")
    return value


def _csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise ObservedArtifactIntegrityError(f"Invalid CSV artifact: {path}") from exc


def _require_columns(frame: pd.DataFrame, required: set[str], path: Path) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ObservedArtifactIntegrityError(
            f"{path.name} is missing required columns: {', '.join(missing)}"
        )


def _finite_nonnegative(frame: pd.DataFrame, columns: list[str], path: Path) -> None:
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any() or not values.map(math.isfinite).all() or (values < 0).any():
            raise ObservedArtifactIntegrityError(
                f"{path.name} contains invalid non-negative metric values in {column}"
            )


def validate_observed_artifacts(
    project_root: str | Path,
    output_dir: str | Path | None = None,
    *,
    source_dir: str | Path | None = None,
) -> dict[str, Any] | None:
    """Validate one measured-traffic bundle before display, reporting, or tracking."""
    root = Path(project_root).resolve()
    output = (
        Path(output_dir).resolve()
        if output_dir is not None
        else (root / "results" / "observed_traffic").resolve()
    )
    provenance_path = output / "provenance.json"
    if not output.exists():
        return None
    if not output.is_dir() or not provenance_path.is_file():
        raise ObservedArtifactIntegrityError(
            f"Observed result directory is incomplete: {output}"
        )
    provenance = _json_object(provenance_path)
    if provenance.get("schema_version") != "urbantwin.observed-benchmark/v2":
        raise ObservedArtifactIntegrityError("Observed provenance schema is incompatible")
    if provenance.get("data_kind") != "real_observed_traffic_analysis":
        raise ObservedArtifactIntegrityError("Observed results have the wrong data_kind")
    gate = provenance.get("study_gate")
    if not isinstance(gate, dict) or gate.get("allow_measured_demand_sumo") is not False:
        raise ObservedArtifactIntegrityError(
            "Observed provenance does not preserve the measured-demand SUMO blocker"
        )
    labels = provenance.get("source_labels")
    if not isinstance(labels, dict) or not str(labels.get("traffic", "")).startswith(
        "measured TfC"
    ):
        raise ObservedArtifactIntegrityError("Observed traffic source label is missing")
    if labels.get("simulation") != "not run; scientific gate failed":
        raise ObservedArtifactIntegrityError("Observed results mislabel simulation status")
    permission = provenance.get("author_reported_permission")
    if not isinstance(permission, dict) or permission.get("status") != (
        "author_reported_academic_research_reuse_authorized"
    ):
        raise ObservedArtifactIntegrityError(
            "Observed provenance lacks the author-reported academic reuse permission status"
        )
    if permission.get("public_redistribution_status") != "requires_exact_written_terms":
        raise ObservedArtifactIntegrityError(
            "Observed provenance overstates public redistribution permission"
        )

    configuration = provenance.get("configuration")
    configured_road_directory = None
    if isinstance(configuration, dict):
        spatial = configuration.get("spatial")
        if isinstance(spatial, dict):
            configured_road_directory = spatial.get("current_osm_directory")
    road_context = provenance.get("road_context")
    if configured_road_directory is not None and not isinstance(road_context, dict):
        raise ObservedArtifactIntegrityError(
            "Observed provenance omits the configured current OSM road context"
        )
    if isinstance(road_context, dict):
        from urbantwin.observed.roads import (
            ObservedRoadError,
            validate_observed_road_bundle,
        )

        recorded_road_directory = road_context.get("input_directory")
        if not isinstance(recorded_road_directory, str) or not recorded_road_directory:
            raise ObservedArtifactIntegrityError(
                "Observed road context has no recorded current OSM input directory"
            )
        road_root = (root / recorded_road_directory).resolve()
        try:
            road_root.relative_to(root)
        except ValueError as exc:
            raise ObservedArtifactIntegrityError(
                "Observed road input directory escapes the project root"
            ) from exc
        try:
            road_manifest = validate_observed_road_bundle(road_root)
        except ObservedRoadError as exc:
            raise ObservedArtifactIntegrityError(
                f"Configured current OSM road bundle is invalid: {exc}"
            ) from exc
        if road_manifest is None:
            raise ObservedArtifactIntegrityError(
                f"Configured current OSM road bundle is missing: {road_root}"
            )
        manifest_path = road_root / "road_manifest.json"
        if sha256_file(manifest_path) != road_context.get("manifest_sha256"):
            raise ObservedArtifactIntegrityError("Current OSM road manifest hash mismatch")
        for key in (
            "data_kind",
            "source_kind",
            "nodes",
            "directed_edges",
            "connectivity",
            "map_matching",
            "simulation_readiness",
            "attribution",
            "license",
        ):
            if road_context.get(key) != road_manifest.get(key):
                raise ObservedArtifactIntegrityError(
                    f"Current OSM road provenance disagrees on {key}"
                )
        if road_manifest.get("simulation_readiness") != (
            "blocked_missing_turns_routes_and_signal_observations"
        ):
            raise ObservedArtifactIntegrityError(
                "Current OSM context does not preserve the simulation-readiness blocker"
            )

    eo_context = provenance.get("eo_context")
    if eo_context is not None:
        if not isinstance(eo_context, dict):
            raise ObservedArtifactIntegrityError("Observed EO context provenance is malformed")
        recorded_eo_directory = eo_context.get("input_directory")
        if not isinstance(recorded_eo_directory, str) or not recorded_eo_directory:
            raise ObservedArtifactIntegrityError(
                "Observed EO context has no recorded input directory"
            )
        eo_root = (root / recorded_eo_directory).resolve()
        try:
            eo_root.relative_to(root)
        except ValueError as exc:
            raise ObservedArtifactIntegrityError(
                "Observed EO input directory escapes the project root"
            ) from exc
        eo_provenance_path = eo_root / "provenance.json"
        eo_stack_path = eo_root / "eo_stack.npz"
        if not eo_provenance_path.is_file() or not eo_stack_path.is_file():
            raise ObservedArtifactIntegrityError(
                f"Observed EO input bundle is incomplete: {eo_root}"
            )
        eo_provenance = _json_object(eo_provenance_path)
        try:
            require_real_provenance(eo_provenance)
        except (KeyError, TypeError, ValueError) as exc:
            raise ObservedArtifactIntegrityError(
                f"Observed EO input provenance is invalid: {exc}"
            ) from exc
        if (
            sha256_file(eo_provenance_path) != eo_context.get("provenance_sha256")
            or sha256_file(eo_stack_path) != eo_context.get("eo_stack_sha256")
            or eo_provenance.get("eo_stack_sha256") != eo_context.get("eo_stack_sha256")
        ):
            raise ObservedArtifactIntegrityError("Observed EO input hash mismatch")
        if eo_provenance.get("sentinel_stac_item_ids") != eo_context.get(
            "sentinel_item_ids"
        ) or eo_provenance.get("worldcover_stac_item_ids") != eo_context.get(
            "worldcover_item_ids"
        ):
            raise ObservedArtifactIntegrityError("Observed EO item identifiers disagree")
        if eo_context.get("worldcover_year") != 2021 or "historic reference" not in str(
            eo_context.get("interpretation", "")
        ):
            raise ObservedArtifactIntegrityError(
                "Observed EO context does not preserve the WorldCover 2021 historic label"
            )

    config_path = root / "configs" / "observed_traffic.yaml"
    if not config_path.is_file() or sha256_file(config_path) != provenance.get("config_sha256"):
        raise ObservedArtifactIntegrityError(
            "Observed results are stale or mismatched with configs/observed_traffic.yaml"
        )
    source = (
        Path(source_dir).resolve()
        if source_dir is not None
        else (root / "data" / "observed" / "tfc" / "normalized_v1").resolve()
    )
    source_manifest_path = source / "source_manifest.json"
    if not source_manifest_path.is_file():
        raise ObservedArtifactIntegrityError(
            f"Normalized source manifest required for validation is missing: {source_manifest_path}"
        )
    if sha256_file(source_manifest_path) != provenance.get("source_manifest_sha256"):
        raise ObservedArtifactIntegrityError("Observed source manifest hash mismatch")
    source_manifest = _json_object(source_manifest_path)
    if source_manifest.get("data_kind") != "real_observed_traffic":
        raise ObservedArtifactIntegrityError("Normalized source is not real observed traffic")
    portal_license = source_manifest.get("license") or {}
    if portal_license.get("status") != "unspecified_by_publisher":
        raise ObservedArtifactIntegrityError(
            "Observed portal license status is not the audited unspecified status"
        )
    if portal_license.get("redistribution_permitted") is not False:
        raise ObservedArtifactIntegrityError("Unexpected observed-data redistribution claim")
    if provenance.get("portal_license") != source_manifest.get("license"):
        raise ObservedArtifactIntegrityError(
            "Observed provenance does not preserve the portal license metadata"
        )
    if provenance.get("license") != source_manifest.get("license"):
        raise ObservedArtifactIntegrityError("Legacy observed license field changed")
    if not isinstance(configuration, dict) or configuration.get("permission") != permission:
        raise ObservedArtifactIntegrityError(
            "Observed permission provenance disagrees with the frozen configuration"
        )
    if (
        ((source_manifest.get("sources") or {}).get("counts") or {}).get("sha256")
        != provenance.get("source_counts_sha256")
        or ((source_manifest.get("sources") or {}).get("locations") or {}).get("sha256")
        != provenance.get("source_locations_sha256")
    ):
        raise ObservedArtifactIntegrityError("Observed source archive hashes disagree")

    artifact_hashes = provenance.get("artifact_hashes")
    if not isinstance(artifact_hashes, dict) or not artifact_hashes:
        raise ObservedArtifactIntegrityError("Observed provenance has no artifact hashes")
    for relative, expected in artifact_hashes.items():
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise ObservedArtifactIntegrityError("Observed artifact hash record is malformed")
        path = (output / relative).resolve()
        try:
            path.relative_to(output)
        except ValueError as exc:
            raise ObservedArtifactIntegrityError(
                f"Observed artifact path escapes its bundle: {relative}"
            ) from exc
        if not path.is_file() or sha256_file(path) != expected:
            raise ObservedArtifactIntegrityError(
                f"Observed artifact is missing or hash-mismatched: {relative}"
            )
    required_files = {
        "observed_metrics.csv",
        "observed_cv_metrics.csv",
        "observed_cv_predictions.csv",
        "observed_predictions.csv",
        "observed_site_errors.csv",
        "observed_vehicle_classes.csv",
        "observed_characterization.csv",
        "segment_features.geojson",
        "historic_eo_context.json",
        "split_manifest.json",
        "cv_split_manifest.json",
    }
    missing_hashes = sorted(required_files - set(artifact_hashes))
    if missing_hashes:
        raise ObservedArtifactIntegrityError(
            "Observed provenance omits required artifacts: " + ", ".join(missing_hashes)
        )

    split_path = output / "split_manifest.json"
    split_manifest = _json_object(split_path)
    if sha256_file(split_path) != provenance.get("split_manifest_sha256"):
        raise ObservedArtifactIntegrityError("Observed split manifest hash mismatch")
    if split_manifest.get("schema_version") != "urbantwin.observed-split/v1":
        raise ObservedArtifactIntegrityError("Observed split schema is incompatible")
    if split_manifest.get("site_overlap") != []:
        raise ObservedArtifactIntegrityError("Observed final holdout is not site-disjoint")
    train_dates = set(split_manifest.get("train_dates") or [])
    test_dates = set(split_manifest.get("test_dates") or [])
    if not train_dates or not test_dates or train_dates & test_dates:
        raise ObservedArtifactIntegrityError("Observed train/test date manifests overlap")
    if max(train_dates) >= min(test_dates):
        raise ObservedArtifactIntegrityError("Observed final dates are not chronological")

    cv_split_path = output / "cv_split_manifest.json"
    cv_split_manifest = _json_object(cv_split_path)
    if sha256_file(cv_split_path) != provenance.get("cv_split_manifest_sha256"):
        raise ObservedArtifactIntegrityError("Observed grouped-CV split manifest hash mismatch")
    if cv_split_manifest.get("schema_version") != "urbantwin.observed-cv-split/v1":
        raise ObservedArtifactIntegrityError("Observed grouped-CV split schema is incompatible")
    if cv_split_manifest.get("protocol") != "training_only_five_fold_grouped_oof":
        raise ObservedArtifactIntegrityError("Observed grouped-CV protocol is mislabeled")
    if (
        cv_split_manifest.get("group_key") != "inferred_site_id"
        or cv_split_manifest.get("n_splits") != 5
        or cv_split_manifest.get("training_rows") != split_manifest.get("train_rows")
        or cv_split_manifest.get("training_inferred_sites")
        != split_manifest.get("train_inferred_sites")
        or cv_split_manifest.get("training_observation_ids_sha256")
        != split_manifest.get("training_observation_ids_sha256")
    ):
        raise ObservedArtifactIntegrityError(
            "Observed grouped-CV manifest disagrees with the frozen training partition"
        )
    folds = cv_split_manifest.get("folds")
    if not isinstance(folds, list) or {fold.get("fold") for fold in folds} != {1, 2, 3, 4, 5}:
        raise ObservedArtifactIntegrityError("Observed grouped-CV manifest lacks five folds")
    if any(fold.get("site_overlap") != [] for fold in folds):
        raise ObservedArtifactIntegrityError("Observed grouped-CV manifest contains site leakage")
    training_rows = int(split_manifest["train_rows"])
    training_sites = int(split_manifest["train_inferred_sites"])
    for fold in folds:
        if (
            int(fold.get("fit_rows", -1)) + int(fold.get("validation_rows", -1))
            != training_rows
            or int(fold.get("fit_inferred_sites", -1))
            + int(fold.get("validation_inferred_sites", -1))
            != training_sites
        ):
            raise ObservedArtifactIntegrityError(
                "Observed grouped-CV fold sizes disagree with the training partition"
            )
    for key in (
        "source_manifest_sha256",
        "source_counts_sha256",
        "source_locations_sha256",
        "config_sha256",
    ):
        if cv_split_manifest.get(key) != provenance.get(key):
            raise ObservedArtifactIntegrityError(
                f"Observed grouped-CV manifest disagrees on {key}"
            )

    paths = {
        "metrics": output / "observed_metrics.csv",
        "cv_metrics": output / "observed_cv_metrics.csv",
        "cv_predictions": output / "observed_cv_predictions.csv",
        "predictions": output / "observed_predictions.csv",
        "site_errors": output / "observed_site_errors.csv",
        "vehicle_classes": output / "observed_vehicle_classes.csv",
        "characterization": output / "observed_characterization.csv",
    }
    values = {name: _csv(path) for name, path in paths.items()}
    metrics = values["metrics"]
    metric_columns = {
        "evaluation",
        "protocol",
        "model",
        "feature_set",
        "information_budget",
        "rows",
        "inferred_sites",
        "mae",
        "rmse",
        "mean_poisson_deviance",
        "mae_ci95_low",
        "mae_ci95_high",
        "rmse_ci95_low",
        "rmse_ci95_high",
    }
    _require_columns(metrics, metric_columns, paths["metrics"])
    if metrics.empty or metrics.duplicated(["model", "feature_set", "protocol"]).any():
        raise ObservedArtifactIntegrityError("Observed holdout metrics are empty or duplicated")
    _finite_nonnegative(
        metrics,
        [
            "rows",
            "inferred_sites",
            "mae",
            "rmse",
            "mean_poisson_deviance",
            "mae_ci95_low",
            "mae_ci95_high",
            "rmse_ci95_low",
            "rmse_ci95_high",
        ],
        paths["metrics"],
    )
    if set(metrics["evaluation"]) != {"frozen_holdout"}:
        raise ObservedArtifactIntegrityError("Observed metrics are not frozen-holdout results")
    persistence = metrics.loc[metrics["model"] == "persistence"]
    if len(persistence) != 1 or persistence.iloc[0]["information_budget"] != (
        "online_forecast_with_previous_measurement"
    ):
        raise ObservedArtifactIntegrityError("Persistence information budget is mislabeled")
    measurement_free = metrics.loc[metrics["model"] != "persistence"]
    if measurement_free.empty or set(measurement_free["information_budget"]) != {
        "measurement_free_estimation"
    }:
        raise ObservedArtifactIntegrityError("Measurement-free model labels are inconsistent")

    predictions = values["predictions"]
    _require_columns(
        predictions,
        {
            "observation_id",
            "model",
            "feature_set",
            "protocol",
            "information_budget",
            "observed_vehicle_total",
            "prediction",
        },
        paths["predictions"],
    )
    _finite_nonnegative(
        predictions,
        ["observed_vehicle_total", "prediction"],
        paths["predictions"],
    )
    counts = predictions.groupby(["model", "feature_set", "protocol"], observed=True).size()
    for row in metrics.itertuples(index=False):
        key = (row.model, row.feature_set, row.protocol)
        if int(counts.get(key, -1)) != int(row.rows):
            raise ObservedArtifactIntegrityError(
                f"Prediction row count disagrees for {row.model}/{row.feature_set}"
            )

    cv_metrics = values["cv_metrics"]
    _require_columns(
        cv_metrics,
        {
            "fold",
            "model",
            "feature_set",
            "information_budget",
            "train_rows",
            "validation_rows",
            "train_inferred_sites",
            "validation_inferred_sites",
            "mae",
            "rmse",
        },
        paths["cv_metrics"],
    )
    if set(pd.to_numeric(cv_metrics["fold"], errors="coerce")) != {1, 2, 3, 4, 5}:
        raise ObservedArtifactIntegrityError("Observed development metrics lack five grouped folds")
    if cv_metrics.duplicated(["fold", "model", "feature_set"]).any():
        raise ObservedArtifactIntegrityError("Observed grouped-CV metrics are duplicated")
    if set(cv_metrics["information_budget"]) != {"measurement_free_estimation"}:
        raise ObservedArtifactIntegrityError("Observed grouped-CV information budget is invalid")

    selection = provenance.get("model_selection")
    if not isinstance(selection, dict):
        raise ObservedArtifactIntegrityError("Observed provenance has no model-selection record")
    if (
        selection.get("selection_partition") != "training_only"
        or selection.get("selection_protocol")
        != "five_fold_grouped_cv_by_inferred_site"
        or selection.get("holdout_scores_used_for_selection") is not False
    ):
        raise ObservedArtifactIntegrityError(
            "Observed model selection is not confined to training-only grouped CV"
        )
    if split_manifest.get("model_selection") != selection or cv_split_manifest.get(
        "model_selection"
    ) != selection:
        raise ObservedArtifactIntegrityError("Observed model-selection records disagree")
    cv_summary = (
        cv_metrics.groupby(["model", "feature_set"], observed=True, as_index=False)
        .agg(mean_mae=("mae", "mean"), mean_rmse=("rmse", "mean"))
        .sort_values(
            ["mean_mae", "mean_rmse", "model", "feature_set"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )
    selected = cv_summary.iloc[0]
    if (
        selection.get("model") != selected["model"]
        or selection.get("feature_set") != selected["feature_set"]
        or not math.isclose(float(selection.get("mean_mae", math.nan)), selected["mean_mae"])
        or not math.isclose(float(selection.get("mean_rmse", math.nan)), selected["mean_rmse"])
        or selection.get("candidate_count") != len(cv_summary)
    ):
        raise ObservedArtifactIntegrityError(
            "Observed selected model does not match training-only grouped-CV ranking"
        )

    lowest_record = provenance.get("lowest_observed_holdout_point_estimate")
    if not isinstance(lowest_record, dict) or lowest_record.get("selection_role") != (
        "descriptive_only_not_used_for_model_selection"
    ):
        raise ObservedArtifactIntegrityError(
            "Observed lowest holdout point estimate is not labeled descriptive-only"
        )
    if split_manifest.get("lowest_observed_holdout_point_estimate") != lowest_record:
        raise ObservedArtifactIntegrityError("Observed lowest holdout records disagree")
    lowest = measurement_free.sort_values(
        ["mae", "rmse", "model", "feature_set"], kind="mergesort"
    ).iloc[0]
    if (
        lowest_record.get("model") != lowest["model"]
        or lowest_record.get("feature_set") != lowest["feature_set"]
        or not math.isclose(float(lowest_record.get("mae", math.nan)), lowest["mae"])
        or not math.isclose(float(lowest_record.get("rmse", math.nan)), lowest["rmse"])
    ):
        raise ObservedArtifactIntegrityError(
            "Observed descriptive holdout record disagrees with frozen metrics"
        )

    cv_predictions = values["cv_predictions"]
    _require_columns(
        cv_predictions,
        {
            "fold",
            "observation_id",
            "inferred_site_id",
            "model",
            "feature_set",
            "protocol",
            "information_budget",
            "observed_vehicle_total",
            "prediction",
        },
        paths["cv_predictions"],
    )
    _finite_nonnegative(
        cv_predictions,
        ["observed_vehicle_total", "prediction"],
        paths["cv_predictions"],
    )
    if set(cv_predictions["protocol"]) != {"training_only_five_fold_grouped_oof"}:
        raise ObservedArtifactIntegrityError("Observed OOF predictions use the wrong protocol")
    if set(cv_predictions["information_budget"]) != {"measurement_free_estimation"}:
        raise ObservedArtifactIntegrityError("Observed OOF information budget is invalid")
    cv_predictions["fold"] = pd.to_numeric(cv_predictions["fold"], errors="coerce")
    if set(cv_predictions["fold"]) != {1, 2, 3, 4, 5}:
        raise ObservedArtifactIntegrityError("Observed OOF predictions lack five folds")
    if cv_predictions.duplicated(["fold", "model", "feature_set", "observation_id"]).any():
        raise ObservedArtifactIntegrityError("Observed OOF predictions contain duplicates")
    expected_training_rows = int(split_manifest["train_rows"])
    specifications = set(
        zip(cv_metrics["model"].astype(str), cv_metrics["feature_set"].astype(str))
    )
    for model, feature_set in specifications:
        subset = cv_predictions.loc[
            (cv_predictions["model"] == model)
            & (cv_predictions["feature_set"] == feature_set)
        ]
        if len(subset) != expected_training_rows or subset["observation_id"].nunique() != (
            expected_training_rows
        ):
            raise ObservedArtifactIntegrityError(
                f"Observed OOF coverage is incomplete for {model}/{feature_set}"
            )
        if _canonical_hash(sorted(subset["observation_id"].astype(str))) != split_manifest.get(
            "training_observation_ids_sha256"
        ):
            raise ObservedArtifactIntegrityError(
                f"Observed OOF membership is mismatched for {model}/{feature_set}"
            )
    for row in cv_metrics.itertuples(index=False):
        count = len(
            cv_predictions.loc[
                (cv_predictions["fold"] == int(row.fold))
                & (cv_predictions["model"] == row.model)
                & (cv_predictions["feature_set"] == row.feature_set)
            ]
        )
        if count != int(row.validation_rows):
            raise ObservedArtifactIntegrityError(
                f"Observed OOF row count disagrees for fold {row.fold} "
                f"{row.model}/{row.feature_set}"
            )
    reference_model, reference_features = min(specifications)
    reference = cv_predictions.loc[
        (cv_predictions["model"] == reference_model)
        & (cv_predictions["feature_set"] == reference_features)
    ]
    for fold_record in folds:
        fold = int(fold_record["fold"])
        validation = reference.loc[reference["fold"] == fold]
        fit = reference.loc[reference["fold"] != fold]
        if (
            _canonical_hash(sorted(validation["observation_id"].astype(str)))
            != fold_record.get("validation_observation_ids_sha256")
            or _canonical_hash(sorted(fit["observation_id"].astype(str)))
            != fold_record.get("fit_observation_ids_sha256")
            or _canonical_hash(sorted(validation["inferred_site_id"].astype(str).unique()))
            != fold_record.get("validation_inferred_site_ids_sha256")
            or _canonical_hash(sorted(fit["inferred_site_id"].astype(str).unique()))
            != fold_record.get("fit_inferred_site_ids_sha256")
        ):
            raise ObservedArtifactIntegrityError(
                f"Observed OOF membership disagrees with grouped-CV fold {fold}"
            )
    _require_columns(
        values["vehicle_classes"],
        {"vehicle_class", "observed_vehicles", "share_of_observed_vehicles"},
        paths["vehicle_classes"],
    )
    if len(values["vehicle_classes"]) != 12:
        raise ObservedArtifactIntegrityError("Observed class summary must retain 12 classes")
    return {
        **values,
        "provenance": provenance,
        "split_manifest": split_manifest,
        "cv_split_manifest": cv_split_manifest,
        "source_manifest": source_manifest,
        "output_dir": output,
        "hashes": dict(artifact_hashes),
    }


__all__ = ["ObservedArtifactIntegrityError", "validate_observed_artifacts"]
