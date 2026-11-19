"""Private secondary diagnostics for a validated observed-traffic benchmark.

The module consumes only a bundle that has passed the repository's strict
observed-artifact validator.  It never refits or reselects the frozen primary
model.  Site-level and derived measured-traffic outputs are written to an
explicitly private, Git-ignored bundle because redistribution rights remain
unconfirmed.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from urbantwin.common import (
    configuration_hash_record,
    load_yaml,
    sha256_file,
    validate_configuration_hash,
)
from urbantwin.observed.artifacts import (
    ObservedArtifactIntegrityError,
    validate_observed_artifacts,
)

CONFIG_SCHEMA = "urbantwin.observed-error-analysis-config/v1"
ANALYSIS_SCHEMA = "urbantwin.observed-error-analysis/v1"
PRIVATE_STATUS = "private_not_redistributable_without_written_authorization"

_PREDICTION_COLUMNS = {
    "observation_id",
    "inferred_site_id",
    "observed_vehicle_total",
    "protocol",
    "model",
    "feature_set",
    "information_budget",
    "prediction",
}
_PRIVATE_REQUIRED_FILES = {
    "site_errors_private.csv",
    "volume_quantile_errors_private.csv",
    "paired_comparisons_private.csv",
    "paired_bootstrap_replicates_private.csv",
    "leave_one_site_out_private.csv",
    "cv_holdout_comparison_private.csv",
    "persistence_private.csv",
    "private_analysis_report.md",
    "figures/site_mae_private.png",
    "figures/volume_error_private.png",
    "figures/paired_uncertainty_private.png",
    "figures/cv_holdout_private.png",
}
_IDENTIFIER_COLUMNS = {
    "observation_id",
    "segment_id",
    "inferred_site_id",
    "observed_date",
    "interval_start_local",
}


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _specification(value: Any, location: str) -> dict[str, str]:
    if not isinstance(value, dict):
        raise TypeError(f"{location} must be a mapping")
    required = {"model", "feature_set", "information_budget"}
    missing = sorted(required - set(value))
    if missing:
        raise ValueError(f"{location} is missing: {', '.join(missing)}")
    result = {key: str(value[key]).strip() for key in required}
    if not all(result.values()):
        raise ValueError(f"{location} values must be non-empty strings")
    return result


def load_error_analysis_config(path: str | Path) -> dict[str, Any]:
    """Load the prespecified, publication-safe secondary-analysis protocol."""

    root = load_yaml(path)
    if root.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError(f"Expected analysis configuration schema {CONFIG_SCHEMA!r}")
    analysis = root.get("analysis")
    if not isinstance(analysis, dict):
        raise TypeError("Analysis configuration must contain an analysis mapping")
    required = {
        "bootstrap_seed",
        "bootstrap_replicates",
        "confidence_level",
        "volume_quantiles",
        "expected_holdout_sites",
        "primary_specification",
        "comparisons",
        "private_outputs_only",
    }
    missing = sorted(required - set(analysis))
    if missing:
        raise ValueError("Analysis configuration is missing: " + ", ".join(missing))
    seed = analysis["bootstrap_seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("bootstrap_seed must be a non-negative integer")
    replicates = _positive_integer(analysis["bootstrap_replicates"], "bootstrap_replicates")
    quantiles = _positive_integer(analysis["volume_quantiles"], "volume_quantiles")
    expected_sites = _positive_integer(
        analysis["expected_holdout_sites"], "expected_holdout_sites"
    )
    if quantiles < 2:
        raise ValueError("volume_quantiles must be at least two")
    confidence = float(analysis["confidence_level"])
    if not np.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence_level must be finite and strictly between zero and one")
    if analysis["private_outputs_only"] is not True:
        raise ValueError("The observed error analysis must retain private_outputs_only: true")
    primary = _specification(analysis["primary_specification"], "primary_specification")
    comparisons = analysis["comparisons"]
    if not isinstance(comparisons, list) or not comparisons:
        raise ValueError("comparisons must be a non-empty list")
    normalized: list[dict[str, Any]] = []
    identifiers: set[str] = set()
    for index, comparison in enumerate(comparisons):
        if not isinstance(comparison, dict):
            raise TypeError(f"comparisons[{index}] must be a mapping")
        identifier = str(comparison.get("id", "")).strip()
        if not identifier or identifier in identifiers:
            raise ValueError("comparison IDs must be non-empty and unique")
        identifiers.add(identifier)
        left = _specification(comparison.get("left"), f"comparisons[{index}].left")
        right = _specification(comparison.get("right"), f"comparisons[{index}].right")
        if left == right:
            raise ValueError(f"Comparison {identifier!r} has identical specifications")
        if {
            left["information_budget"],
            right["information_budget"],
        } != {"measurement_free_estimation"}:
            raise ValueError(
                "Paired comparisons must remain within the measurement-free information budget"
            )
        normalized.append({"id": identifier, "left": left, "right": right})
    result = dict(analysis)
    result.update(
        {
            "bootstrap_seed": seed,
            "bootstrap_replicates": replicates,
            "confidence_level": confidence,
            "volume_quantiles": quantiles,
            "expected_holdout_sites": expected_sites,
            "primary_specification": primary,
            "comparisons": normalized,
        }
    )
    return result


def _require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing columns: {', '.join(missing)}")


def _validated_predictions(frame: pd.DataFrame, name: str = "predictions") -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError(f"{name} must be a non-empty table")
    _require_columns(frame, _PREDICTION_COLUMNS, name)
    result = frame.copy()
    for column in ("observation_id", "inferred_site_id", "model", "feature_set"):
        if result[column].isna().any() or result[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"{name} contains missing or empty {column} values")
        result[column] = result[column].astype(str)
    for column in ("observed_vehicle_total", "prediction"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
        if (
            result[column].isna().any()
            or not np.isfinite(result[column]).all()
            or (result[column] < 0).any()
        ):
            raise ValueError(f"{name} contains invalid non-negative values in {column}")
    result["residual_prediction_minus_observed"] = (
        result["prediction"] - result["observed_vehicle_total"]
    )
    return result


def _spec_mask(frame: pd.DataFrame, specification: Mapping[str, str]) -> pd.Series:
    return (
        frame["model"].eq(specification["model"])
        & frame["feature_set"].eq(specification["feature_set"])
        & frame["information_budget"].eq(specification["information_budget"])
    )


def _spec_label(specification: Mapping[str, str]) -> str:
    return f"{specification['model']} / {specification['feature_set']}"


def paired_prediction_rows(
    predictions: pd.DataFrame,
    left: Mapping[str, str],
    right: Mapping[str, str],
) -> pd.DataFrame:
    """Align two specifications on identical holdout observations and sites."""

    frame = _validated_predictions(predictions)
    left_spec = _specification(dict(left), "left specification")
    right_spec = _specification(dict(right), "right specification")
    selected: list[pd.DataFrame] = []
    for side, specification in (("left", left_spec), ("right", right_spec)):
        subset = frame.loc[
            _spec_mask(frame, specification),
            [
                "observation_id",
                "inferred_site_id",
                "observed_vehicle_total",
                "prediction",
            ],
        ].copy()
        if subset.empty:
            raise ValueError(f"No predictions match the {side} specification")
        if subset["observation_id"].duplicated().any():
            raise ValueError(f"The {side} specification contains duplicate observation IDs")
        subset = subset.rename(
            columns={
                "inferred_site_id": f"inferred_site_id_{side}",
                "observed_vehicle_total": f"observed_vehicle_total_{side}",
                "prediction": f"prediction_{side}",
            }
        )
        selected.append(subset)
    paired = selected[0].merge(
        selected[1], on="observation_id", how="outer", validate="one_to_one", indicator=True
    )
    if not paired["_merge"].eq("both").all():
        raise ValueError("Paired specifications do not cover identical observation IDs")
    if not paired["inferred_site_id_left"].eq(paired["inferred_site_id_right"]).all():
        raise ValueError("Paired specifications disagree on inferred-site membership")
    if not np.allclose(
        paired["observed_vehicle_total_left"],
        paired["observed_vehicle_total_right"],
        atol=0,
        rtol=0,
    ):
        raise ValueError("Paired specifications disagree on observed targets")
    result = paired.rename(
        columns={
            "inferred_site_id_left": "inferred_site_id",
            "observed_vehicle_total_left": "observed_vehicle_total",
        }
    ).drop(
        columns=[
            "inferred_site_id_right",
            "observed_vehicle_total_right",
            "_merge",
        ]
    )
    result["residual_left"] = result["prediction_left"] - result["observed_vehicle_total"]
    result["residual_right"] = result["prediction_right"] - result["observed_vehicle_total"]
    result["absolute_error_left"] = result["residual_left"].abs()
    result["absolute_error_right"] = result["residual_right"].abs()
    result["squared_error_left"] = result["residual_left"] ** 2
    result["squared_error_right"] = result["residual_right"] ** 2
    return result.sort_values("observation_id", kind="mergesort").reset_index(drop=True)


def _anonymous_site_map(values: Sequence[Any]) -> dict[str, str]:
    import hashlib

    unique = sorted({str(value) for value in values}, key=lambda value: hashlib.sha256(value.encode()).hexdigest())
    return {value: f"site_{index:02d}" for index, value in enumerate(unique, start=1)}


def _anonymize_sites(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    mapping = _anonymous_site_map(result["inferred_site_id"].astype(str))
    result.insert(0, "anonymous_site", result["inferred_site_id"].astype(str).map(mapping))
    return result.drop(columns="inferred_site_id")


def site_error_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    """Calculate per-site error metrics without treating intervals as independent sites."""

    frame = _validated_predictions(predictions)
    frame["absolute_error"] = frame["residual_prediction_minus_observed"].abs()
    frame["squared_error"] = frame["residual_prediction_minus_observed"] ** 2
    summary = (
        frame.groupby(
            [
                "protocol",
                "model",
                "feature_set",
                "information_budget",
                "inferred_site_id",
            ],
            observed=True,
            as_index=False,
        )
        .agg(
            rows=("observation_id", "size"),
            observed_mean=("observed_vehicle_total", "mean"),
            observed_max=("observed_vehicle_total", "max"),
            mae=("absolute_error", "mean"),
            rmse_squared=("squared_error", "mean"),
            mean_bias=("residual_prediction_minus_observed", "mean"),
            underprediction_rate=("residual_prediction_minus_observed", lambda x: float((x < 0).mean())),
            overprediction_rate=("residual_prediction_minus_observed", lambda x: float((x > 0).mean())),
        )
        .sort_values(["model", "feature_set", "inferred_site_id"], kind="mergesort")
        .reset_index(drop=True)
    )
    summary["rmse"] = np.sqrt(summary.pop("rmse_squared"))
    return summary


def _observation_quantiles(predictions: pd.DataFrame, count: int) -> pd.DataFrame:
    frame = _validated_predictions(predictions)
    target_counts = frame.groupby("observation_id", observed=True).agg(
        target_values=("observed_vehicle_total", "nunique"),
        site_values=("inferred_site_id", "nunique"),
        observed_vehicle_total=("observed_vehicle_total", "first"),
        inferred_site_id=("inferred_site_id", "first"),
    )
    if (target_counts[["target_values", "site_values"]] != 1).any().any():
        raise ValueError("Prediction specifications disagree on observation targets or sites")
    if target_counts["observed_vehicle_total"].nunique() < 2:
        raise ValueError("Observed-volume quantiles require at least two distinct totals")
    codes = pd.qcut(
        target_counts["observed_vehicle_total"],
        q=count,
        labels=False,
        duplicates="drop",
    )
    actual_count = int(codes.nunique())
    if codes.isna().any() or actual_count != count:
        raise ValueError(
            "Observed-volume distribution cannot support the requested "
            f"{count} distinct quantiles"
        )
    bin_count = int(codes.max()) + 1
    label_width = len(str(bin_count))
    target_counts["volume_quantile"] = codes.astype(int).map(
        lambda value: f"Q{value + 1:0{label_width}d}_of_{bin_count}"
    )
    return target_counts.reset_index()[
        ["observation_id", "inferred_site_id", "observed_vehicle_total", "volume_quantile"]
    ]


def volume_quantile_summary(predictions: pd.DataFrame, quantiles: int) -> pd.DataFrame:
    """Summarize residual direction and magnitude over target-derived volume strata."""

    frame = _validated_predictions(predictions)
    assignments = _observation_quantiles(frame, quantiles)
    merged = frame.merge(
        assignments.drop(columns=["inferred_site_id", "observed_vehicle_total"]),
        on="observation_id",
        validate="many_to_one",
    )
    merged["absolute_error"] = merged["residual_prediction_minus_observed"].abs()
    merged["squared_error"] = merged["residual_prediction_minus_observed"] ** 2
    summary = (
        merged.groupby(
            [
                "protocol",
                "model",
                "feature_set",
                "information_budget",
                "volume_quantile",
            ],
            observed=True,
            as_index=False,
        )
        .agg(
            rows=("observation_id", "size"),
            inferred_sites=("inferred_site_id", "nunique"),
            observed_min=("observed_vehicle_total", "min"),
            observed_max=("observed_vehicle_total", "max"),
            observed_mean=("observed_vehicle_total", "mean"),
            mae=("absolute_error", "mean"),
            rmse_squared=("squared_error", "mean"),
            mean_bias=("residual_prediction_minus_observed", "mean"),
            underprediction_rate=("residual_prediction_minus_observed", lambda x: float((x < 0).mean())),
            overprediction_rate=("residual_prediction_minus_observed", lambda x: float((x > 0).mean())),
        )
        .sort_values(["model", "feature_set", "volume_quantile"], kind="mergesort")
        .reset_index(drop=True)
    )
    summary["rmse"] = np.sqrt(summary.pop("rmse_squared"))
    return summary


def _paired_point_metrics(paired: pd.DataFrame) -> dict[str, float | int]:
    if paired.empty:
        raise ValueError("Paired comparison contains no rows")
    left_mae = float(paired["absolute_error_left"].mean())
    right_mae = float(paired["absolute_error_right"].mean())
    left_rmse = float(np.sqrt(paired["squared_error_left"].mean()))
    right_rmse = float(np.sqrt(paired["squared_error_right"].mean()))
    return {
        "rows": len(paired),
        "inferred_sites": int(paired["inferred_site_id"].nunique()),
        "left_mae": left_mae,
        "right_mae": right_mae,
        "delta_mae_left_minus_right": left_mae - right_mae,
        "left_rmse": left_rmse,
        "right_rmse": right_rmse,
        "delta_rmse_left_minus_right": left_rmse - right_rmse,
    }


def paired_site_bootstrap(
    paired: pd.DataFrame,
    *,
    replicates: int,
    seed: int,
    confidence_level: float,
) -> tuple[dict[str, float | int], pd.DataFrame]:
    """Bootstrap paired error differences by resampling whole inferred sites."""

    replicates = _positive_integer(replicates, "replicates")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if not np.isfinite(confidence_level) or not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be strictly between zero and one")
    required = {
        "inferred_site_id",
        "absolute_error_left",
        "absolute_error_right",
        "squared_error_left",
        "squared_error_right",
    }
    _require_columns(paired, required, "paired predictions")
    if paired.empty:
        raise ValueError("Paired predictions must not be empty")
    numeric = list(required - {"inferred_site_id"})
    if paired[numeric].isna().any().any() or not np.isfinite(paired[numeric]).all().all():
        raise ValueError("Paired predictions contain invalid error values")
    grouped = (
        paired.groupby("inferred_site_id", observed=True)
        .agg(
            rows=("inferred_site_id", "size"),
            absolute_left=("absolute_error_left", "sum"),
            absolute_right=("absolute_error_right", "sum"),
            squared_left=("squared_error_left", "sum"),
            squared_right=("squared_error_right", "sum"),
        )
        .sort_index()
    )
    if len(grouped) < 2:
        raise ValueError("Whole-site bootstrap requires at least two inferred sites")
    generator = np.random.default_rng(seed)
    sampled = generator.integers(0, len(grouped), size=(replicates, len(grouped)))
    rows = grouped["rows"].to_numpy(float)[sampled].sum(axis=1)
    absolute_left = grouped["absolute_left"].to_numpy(float)[sampled].sum(axis=1)
    absolute_right = grouped["absolute_right"].to_numpy(float)[sampled].sum(axis=1)
    squared_left = grouped["squared_left"].to_numpy(float)[sampled].sum(axis=1)
    squared_right = grouped["squared_right"].to_numpy(float)[sampled].sum(axis=1)
    delta_mae = absolute_left / rows - absolute_right / rows
    delta_rmse = np.sqrt(squared_left / rows) - np.sqrt(squared_right / rows)
    samples = pd.DataFrame(
        {
            "replicate": np.arange(replicates, dtype=int),
            "delta_mae_left_minus_right": delta_mae,
            "delta_rmse_left_minus_right": delta_rmse,
        }
    )
    alpha = (1.0 - confidence_level) / 2.0
    point = _paired_point_metrics(paired)
    point.update(
        {
            "bootstrap_replicates": replicates,
            "bootstrap_seed": seed,
            "confidence_level": confidence_level,
            "delta_mae_ci_low": float(np.quantile(delta_mae, alpha)),
            "delta_mae_ci_high": float(np.quantile(delta_mae, 1.0 - alpha)),
            "delta_rmse_ci_low": float(np.quantile(delta_rmse, alpha)),
            "delta_rmse_ci_high": float(np.quantile(delta_rmse, 1.0 - alpha)),
        }
    )
    return point, samples


def leave_one_site_out_influence(paired: pd.DataFrame) -> pd.DataFrame:
    """Measure how each held-out site changes the paired error difference."""

    _require_columns(
        paired,
        {
            "inferred_site_id",
            "absolute_error_left",
            "absolute_error_right",
            "squared_error_left",
            "squared_error_right",
        },
        "paired predictions",
    )
    sites = sorted(paired["inferred_site_id"].astype(str).unique())
    if len(sites) < 2:
        raise ValueError("Leave-one-site-out influence requires at least two inferred sites")
    full = _paired_point_metrics(paired)
    rows: list[dict[str, Any]] = []
    for site in sites:
        subset = paired.loc[paired["inferred_site_id"].astype(str) != site]
        metric = _paired_point_metrics(subset)
        rows.append(
            {
                "inferred_site_id": site,
                "remaining_rows": metric["rows"],
                "remaining_sites": metric["inferred_sites"],
                "delta_mae_without_site": metric["delta_mae_left_minus_right"],
                "delta_rmse_without_site": metric["delta_rmse_left_minus_right"],
                "mae_influence": metric["delta_mae_left_minus_right"]
                - full["delta_mae_left_minus_right"],
                "rmse_influence": metric["delta_rmse_left_minus_right"]
                - full["delta_rmse_left_minus_right"],
            }
        )
    return pd.DataFrame(rows)


def cv_holdout_comparison(
    cv_metrics: pd.DataFrame, holdout_metrics: pd.DataFrame
) -> pd.DataFrame:
    """Compare training-only grouped-CV summaries with the untouched holdout."""

    _require_columns(
        cv_metrics,
        {"fold", "model", "feature_set", "information_budget", "mae", "rmse"},
        "CV metrics",
    )
    _require_columns(
        holdout_metrics,
        {"model", "feature_set", "information_budget", "mae", "rmse"},
        "holdout metrics",
    )
    development = cv_metrics.loc[
        cv_metrics["information_budget"] == "measurement_free_estimation"
    ]
    holdout = holdout_metrics.loc[
        holdout_metrics["information_budget"] == "measurement_free_estimation"
    ]
    summary = development.groupby(["model", "feature_set"], observed=True, as_index=False).agg(
        cv_folds=("fold", "nunique"),
        cv_mean_mae=("mae", "mean"),
        cv_sd_mae=("mae", "std"),
        cv_mean_rmse=("rmse", "mean"),
        cv_sd_rmse=("rmse", "std"),
    )
    if summary.empty or not summary["cv_folds"].eq(5).all():
        raise ValueError("Training-only diagnostics require five grouped CV folds per model")
    if holdout.duplicated(["model", "feature_set"]).any():
        raise ValueError("Holdout metrics contain duplicated model specifications")
    result = summary.merge(
        holdout[["model", "feature_set", "mae", "rmse"]].rename(
            columns={"mae": "holdout_mae", "rmse": "holdout_rmse"}
        ),
        on=["model", "feature_set"],
        how="inner",
        validate="one_to_one",
    )
    if len(result) != len(summary):
        raise ValueError("CV and holdout specifications are not aligned")
    result["holdout_minus_cv_mae"] = result["holdout_mae"] - result["cv_mean_mae"]
    result["holdout_minus_cv_rmse"] = result["holdout_rmse"] - result["cv_mean_rmse"]
    return result.sort_values(["cv_mean_mae", "model", "feature_set"], kind="mergesort")


def _persistence_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    frame = _validated_predictions(predictions)
    persistence = frame.loc[frame["information_budget"] == "online_forecast_with_previous_measurement"]
    if persistence.empty or set(persistence["model"]) != {"persistence"}:
        raise ValueError("Validated predictions lack the separate online persistence diagnostic")
    if persistence["observation_id"].duplicated().any():
        raise ValueError("Persistence predictions contain duplicated observations")
    residual = persistence["residual_prediction_minus_observed"]
    return pd.DataFrame(
        [
            {
                "model": "persistence",
                "information_budget": "online_forecast_with_previous_measurement",
                "rows": len(persistence),
                "inferred_sites": int(persistence["inferred_site_id"].nunique()),
                "mae": float(residual.abs().mean()),
                "rmse": float(np.sqrt(np.square(residual).mean())),
                "mean_bias": float(residual.mean()),
                "interpretation": (
                    "Separate online diagnostic using the immediately previous measured interval; "
                    "not a measurement-free model comparison."
                ),
            }
        ]
    )


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False, lineterminator="\n", float_format="%.12g")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _git_state(root: Path) -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "unavailable", "tracked_worktree_dirty": None}
    return {"commit": commit, "tracked_worktree_dirty": dirty}


def _write_figures(
    directory: Path,
    site_errors: pd.DataFrame,
    volume_errors: pd.DataFrame,
    paired: pd.DataFrame,
    cv_holdout: pd.DataFrame,
    primary_specification: Mapping[str, str],
) -> list[Path]:
    import matplotlib.pyplot as plt

    figures = directory / "figures"
    figures.mkdir()
    paths: list[Path] = []

    measurement_free = site_errors.loc[
        site_errors["information_budget"] == "measurement_free_estimation"
    ].copy()
    measurement_free["specification"] = (
        measurement_free["model"] + " / " + measurement_free["feature_set"]
    )
    selected_labels = set(paired["left_label"]) | set(paired["right_label"])
    plotted = measurement_free.loc[measurement_free["specification"].isin(selected_labels)]
    figure, axis = plt.subplots(figsize=(10.5, 5.4), constrained_layout=True)
    for label, group in plotted.groupby("specification", sort=True):
        ordered = group.sort_values("anonymous_site")
        axis.scatter(ordered["anonymous_site"], ordered["mae"], s=45, label=label)
    axis.set(
        title="Measurement-free holdout MAE by anonymous inferred site",
        xlabel="Anonymous inferred site (ten-site holdout)",
        ylabel="MAE (vehicles per 20-minute interval)",
    )
    axis.tick_params(axis="x", labelrotation=45)
    axis.grid(alpha=0.2)
    axis.legend(frameon=False, fontsize=8)
    path = figures / "site_mae_private.png"
    figure.savefig(path, dpi=170)
    plt.close(figure)
    paths.append(path)

    primary = volume_errors.loc[
        (volume_errors["model"] == primary_specification["model"])
        & (volume_errors["feature_set"] == primary_specification["feature_set"])
        & (
            volume_errors["information_budget"]
            == primary_specification["information_budget"]
        )
    ].sort_values("volume_quantile")
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    axes[0].bar(primary["volume_quantile"], primary["mae"], color="#2b6f8a")
    axes[1].bar(primary["volume_quantile"], primary["mean_bias"], color="#b45f06")
    axes[0].set(title="Absolute error by observed-volume stratum", ylabel="MAE")
    axes[1].set(
        title="Prediction bias by observed-volume stratum",
        ylabel="Mean prediction minus observation",
    )
    for axis in axes:
        axis.tick_params(axis="x", labelrotation=25)
        axis.grid(axis="y", alpha=0.2)
    path = figures / "volume_error_private.png"
    figure.savefig(path, dpi=170)
    plt.close(figure)
    paths.append(path)

    figure, axis = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    positions = np.arange(len(paired))
    values = paired["delta_mae_left_minus_right"].to_numpy(float)
    lower = values - paired["delta_mae_ci_low"].to_numpy(float)
    upper = paired["delta_mae_ci_high"].to_numpy(float) - values
    axis.errorbar(positions, values, yerr=np.vstack([lower, upper]), fmt="o", capsize=4)
    axis.axhline(0, color="black", linewidth=1)
    axis.set(
        title="Paired whole-site bootstrap: MAE difference",
        ylabel="Left minus right MAE (negative favors left)",
        xticks=positions,
        xticklabels=(
            paired["comparison_id"]
            .str.replace("_vs_", "\nvs\n", regex=False)
            .str.replace("_", " ", regex=False)
        ),
    )
    axis.grid(axis="y", alpha=0.2)
    path = figures / "paired_uncertainty_private.png"
    figure.savefig(path, dpi=170)
    plt.close(figure)
    paths.append(path)

    ordered = cv_holdout.sort_values("cv_mean_mae", kind="mergesort")
    labels = (
        ordered["model"].str.replace("_", " ", regex=False)
        + "\n"
        + ordered["feature_set"].str.replace("_", " ", regex=False)
    )
    positions = np.arange(len(ordered))
    width = 0.38
    figure, axis = plt.subplots(figsize=(11, 5.2), constrained_layout=True)
    axis.bar(positions - width / 2, ordered["cv_mean_mae"], width, label="Training CV mean")
    axis.bar(positions + width / 2, ordered["holdout_mae"], width, label="Frozen holdout")
    axis.set(
        title="Training-only grouped CV and frozen joint holdout",
        ylabel="MAE (vehicles per 20-minute interval)",
        xticks=positions,
        xticklabels=labels,
    )
    axis.tick_params(axis="x", labelrotation=20)
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.2)
    path = figures / "cv_holdout_private.png"
    figure.savefig(path, dpi=170)
    plt.close(figure)
    paths.append(path)
    return paths


def _private_report(
    *,
    site_errors: pd.DataFrame,
    volume_errors: pd.DataFrame,
    paired: pd.DataFrame,
    influence: pd.DataFrame,
    cv_holdout: pd.DataFrame,
    persistence: pd.DataFrame,
    primary_specification: Mapping[str, str],
) -> str:
    primary_sites = site_errors.loc[
        (site_errors["model"] == primary_specification["model"])
        & (site_errors["feature_set"] == primary_specification["feature_set"])
        & (
            site_errors["information_budget"]
            == primary_specification["information_budget"]
        )
    ].sort_values("mae", ascending=False)
    primary_volume = volume_errors.loc[
        (volume_errors["model"] == primary_specification["model"])
        & (volume_errors["feature_set"] == primary_specification["feature_set"])
        & (
            volume_errors["information_budget"]
            == primary_specification["information_budget"]
        )
    ].sort_values("volume_quantile")
    lowest = primary_volume.iloc[0]
    highest = primary_volume.iloc[-1]
    selected = cv_holdout.loc[
        (cv_holdout["model"] == primary_specification["model"])
        & (cv_holdout["feature_set"] == primary_specification["feature_set"])
    ].iloc[0]
    lines = [
        "# Private observed-traffic error analysis",
        "",
        "> Redistribution status: private derived evidence. Do not publish without explicit written authorization covering derived outputs.",
        "",
        "## Scope",
        "",
        "This is a secondary exploratory analysis of the unchanged frozen joint future-date/unseen-inferred-site holdout. It does not reselect the primary model, refit any estimator, or establish separate temporal and geographic generalization.",
        "",
        "## Error heterogeneity",
        "",
        f"The primary training-CV-selected specification ({_spec_label(primary_specification)}) has site MAE ranging from {primary_sites['mae'].min():.2f} to {primary_sites['mae'].max():.2f} vehicles per 20-minute interval across {len(primary_sites)} anonymous held-out sites.",
        f"In the lowest observed-volume stratum ({lowest['volume_quantile']}), its mean bias is {lowest['mean_bias']:.2f} and overprediction rate is {100 * lowest['overprediction_rate']:.1f}%.",
        f"In the highest observed-volume stratum ({highest['volume_quantile']}), its mean bias is {highest['mean_bias']:.2f} and underprediction rate is {100 * highest['underprediction_rate']:.1f}%.",
        "",
        "## Paired model comparisons",
        "",
        "All deltas are left minus right on identical holdout rows; negative MAE deltas favor the left specification. Confidence intervals resample whole inferred sites.",
        "",
    ]
    for row in paired.itertuples(index=False):
        lines.append(
            f"- `{row.comparison_id}`: ΔMAE {row.delta_mae_left_minus_right:.2f} "
            f"(site-bootstrap {100 * row.confidence_level:.0f}% interval "
            f"{row.delta_mae_ci_low:.2f} to {row.delta_mae_ci_high:.2f})."
        )
    maximum_influence = influence.iloc[influence["mae_influence"].abs().argmax()]
    lines.extend(
        [
            "",
            "## Influence and generalization limits",
            "",
            f"The largest absolute leave-one-site-out change in a paired MAE difference is {abs(maximum_influence['mae_influence']):.2f} vehicles per interval. With only ten held-out inferred sites, individual sites can materially affect rankings and intervals.",
            f"For the training-CV-selected specification, mean grouped-CV MAE is {selected['cv_mean_mae']:.2f}; frozen-holdout MAE is {selected['holdout_mae']:.2f}, a holdout-minus-CV difference of {selected['holdout_minus_cv_mae']:.2f}.",
            "Every directional segment was surveyed on one date, so date and site effects remain confounded. The holdout cannot support separate claims about future-date and unseen-site performance.",
            "",
            "## Online persistence",
            "",
            f"Exactly-adjacent one-step persistence covers {int(persistence.iloc[0]['rows'])} rows and has MAE {persistence.iloc[0]['mae']:.2f}. It uses the previous measured count and is therefore a separate online information budget, not a measurement-free competitor.",
            "",
            "## Interpretation restrictions",
            "",
            "No retrospective EO variables enter these predictions. Results are descriptive, not causal, and do not establish population-level Greater Cairo generalization. Site-level tables, derived figures, and numerical findings in this directory remain private pending explicit redistribution terms.",
            "",
        ]
    )
    return "\n".join(lines)


def validate_error_analysis_bundle(
    output_dir: str | Path,
    config_file: str | Path,
    *,
    upstream_provenance_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate an atomic private analysis bundle and its safe output contract."""

    directory = Path(output_dir).resolve()
    manifest_path = directory / "analysis_manifest.json"
    if not directory.is_dir() or not manifest_path.is_file():
        raise ObservedArtifactIntegrityError(f"Private error-analysis bundle is missing: {directory}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ObservedArtifactIntegrityError("Private analysis manifest is invalid") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != ANALYSIS_SCHEMA:
        raise ObservedArtifactIntegrityError("Private analysis manifest schema is incompatible")
    if manifest.get("publication_status") != PRIVATE_STATUS:
        raise ObservedArtifactIntegrityError("Private analysis publication status is unsafe")
    if manifest.get("analysis_module_sha256") != sha256_file(Path(__file__)):
        raise ObservedArtifactIntegrityError(
            "Private analysis was generated by a different analytical implementation"
        )
    analysis_git = manifest.get("analysis_git")
    if not isinstance(analysis_git, dict):
        raise ObservedArtifactIntegrityError("Private analysis lacks code-revision provenance")
    analysis_commit = analysis_git.get("commit")
    analysis_dirty = analysis_git.get("tracked_worktree_dirty")
    if (
        not isinstance(analysis_commit, str)
        or not analysis_commit.strip()
        or not isinstance(analysis_dirty, bool)
        and analysis_dirty is not None
    ):
        raise ObservedArtifactIntegrityError(
            "Private analysis code-revision provenance is malformed"
        )
    configuration = manifest.get("configuration")
    if not isinstance(configuration, dict):
        raise ObservedArtifactIntegrityError("Private analysis lacks a configuration snapshot")
    try:
        validate_configuration_hash(
            config_file,
            manifest,
            mode="canonical_v1",
            embedded_configuration=configuration,
        )
        load_error_analysis_config(config_file)
    except (TypeError, ValueError) as exc:
        raise ObservedArtifactIntegrityError(
            f"Private analysis configuration provenance is invalid: {exc}"
        ) from exc
    if (
        upstream_provenance_sha256 is not None
        and manifest.get("upstream_provenance_sha256") != upstream_provenance_sha256
    ):
        raise ObservedArtifactIntegrityError("Private analysis is bound to another source bundle")
    hashes = manifest.get("artifact_hashes")
    if not isinstance(hashes, dict) or not hashes:
        raise ObservedArtifactIntegrityError("Private analysis has no artifact hashes")
    if missing := sorted(_PRIVATE_REQUIRED_FILES - set(hashes)):
        raise ObservedArtifactIntegrityError(
            "Private analysis omits required artifacts: " + ", ".join(missing)
        )
    for relative, expected in hashes.items():
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise ObservedArtifactIntegrityError("Private analysis artifact hash is malformed")
        path = (directory / relative).resolve()
        try:
            path.relative_to(directory)
        except ValueError as exc:
            raise ObservedArtifactIntegrityError(
                f"Private analysis artifact path escapes its bundle: {relative}"
            ) from exc
        if not path.is_file() or sha256_file(path) != expected:
            raise ObservedArtifactIntegrityError(
                f"Private analysis artifact is missing or hash-mismatched: {relative}"
            )
        if path.suffix == ".csv":
            try:
                columns = set(pd.read_csv(path, nrows=0).columns)
            except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
                raise ObservedArtifactIntegrityError(
                    f"Private analysis CSV is invalid: {relative}"
                ) from exc
            if columns & _IDENTIFIER_COLUMNS:
                raise ObservedArtifactIntegrityError(
                    f"Private analysis CSV leaks source identifiers: {relative}"
                )
    return {"manifest": manifest, "artifact_hashes": dict(hashes), "output_dir": directory}


def run_observed_error_analysis(
    project_root: str | Path,
    bundle_dir: str | Path,
    output_dir: str | Path,
    config_file: str | Path,
    *,
    validation_root: str | Path | None = None,
    source_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Generate an atomic private diagnostic bundle from validated predictions."""

    root = Path(project_root).resolve()
    bundle = Path(bundle_dir).resolve()
    destination = Path(output_dir).resolve()
    config_path = Path(config_file).resolve()
    try:
        relative_destination = destination.relative_to(root)
    except ValueError as exc:
        raise ValueError("Private error-analysis output must remain inside the project root") from exc
    if relative_destination.parts[:2] != (".urbantwin", "private"):
        raise ValueError(
            "Private error-analysis output must remain under .urbantwin/private"
        )
    if destination.exists():
        raise FileExistsError(f"Private error-analysis output already exists: {destination}")
    configuration = load_yaml(config_path)
    protocol = load_error_analysis_config(config_path)
    strict_root = Path(validation_root).resolve() if validation_root is not None else root
    verified = validate_observed_artifacts(
        strict_root,
        bundle,
        source_dir=source_dir,
    )
    if verified is None:
        raise FileNotFoundError(f"Validated observed benchmark is missing: {bundle}")
    predictions = _validated_predictions(verified["predictions"])
    measurement_free = predictions.loc[
        predictions["information_budget"] == "measurement_free_estimation"
    ]
    holdout_sites = int(measurement_free["inferred_site_id"].nunique())
    if holdout_sites != protocol["expected_holdout_sites"]:
        raise ValueError(
            "Frozen holdout site count differs from the prespecified analysis: "
            f"{holdout_sites} != {protocol['expected_holdout_sites']}"
        )
    model_selection = verified["provenance"].get("model_selection")
    if not isinstance(model_selection, dict) or model_selection.get(
        "holdout_scores_used_for_selection"
    ) is not False:
        raise ValueError("Upstream model selection is not demonstrably training-only")
    primary = protocol["primary_specification"]
    if (
        model_selection.get("model") != primary["model"]
        or model_selection.get("feature_set") != primary["feature_set"]
    ):
        raise ValueError("Analysis primary specification differs from frozen CV selection")

    site_errors_raw = site_error_summary(predictions)
    site_errors = _anonymize_sites(site_errors_raw)
    volume_errors = volume_quantile_summary(predictions, protocol["volume_quantiles"])
    comparison_rows: list[dict[str, Any]] = []
    replicate_frames: list[pd.DataFrame] = []
    influence_frames: list[pd.DataFrame] = []
    for index, comparison in enumerate(protocol["comparisons"]):
        paired = paired_prediction_rows(predictions, comparison["left"], comparison["right"])
        point, samples = paired_site_bootstrap(
            paired,
            replicates=protocol["bootstrap_replicates"],
            seed=protocol["bootstrap_seed"] + index,
            confidence_level=protocol["confidence_level"],
        )
        comparison_rows.append(
            {
                "comparison_id": comparison["id"],
                "left_label": _spec_label(comparison["left"]),
                "right_label": _spec_label(comparison["right"]),
                "sign_convention": "left_minus_right_negative_favors_left",
                **point,
            }
        )
        samples.insert(0, "comparison_id", comparison["id"])
        replicate_frames.append(samples)
        influence = _anonymize_sites(leave_one_site_out_influence(paired))
        influence.insert(0, "comparison_id", comparison["id"])
        influence_frames.append(influence)
    comparisons = pd.DataFrame(comparison_rows)
    bootstrap_samples = pd.concat(replicate_frames, ignore_index=True)
    influence = pd.concat(influence_frames, ignore_index=True)
    cv_holdout = cv_holdout_comparison(verified["cv_metrics"], verified["metrics"])
    cv_holdout["selected_by_training_cv"] = (
        cv_holdout["model"].eq(model_selection["model"])
        & cv_holdout["feature_set"].eq(model_selection["feature_set"])
    )
    persistence = _persistence_summary(predictions)

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        _write_csv(staging / "site_errors_private.csv", site_errors)
        _write_csv(staging / "volume_quantile_errors_private.csv", volume_errors)
        _write_csv(staging / "paired_comparisons_private.csv", comparisons)
        _write_csv(
            staging / "paired_bootstrap_replicates_private.csv", bootstrap_samples
        )
        _write_csv(staging / "leave_one_site_out_private.csv", influence)
        _write_csv(staging / "cv_holdout_comparison_private.csv", cv_holdout)
        _write_csv(staging / "persistence_private.csv", persistence)
        (staging / "private_analysis_report.md").write_text(
            _private_report(
                site_errors=site_errors,
                volume_errors=volume_errors,
                paired=comparisons,
                influence=influence,
                cv_holdout=cv_holdout,
                persistence=persistence,
                primary_specification=primary,
            ),
            encoding="utf-8",
            newline="\n",
        )
        figure_paths = _write_figures(
            staging,
            site_errors,
            volume_errors,
            comparisons,
            cv_holdout,
            primary,
        )
        artifact_paths = sorted(path for path in staging.rglob("*") if path.is_file())
        upstream_provenance_path = bundle / "provenance.json"
        manifest = {
            "schema_version": ANALYSIS_SCHEMA,
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "data_kind": "derived_real_observed_traffic_diagnostics",
            "scientific_role": "secondary_exploratory_error_analysis",
            "publication_status": PRIVATE_STATUS,
            "configuration": configuration,
            **configuration_hash_record(config_path, configuration),
            "analysis_git": _git_state(root),
            "analysis_module_sha256": sha256_file(Path(__file__)),
            "upstream_provenance_sha256": sha256_file(upstream_provenance_path),
            "upstream_provenance_schema": verified["provenance"]["schema_version"],
            "upstream_git_commit": verified["provenance"]["git"]["commit"],
            "upstream_config_sha256": verified["provenance"]["config_sha256"],
            "upstream_split_manifest_sha256": verified["provenance"][
                "split_manifest_sha256"
            ],
            "upstream_artifact_hashes": {
                name: verified["hashes"][name]
                for name in (
                    "observed_predictions.csv",
                    "observed_cv_predictions.csv",
                    "observed_metrics.csv",
                    "observed_cv_metrics.csv",
                    "split_manifest.json",
                    "cv_split_manifest.json",
                )
            },
            "frozen_model_selection": model_selection,
            "holdout_rows": int(verified["split_manifest"]["test_rows"]),
            "holdout_inferred_sites": holdout_sites,
            "bootstrap_unit": "whole_inferred_site",
            "bootstrap_replicates": protocol["bootstrap_replicates"],
            "bootstrap_seed": protocol["bootstrap_seed"],
            "figure_files": [path.relative_to(staging).as_posix() for path in figure_paths],
            "artifact_hashes": {
                path.relative_to(staging).as_posix(): sha256_file(path)
                for path in artifact_paths
            },
            "interpretation_limits": [
                "The holdout jointly changes date and inferred site; their effects are confounded.",
                "Only ten inferred sites are available for holdout uncertainty.",
                "Holdout results do not reselect the training-CV-selected primary model.",
                "Persistence uses a previous measured interval and has a different information budget.",
                "No retrospective EO descriptor is a prediction feature.",
                "Derived outputs remain private pending explicit redistribution terms.",
            ],
        }
        _write_json(staging / "analysis_manifest.json", manifest)
        validate_error_analysis_bundle(
            staging,
            config_path,
            upstream_provenance_sha256=manifest["upstream_provenance_sha256"],
        )
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        "output_dir": destination,
        "site_errors": site_errors,
        "volume_errors": volume_errors,
        "paired_comparisons": comparisons,
        "leave_one_site_out": influence,
        "cv_holdout": cv_holdout,
        "persistence": persistence,
        "manifest": manifest,
    }


__all__ = [
    "ANALYSIS_SCHEMA",
    "CONFIG_SCHEMA",
    "cv_holdout_comparison",
    "leave_one_site_out_influence",
    "load_error_analysis_config",
    "paired_prediction_rows",
    "paired_site_bootstrap",
    "run_observed_error_analysis",
    "site_error_summary",
    "validate_error_analysis_bundle",
    "volume_quantile_summary",
]
