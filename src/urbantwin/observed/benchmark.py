"""Leakage-controlled benchmark for the measured 2024 TfC link counts."""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error, mean_poisson_deviance, median_absolute_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from urbantwin.common import load_yaml, sha256_file
from urbantwin.observed.features import (
    build_analysis_table,
    historic_eo_context,
    spatial_segment_features,
)

SCHEMA_VERSION = "urbantwin.observed-benchmark/v1"

_TIME_NUMERIC = ["time_sin", "time_cos"]
_TIME_CATEGORICAL = ["day_of_week"]
_SPATIAL_NUMERIC = [
    "geometry_length_m",
    "centroid_easting_m",
    "centroid_northing_m",
]
_SPATIAL_CATEGORICAL = ["direction_code", "highway", "oneway"]


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _git_state(root: Path) -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain=v1"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "unavailable", "dirty": "unknown"}
    return {"commit": commit, "dirty": bool(status.strip())}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def _recorded_input_path(path: Path, project_root: Path) -> str:
    """Record a portable repository-relative input path."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(project_root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(
            f"Observed-study inputs must remain within the project root: {resolved}"
        ) from exc


def _feature_columns(feature_set: str) -> tuple[list[str], list[str]]:
    if feature_set == "time_only":
        return list(_TIME_NUMERIC), list(_TIME_CATEGORICAL)
    if feature_set == "spatial_only":
        return list(_SPATIAL_NUMERIC), list(_SPATIAL_CATEGORICAL)
    if feature_set == "combined":
        return (
            [*_TIME_NUMERIC, *_SPATIAL_NUMERIC],
            [*_TIME_CATEGORICAL, *_SPATIAL_CATEGORICAL],
        )
    raise ValueError(f"Unsupported observed feature set: {feature_set}")


def _pipeline(model: str, feature_set: str, config: dict[str, Any], seed: int) -> Pipeline:
    numeric, categorical = _feature_columns(feature_set)
    preprocessor = ColumnTransformer(
        [
            ("numeric", StandardScaler(), numeric),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    if model == "poisson":
        estimator = PoissonRegressor(alpha=1.0, max_iter=1_000)
    elif model == "random_forest":
        estimator = RandomForestRegressor(
            n_estimators=int(config["random_forest_trees"]),
            min_samples_leaf=5,
            max_features=0.8,
            random_state=seed,
            n_jobs=1,
        )
    elif model == "histogram_gradient_boosting":
        estimator = HistGradientBoostingRegressor(
            loss="poisson",
            max_iter=int(config["histogram_gradient_boosting_iterations"]),
            learning_rate=0.05,
            l2_regularization=1.0,
            random_state=seed,
        )
    else:
        raise ValueError(f"Unsupported observed model: {model}")
    return Pipeline([("features", preprocessor), ("model", estimator)])


def _historical_mean(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    target = "observed_vehicle_total"
    by_slot = train.groupby(["day_of_week", "minute_of_day"], observed=True)[target].mean()
    by_minute = train.groupby("minute_of_day", observed=True)[target].mean()
    global_mean = float(train[target].mean())
    predictions = []
    for row in test[["day_of_week", "minute_of_day"]].itertuples(index=False):
        value = by_slot.get((row.day_of_week, row.minute_of_day), np.nan)
        if not np.isfinite(value):
            value = by_minute.get(row.minute_of_day, global_mean)
        predictions.append(float(value))
    return np.asarray(predictions, dtype=float)


def _metric_values(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    prediction = np.maximum(np.asarray(y_pred, dtype=float), 0.0)
    observed = np.asarray(y_true, dtype=float)
    residual = prediction - observed
    return {
        "mae": float(mean_absolute_error(observed, prediction)),
        "rmse": float(np.sqrt(np.mean(np.square(residual)))),
        "median_absolute_error": float(median_absolute_error(observed, prediction)),
        "mean_bias": float(np.mean(residual)),
        "mean_poisson_deviance": float(
            mean_poisson_deviance(observed, np.maximum(prediction, 1e-9))
        ),
    }


def _bootstrap_interval(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
    *,
    replicates: int,
    seed: int,
) -> tuple[float, float]:
    unique = np.asarray(sorted(set(groups.astype(str))))
    if len(unique) < 2:
        return math.nan, math.nan
    positions = {group: np.flatnonzero(groups.astype(str) == group) for group in unique}
    generator = np.random.default_rng(seed)
    values = []
    for _ in range(replicates):
        sampled = generator.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([positions[group] for group in sampled])
        values.append(metric(y_true[indices], y_pred[indices]))
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def _metric_record(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, float]:
    result = _metric_values(y_true, y_pred)
    mae_low, mae_high = _bootstrap_interval(
        y_true,
        y_pred,
        groups,
        lambda actual, estimate: float(mean_absolute_error(actual, estimate)),
        replicates=replicates,
        seed=seed,
    )
    rmse_low, rmse_high = _bootstrap_interval(
        y_true,
        y_pred,
        groups,
        lambda actual, estimate: float(np.sqrt(np.mean(np.square(estimate - actual)))),
        replicates=replicates,
        seed=seed + 1,
    )
    result.update(
        {
            "mae_ci95_low": mae_low,
            "mae_ci95_high": mae_high,
            "rmse_ci95_low": rmse_low,
            "rmse_ci95_high": rmse_high,
        }
    )
    return result


def _prediction_rows(
    test: pd.DataFrame,
    prediction: np.ndarray,
    *,
    protocol: str,
    model: str,
    feature_set: str,
    information_budget: str,
) -> pd.DataFrame:
    output = test[
        [
            "observation_id",
            "segment_id",
            "inferred_site_id",
            "observed_date",
            "interval_start_local",
            "direction_code",
            "observed_vehicle_total",
        ]
    ].copy()
    output["protocol"] = protocol
    output["model"] = model
    output["feature_set"] = feature_set
    output["information_budget"] = information_budget
    output["prediction"] = np.maximum(np.asarray(prediction, dtype=float), 0.0)
    output["residual_prediction_minus_observed"] = (
        output["prediction"] - output["observed_vehicle_total"]
    )
    return output


def _add_previous_interval(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.sort_values(
        ["segment_id", "observed_date", "minute_of_day", "observation_id"]
    ).copy()
    grouped = output.groupby(["segment_id", "observed_date"], sort=False, observed=True)
    output["previous_observed_total"] = grouped["observed_vehicle_total"].shift(1)
    output["previous_minute_of_day"] = grouped["minute_of_day"].shift(1)
    output["persistence_eligible"] = (
        output["minute_of_day"] - output["previous_minute_of_day"]
    ).eq(20) & output["previous_observed_total"].notna()
    return output


def _split_protocol(frame: pd.DataFrame, test_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    dates = sorted(pd.Timestamp(value) for value in frame["observed_date"].unique())
    if not 0 < test_fraction < 1:
        raise ValueError("chronological_test_date_fraction must be between zero and one")
    test_count = max(1, math.ceil(len(dates) * test_fraction))
    if len(dates) - test_count < 2:
        raise ValueError("Too few distinct dates for a chronological development/holdout split")
    test_dates = set(dates[-test_count:])
    train = frame.loc[~frame["observed_date"].isin(test_dates)].copy()
    test = frame.loc[frame["observed_date"].isin(test_dates)].copy()
    overlap = sorted(set(train["inferred_site_id"]) & set(test["inferred_site_id"]))
    if overlap:
        raise ValueError(
            "Frozen future-date holdout is not site-disjoint; overlapping inferred sites: "
            + ", ".join(overlap[:5])
        )
    manifest = {
        "protocol": "future_date_and_unseen_inferred_site_holdout",
        "warning": (
            "Date and inferred site are confounded because every directional segment was "
            "surveyed on one date; this is not an isolated temporal-generalization test."
        ),
        "train_dates": sorted(value.date().isoformat() for value in set(train["observed_date"])),
        "test_dates": sorted(value.date().isoformat() for value in set(test["observed_date"])),
        "train_rows": len(train),
        "test_rows": len(test),
        "train_inferred_sites": int(train["inferred_site_id"].nunique()),
        "test_inferred_sites": int(test["inferred_site_id"].nunique()),
        "site_overlap": overlap,
        "excluded_ambiguous_rows": int((~frame["eligible_for_spatial_benchmark"]).sum()),
    }
    return train, test, manifest


def _model_specs(config: dict[str, Any]) -> Iterable[tuple[str, str]]:
    models = [
        model
        for model in config["models"]
        if model in {"poisson", "random_forest", "histogram_gradient_boosting"}
    ]
    for model in models:
        for feature_set in config["feature_sets"]:
            yield model, feature_set


def _write_figures(
    output: Path,
    full_frame: pd.DataFrame,
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    segments,
    selected_segment_ids: list[str],
) -> list[Path]:
    import matplotlib.pyplot as plt

    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    selected = full_frame.loc[full_frame["segment_id"].isin(selected_segment_ids)].copy()
    if not selected.empty:
        figure, axis = plt.subplots(figsize=(10, 4.8))
        for direction, rows in selected.groupby("direction_code", sort=True):
            rows = rows.sort_values("minute_of_day")
            axis.plot(
                rows["minute_of_day"] / 60,
                rows["observed_vehicle_total"],
                label=f"Direction code {direction}",
            )
        axis.set(
            title="Measured 20-minute flow: selected 6th of October Bridge links",
            xlabel="Publisher clock time (timezone not specified)",
            ylabel="Observed vehicles per 20 minutes",
        )
        axis.legend()
        axis.grid(alpha=0.2)
        figure.tight_layout()
        path = figures / "selected_link_measured_flow.png"
        figure.savefig(path, dpi=180)
        plt.close(figure)
        paths.append(path)

    final_metrics = metrics.loc[
        (metrics["evaluation"] == "frozen_holdout")
        & (metrics["information_budget"] == "measurement_free_estimation")
    ].sort_values("mae")
    figure, axis = plt.subplots(figsize=(11, 5.5))
    labels = [f"{row.model}\n{row.feature_set}" for row in final_metrics.itertuples()]
    positions = np.arange(len(final_metrics))
    values = final_metrics["mae"].to_numpy(float)
    lower = values - final_metrics["mae_ci95_low"].to_numpy(float)
    upper = final_metrics["mae_ci95_high"].to_numpy(float) - values
    axis.bar(positions, values, color="#2b6f8a")
    axis.errorbar(positions, values, yerr=np.vstack([lower, upper]), fmt="none", color="black")
    axis.set_xticks(positions, labels, rotation=30, ha="right")
    axis.set(
        title="Frozen joint future-date/unseen-site holdout",
        ylabel="MAE (vehicles per 20-minute interval)",
    )
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    path = figures / "observed_model_comparison.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    paths.append(path)

    totals = full_frame.groupby("segment_id", observed=True)["observed_vehicle_total"].sum()
    mapped = segments.copy()
    mapped["measured_total"] = mapped["segment_id"].map(totals)
    figure, axis = plt.subplots(figsize=(9, 8))
    mapped.plot(
        ax=axis,
        column="measured_total",
        cmap="viridis",
        linewidth=1.5,
        legend=True,
        missing_kwds={"color": "lightgray", "label": "No count rows"},
    )
    mapped.loc[mapped["segment_id"].isin(selected_segment_ids)].plot(
        ax=axis, color="none", edgecolor="#d62728", linewidth=4
    )
    axis.set_title("TfC directional survey segments and measured vehicle totals")
    axis.set_axis_off()
    figure.tight_layout()
    path = figures / "survey_segments_measured_flow.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    paths.append(path)

    best = final_metrics.iloc[0]
    best_predictions = predictions.loc[
        (predictions["model"] == best["model"])
        & (predictions["feature_set"] == best["feature_set"])
    ]
    figure, axis = plt.subplots(figsize=(6.5, 6))
    axis.scatter(
        best_predictions["observed_vehicle_total"],
        best_predictions["prediction"],
        alpha=0.35,
        s=16,
    )
    maximum = float(
        max(best_predictions["observed_vehicle_total"].max(), best_predictions["prediction"].max())
    )
    axis.plot([0, maximum], [0, maximum], linestyle="--", color="black", linewidth=1)
    axis.set(
        title=f"Observed versus estimated: {best['model']} ({best['feature_set']})",
        xlabel="Observed vehicles per 20 minutes",
        ylabel="Estimated vehicles per 20 minutes",
    )
    axis.grid(alpha=0.2)
    figure.tight_layout()
    path = figures / "observed_vs_estimated.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    paths.append(path)

    persistence = predictions.loc[predictions["model"] == "persistence"]
    if not persistence.empty:
        figure, axis = plt.subplots(figsize=(6.5, 6))
        axis.scatter(
            persistence["observed_vehicle_total"],
            persistence["prediction"],
            alpha=0.35,
            s=16,
            color="#8a4f2b",
        )
        maximum = float(
            max(persistence["observed_vehicle_total"].max(), persistence["prediction"].max())
        )
        axis.plot([0, maximum], [0, maximum], linestyle="--", color="black", linewidth=1)
        axis.set(
            title="Online persistence: exactly adjacent 20-minute intervals",
            xlabel="Observed vehicles per 20 minutes",
            ylabel="Previous measured interval",
        )
        axis.grid(alpha=0.2)
        figure.tight_layout()
        path = figures / "online_persistence_diagnostic.png"
        figure.savefig(path, dpi=180)
        plt.close(figure)
        paths.append(path)
    return paths


def _write_eo_map(
    output: Path,
    eo_dir: Path,
    segments,
    selected_segment_ids: list[str],
) -> Path:
    import matplotlib.pyplot as plt
    from rasterio.transform import Affine, array_bounds

    provenance = _read_json(eo_dir / "provenance.json")
    with np.load(eo_dir / "eo_stack.npz", allow_pickle=False) as archive:
        bands = archive["bands"].copy()
        valid = archive["valid_mask"].astype(bool)
    red, nir = bands[2], bands[3]
    denominator = nir + red
    ndvi = np.divide(
        nir - red,
        denominator,
        out=np.full_like(nir, np.nan, dtype=np.float32),
        where=(denominator != 0) & valid,
    )
    transform = Affine(*provenance["transform"])
    west, south, east, north = array_bounds(ndvi.shape[0], ndvi.shape[1], transform)
    selected = segments.loc[segments["segment_id"].isin(selected_segment_ids)].to_crs(
        provenance["crs"]
    )
    figure, axis = plt.subplots(figsize=(9, 6.5))
    image = axis.imshow(
        ndvi,
        extent=[west, east, south, north],
        origin="upper",
        cmap="RdYlGn",
        vmin=-0.4,
        vmax=0.6,
    )
    selected.plot(ax=axis, color="none", edgecolor="black", linewidth=3)
    figure.colorbar(image, ax=axis, label="NDVI")
    reference_year = provenance.get("configuration", {}).get("reference_year", "recorded")
    axis.set(
        title=f"Selected observed links over real Sentinel-2 L2A {reference_year} NDVI context",
        xlabel="Easting (EPSG:32636, m)",
        ylabel="Northing (EPSG:32636, m)",
    )
    figure.tight_layout()
    path = output / "figures" / "selected_link_historic_eo_context.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def _write_road_map(
    output: Path,
    road_dir: Path,
    segments,
    selected_segment_ids: list[str],
) -> Path:
    import geopandas as gpd
    import matplotlib.pyplot as plt

    roads = gpd.read_file(road_dir / "edges_projected.geojson")
    selected = segments.loc[segments["segment_id"].isin(selected_segment_ids)].to_crs(roads.crs)
    figure, axis = plt.subplots(figsize=(9, 7))
    roads.plot(ax=axis, color="#9BA9B4", linewidth=0.55, alpha=0.7)
    selected.plot(ax=axis, color="#C33D2E", linewidth=4.0)
    axis.set(
        title="Current OSM drive network and selected TfC observed links",
        xlabel="Easting (EPSG:32636, m)",
        ylabel="Northing (EPSG:32636, m)",
    )
    axis.set_aspect("equal")
    figure.tight_layout()
    path = output / "figures" / "selected_link_current_osm_context.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def run_observed_benchmark(
    study_dir: str | Path,
    output_dir: str | Path,
    config_path: str | Path,
    *,
    eo_dir: str | Path | None = None,
    road_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Run the frozen observed-link benchmark and atomically save its local bundle."""
    study_dir = Path(study_dir).resolve()
    output_dir = Path(output_dir).resolve()
    config_path = Path(config_path).resolve()
    project_root = config_path.parents[1]
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite observed benchmark: {output_dir}")
    configuration = load_yaml(config_path)
    benchmark = configuration.get("benchmark")
    spatial = configuration.get("spatial")
    selection = configuration.get("site_selection")
    if not isinstance(benchmark, dict) or not isinstance(spatial, dict) or not isinstance(selection, dict):
        raise TypeError("Observed config requires benchmark, spatial, and site_selection mappings")
    source_manifest_path = study_dir / "source_manifest.json"
    source_manifest = _read_json(source_manifest_path)
    for role in ("traffic_counts", "survey_locations"):
        config_source = (configuration.get("sources") or {}).get(role, {})
        manifest_role = "counts" if role == "traffic_counts" else "locations"
        actual_hash = ((source_manifest.get("sources") or {}).get(manifest_role) or {}).get("sha256")
        if config_source.get("expected_sha256") != actual_hash:
            raise ValueError(f"Observed {role} hash does not match the frozen study config")
    road_context = None
    road_root = None
    if road_dir is not None:
        from urbantwin.observed.roads import validate_observed_road_bundle

        road_root = Path(road_dir).resolve()
        road_context = validate_observed_road_bundle(road_root)
        if road_context is None:
            raise FileNotFoundError(f"Observed road bundle is missing: {road_root}")
        if road_context.get("routable_graph") is not True:
            raise ValueError(
                "The primary observed study requires a directed graph; the PBF fallback is "
                "geometry-only and remains useful only as separate map context."
            )

    frame = build_analysis_table(
        study_dir,
        projected_crs=str(spatial["projected_crs"]),
        inferred_site_link_distance_m=float(spatial["inferred_site_link_distance_m"]),
    )
    eligible = frame.loc[frame["eligible_for_spatial_benchmark"]].copy()
    train, test, split_manifest = _split_protocol(
        eligible, float(benchmark["chronological_test_date_fraction"])
    )
    split_manifest["excluded_ambiguous_rows"] = len(frame) - len(eligible)
    split_manifest.update(
        {
            "schema_version": "urbantwin.observed-split/v1",
            "inferred_site_definition": (
                "Connected components of released survey-line geometries in EPSG:32636 "
                f"within {spatial['inferred_site_link_distance_m']} metres."
            ),
            "inferred_site_warning": spatial["inference_warning"],
            "training_observation_ids_sha256": _canonical_hash(sorted(train["observation_id"])),
            "test_observation_ids_sha256": _canonical_hash(sorted(test["observation_id"])),
            "leakage_controls": [
                "No current-period class count or total count is used as a model feature.",
                "No segment ID, inferred-site ID, OSM ID, road name, or exact date is a feature.",
                "Opposite-direction and duplicate survey records are not used as features.",
                "Five conflicting OSM joins are excluded from the primary benchmark.",
                "Persistence uses only an exactly adjacent previous 20-minute observation.",
            ],
        }
    )
    if train["inferred_site_id"].nunique() < 5:
        raise ValueError("At least five inferred training sites are required for grouped CV")

    seed = int(configuration["random_seed"])
    replicates = int(benchmark["bootstrap_replicates"])
    floor = float(benchmark.get("prediction_floor", 0.0))
    target = "observed_vehicle_total"
    metric_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    cv_rows: list[dict[str, Any]] = []

    grouped = GroupKFold(n_splits=5)
    for fold, (fit_index, validation_index) in enumerate(
        grouped.split(train, groups=train["inferred_site_id"]), start=1
    ):
        fit = train.iloc[fit_index]
        validation = train.iloc[validation_index]
        specifications = [("historical_mean", "time_only"), *_model_specs(benchmark)]
        for model, feature_set in specifications:
            if model == "historical_mean":
                estimate = _historical_mean(fit, validation)
            else:
                fitted = _pipeline(model, feature_set, benchmark, seed)
                fitted.fit(fit, fit[target])
                estimate = fitted.predict(validation)
            estimate = np.maximum(estimate, floor)
            cv_rows.append(
                {
                    "fold": fold,
                    "model": model,
                    "feature_set": feature_set,
                    "information_budget": "measurement_free_estimation",
                    "train_rows": len(fit),
                    "validation_rows": len(validation),
                    "train_inferred_sites": int(fit["inferred_site_id"].nunique()),
                    "validation_inferred_sites": int(
                        validation["inferred_site_id"].nunique()
                    ),
                    **_metric_values(validation[target].to_numpy(), estimate),
                }
            )

    specifications = [("historical_mean", "time_only"), *_model_specs(benchmark)]
    for model_number, (model, feature_set) in enumerate(specifications):
        if model == "historical_mean":
            prediction = _historical_mean(train, test)
        else:
            fitted = _pipeline(model, feature_set, benchmark, seed)
            fitted.fit(train, train[target])
            prediction = fitted.predict(test)
        prediction = np.maximum(prediction, floor)
        metric_rows.append(
            {
                "evaluation": "frozen_holdout",
                "protocol": "future_date_and_unseen_inferred_site_holdout",
                "model": model,
                "feature_set": feature_set,
                "information_budget": "measurement_free_estimation",
                "rows": len(test),
                "inferred_sites": int(test["inferred_site_id"].nunique()),
                **_metric_record(
                    test[target].to_numpy(),
                    prediction,
                    test["inferred_site_id"].astype(str).to_numpy(),
                    replicates=replicates,
                    seed=seed + model_number * 10,
                ),
            }
        )
        prediction_frames.append(
            _prediction_rows(
                test,
                prediction,
                protocol="future_date_and_unseen_inferred_site_holdout",
                model=model,
                feature_set=feature_set,
                information_budget="measurement_free_estimation",
            )
        )

    lagged = _add_previous_interval(test)
    persistence = lagged.loc[lagged["persistence_eligible"]].copy()
    if persistence.empty:
        raise ValueError("The frozen holdout has no exactly adjacent intervals for persistence")
    persistence_prediction = persistence["previous_observed_total"].to_numpy(float)
    metric_rows.append(
        {
            "evaluation": "frozen_holdout",
            "protocol": "online_one_step_exactly_adjacent_intervals",
            "model": "persistence",
            "feature_set": "previous_observed_interval",
            "information_budget": "online_forecast_with_previous_measurement",
            "rows": len(persistence),
            "inferred_sites": int(persistence["inferred_site_id"].nunique()),
            **_metric_record(
                persistence[target].to_numpy(),
                persistence_prediction,
                persistence["inferred_site_id"].astype(str).to_numpy(),
                replicates=replicates,
                seed=seed + 9_000,
            ),
        }
    )
    prediction_frames.append(
        _prediction_rows(
            persistence,
            persistence_prediction,
            protocol="online_one_step_exactly_adjacent_intervals",
            model="persistence",
            feature_set="previous_observed_interval",
            information_budget="online_forecast_with_previous_measurement",
        )
    )

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    cv_metrics = pd.DataFrame(cv_rows)
    site_errors = (
        predictions.assign(
            absolute_error=lambda value: value["residual_prediction_minus_observed"].abs(),
            squared_error=lambda value: value["residual_prediction_minus_observed"] ** 2,
        )
        .groupby(
            ["protocol", "model", "feature_set", "information_budget", "inferred_site_id"],
            observed=True,
            as_index=False,
        )
        .agg(
            rows=("observation_id", "size"),
            mae=("absolute_error", "mean"),
            rmse_squared=("squared_error", "mean"),
            mean_bias=("residual_prediction_minus_observed", "mean"),
        )
    )
    site_errors["rmse"] = np.sqrt(site_errors.pop("rmse_squared"))
    class_summary = pd.DataFrame(
        {
            "vehicle_class": source_manifest["normalization"]["vehicle_classes"],
            "observed_vehicles": [
                int(frame[column].sum())
                for column in source_manifest["normalization"]["vehicle_classes"]
            ],
        }
    )
    class_summary["share_of_observed_vehicles"] = (
        class_summary["observed_vehicles"] / class_summary["observed_vehicles"].sum()
    )
    characterization = (
        frame.groupby(["observed_date", "day_normalized"], observed=True, as_index=False)
        .agg(
            directional_intervals=("observation_id", "size"),
            directional_segments=("segment_id", "nunique"),
            inferred_sites=("inferred_site_id", "nunique"),
            observed_vehicles=("observed_vehicle_total", "sum"),
        )
        .sort_values("observed_date")
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        metrics.to_csv(staging / "observed_metrics.csv", index=False, lineterminator="\n")
        cv_metrics.to_csv(staging / "observed_cv_metrics.csv", index=False, lineterminator="\n")
        predictions.to_csv(
            staging / "observed_predictions.csv", index=False, lineterminator="\n"
        )
        site_errors.to_csv(staging / "observed_site_errors.csv", index=False, lineterminator="\n")
        class_summary.to_csv(
            staging / "observed_vehicle_classes.csv", index=False, lineterminator="\n"
        )
        characterization.to_csv(
            staging / "observed_characterization.csv", index=False, lineterminator="\n"
        )
        (staging / "split_manifest.json").write_text(
            json.dumps(split_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        segments = spatial_segment_features(
            study_dir,
            projected_crs=str(spatial["projected_crs"]),
            inferred_site_link_distance_m=float(spatial["inferred_site_link_distance_m"]),
        )
        segments.to_file(staging / "segment_features.geojson", driver="GeoJSON")
        eo_context: list[dict[str, Any]] = []
        eo_source_context = None
        if eo_dir is not None:
            eo_root = Path(eo_dir).resolve()
            eo_context = historic_eo_context(
                study_dir,
                eo_root,
                [str(value) for value in selection["selected_segment_ids"]],
            )
            eo_provenance_path = eo_root / "provenance.json"
            eo_stack_path = eo_root / "eo_stack.npz"
            eo_provenance = _read_json(eo_provenance_path)
            eo_source_context = {
                "input_directory": _recorded_input_path(eo_root, project_root),
                "data_kind": eo_provenance["data_kind"],
                "provenance_sha256": sha256_file(eo_provenance_path),
                "eo_stack_sha256": sha256_file(eo_stack_path),
                "sentinel_item_ids": eo_provenance["sentinel_stac_item_ids"],
                "worldcover_item_ids": eo_provenance["worldcover_stac_item_ids"],
                "worldcover_year": 2021,
                "interpretation": (
                    "Sentinel-2 L2A near-survey context; WorldCover 2021 historic reference, "
                    "not 2024 ground truth"
                ),
            }
        (staging / "historic_eo_context.json").write_text(
            json.dumps(eo_context, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        figure_paths = _write_figures(
            staging,
            frame,
            metrics,
            predictions,
            segments,
            [str(value) for value in selection["selected_segment_ids"]],
        )
        if eo_context:
            figure_paths.append(
                _write_eo_map(
                    staging,
                    eo_root,
                    segments,
                    [str(value) for value in selection["selected_segment_ids"]],
                )
            )
        if road_context is not None:
            figure_paths.append(
                _write_road_map(
                    staging,
                    road_root,
                    segments,
                    [str(value) for value in selection["selected_segment_ids"]],
                )
            )
        artifact_paths = sorted(
            path for path in staging.rglob("*") if path.is_file() and path.name != "provenance.json"
        )
        provenance = {
            "schema_version": SCHEMA_VERSION,
            "data_kind": "real_observed_traffic_analysis",
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "source_manifest_sha256": sha256_file(source_manifest_path),
            "source_counts_sha256": source_manifest["sources"]["counts"]["sha256"],
            "source_locations_sha256": source_manifest["sources"]["locations"]["sha256"],
            "config_sha256": sha256_file(config_path),
            "configuration": configuration,
            "git": _git_state(config_path.parents[1]),
            "split_manifest_sha256": sha256_file(staging / "split_manifest.json"),
            "license": source_manifest["license"],
            "source_labels": {
                "traffic": "measured TfC directional link counts (2024)",
                "geometry": "TfC-released OSM-derived survey lines (2026 OSM extract)",
                "current_osm": (
                    "current third-party OSM directed road graph for GIS context"
                    if road_context is not None
                    else "not included"
                ),
                "eo": (
                    "real Sentinel-2 L2A context with WorldCover 2021 historic reference"
                    if eo_context
                    else "not included"
                ),
                "simulation": "not run; scientific gate failed",
            },
            "eo_context": eo_source_context,
            "study_gate": configuration["scientific_gate"],
            "road_context": (
                {
                    "input_directory": _recorded_input_path(road_root, project_root),
                    "manifest_sha256": sha256_file(road_root / "road_manifest.json"),
                    "data_kind": road_context["data_kind"],
                    "source_kind": road_context["source_kind"],
                    "generated_utc": road_context["generated_utc"],
                    "nodes": road_context["nodes"],
                    "directed_edges": road_context["directed_edges"],
                    "connectivity": road_context["connectivity"],
                    "map_matching": road_context["map_matching"],
                    "simulation_readiness": road_context["simulation_readiness"],
                    "attribution": road_context["attribution"],
                    "license": road_context["license"],
                }
                if road_context is not None
                else None
            ),
            "artifact_hashes": {
                path.relative_to(staging).as_posix(): sha256_file(path) for path in artifact_paths
            },
            "figure_files": [path.relative_to(staging).as_posix() for path in figure_paths],
            "interpretation_limits": [
                split_manifest["warning"],
                "Inferred site groups are analytical clusters, not publisher location IDs.",
                "EO context is descriptive and cannot establish a causal traffic relationship.",
                "Current OSM context is third-party map data, not a 2024 traffic observation.",
                "No measured-demand signal-control SUMO experiment was run.",
            ],
        }
        (staging / "provenance.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(staging, output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return metrics


__all__ = ["SCHEMA_VERSION", "run_observed_benchmark"]
