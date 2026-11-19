"""Deterministic synthetic tests for private observed-traffic diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from urbantwin.common import sha256_file
from urbantwin.observed import error_analysis as analysis
from urbantwin.observed.artifacts import ObservedArtifactIntegrityError

RF_COMBINED = {
    "model": "random_forest",
    "feature_set": "combined",
    "information_budget": "measurement_free_estimation",
}
HGB_COMBINED = {
    "model": "histogram_gradient_boosting",
    "feature_set": "combined",
    "information_budget": "measurement_free_estimation",
}
RF_SPATIAL = {
    "model": "random_forest",
    "feature_set": "spatial_only",
    "information_budget": "measurement_free_estimation",
}
HISTORICAL = {
    "model": "historical_mean",
    "feature_set": "time_only",
    "information_budget": "measurement_free_estimation",
}


def _predictions(site_count: int = 3) -> pd.DataFrame:
    specifications = [
        (RF_COMBINED, np.array([2.0, 1.0, -3.0, -10.0])),
        (HGB_COMBINED, np.array([4.0, 3.0, -8.0, -20.0])),
        (RF_SPATIAL, np.array([3.0, 2.0, -5.0, -15.0])),
        (HISTORICAL, np.array([20.0, 15.0, -10.0, -30.0])),
    ]
    targets = np.array([10.0, 20.0, 100.0, 200.0])
    rows: list[dict] = []
    for site_index in range(site_count):
        site = f"restricted-site-{site_index}"
        for specification, residuals in specifications:
            for interval, (target, residual) in enumerate(zip(targets, residuals, strict=True)):
                rows.append(
                    {
                        "observation_id": f"observation-{site_index}-{interval}",
                        "inferred_site_id": site,
                        "observed_vehicle_total": target,
                        "protocol": "future_date_and_unseen_inferred_site_holdout",
                        **specification,
                        "prediction": target + residual,
                    }
                )
        for interval, target in enumerate(targets[1:], start=1):
            rows.append(
                {
                    "observation_id": f"observation-{site_index}-{interval}",
                    "inferred_site_id": site,
                    "observed_vehicle_total": target,
                    "protocol": "online_one_step_exactly_adjacent_intervals",
                    "model": "persistence",
                    "feature_set": "previous_observed_interval",
                    "information_budget": "online_forecast_with_previous_measurement",
                    "prediction": target - 1.0,
                }
            )
    return pd.DataFrame(rows)


def _holdout_metrics() -> pd.DataFrame:
    predictions = _predictions()
    rows = []
    for (model, feature_set, budget), group in predictions.groupby(
        ["model", "feature_set", "information_budget"], observed=True
    ):
        residual = group["prediction"] - group["observed_vehicle_total"]
        rows.append(
            {
                "model": model,
                "feature_set": feature_set,
                "information_budget": budget,
                "mae": residual.abs().mean(),
                "rmse": np.sqrt(np.square(residual).mean()),
            }
        )
    return pd.DataFrame(rows)


def _cv_metrics() -> pd.DataFrame:
    rows = []
    for specification, base in (
        (RF_COMBINED, 4.0),
        (HGB_COMBINED, 8.0),
        (RF_SPATIAL, 6.0),
        (HISTORICAL, 15.0),
    ):
        for fold in range(1, 6):
            rows.append(
                {
                    "fold": fold,
                    **specification,
                    "mae": base + fold,
                    "rmse": base + fold + 2.0,
                }
            )
    return pd.DataFrame(rows)


def _write_config(path: Path, *, expected_sites: int = 3) -> Path:
    value = {
        "schema_version": analysis.CONFIG_SCHEMA,
        "analysis": {
            "bootstrap_seed": 17,
            "bootstrap_replicates": 50,
            "confidence_level": 0.95,
            "volume_quantiles": 4,
            "expected_holdout_sites": expected_sites,
            "private_outputs_only": True,
            "primary_specification": RF_COMBINED,
            "comparisons": [
                {
                    "id": "rf_vs_hgb",
                    "left": RF_COMBINED,
                    "right": HGB_COMBINED,
                },
                {
                    "id": "rf_vs_spatial",
                    "left": RF_COMBINED,
                    "right": RF_SPATIAL,
                },
                {
                    "id": "rf_vs_historical",
                    "left": RF_COMBINED,
                    "right": HISTORICAL,
                },
            ],
        },
    }
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    return path


def _verified_bundle() -> dict:
    return {
        "predictions": _predictions(),
        "metrics": _holdout_metrics(),
        "cv_metrics": _cv_metrics(),
        "provenance": {
            "schema_version": "urbantwin.observed-benchmark/v2",
            "git": {"commit": "historical-revision"},
            "config_sha256": "a" * 64,
            "model_selection": {
                "model": "random_forest",
                "feature_set": "combined",
                "holdout_scores_used_for_selection": False,
            },
            "split_manifest_sha256": "b" * 64,
        },
        "split_manifest": {"test_rows": 12},
        "hashes": {
            "observed_predictions.csv": "1" * 64,
            "observed_cv_predictions.csv": "2" * 64,
            "observed_metrics.csv": "3" * 64,
            "observed_cv_metrics.csv": "4" * 64,
            "split_manifest.json": "5" * 64,
            "cv_split_manifest.json": "6" * 64,
        },
    }


def test_paired_rows_are_identical_and_error_sign_is_explicit() -> None:
    paired = analysis.paired_prediction_rows(_predictions(), RF_COMBINED, HGB_COMBINED)

    assert len(paired) == 12
    assert paired["observation_id"].is_unique
    assert paired["inferred_site_id"].nunique() == 3
    assert (paired["absolute_error_left"] < paired["absolute_error_right"]).all()
    assert paired["absolute_error_left"].mean() == pytest.approx(4.0)
    assert paired["absolute_error_right"].mean() == pytest.approx(8.75)


@pytest.mark.parametrize("damage, message", [
    ("missing", "identical observation IDs"),
    ("duplicate", "duplicate observation IDs"),
    ("target", "disagree on observed targets"),
    ("site", "disagree on inferred-site membership"),
])
def test_paired_rows_reject_misaligned_or_ambiguous_records(damage: str, message: str) -> None:
    predictions = _predictions()
    mask = (
        predictions["model"].eq(HGB_COMBINED["model"])
        & predictions["feature_set"].eq(HGB_COMBINED["feature_set"])
    )
    index = predictions.index[mask][0]
    if damage == "missing":
        predictions = predictions.drop(index)
    elif damage == "duplicate":
        predictions = pd.concat([predictions, predictions.loc[[index]]], ignore_index=True)
    elif damage == "target":
        predictions.loc[index, "observed_vehicle_total"] += 1
    else:
        predictions.loc[index, "inferred_site_id"] = "different-site"

    with pytest.raises(ValueError, match=message):
        analysis.paired_prediction_rows(predictions, RF_COMBINED, HGB_COMBINED)


def test_site_metrics_and_volume_strata_preserve_error_direction() -> None:
    site = analysis.site_error_summary(_predictions())
    selected = site.loc[
        site["model"].eq("random_forest") & site["feature_set"].eq("combined")
    ]
    assert len(selected) == 3
    assert selected["rows"].eq(4).all()
    assert selected["mae"].eq(4.0).all()
    assert selected["rmse"].eq(np.sqrt(28.5)).all()
    assert selected["mean_bias"].eq(-2.5).all()

    volume = analysis.volume_quantile_summary(_predictions(), quantiles=4)
    selected_volume = volume.loc[
        volume["model"].eq("random_forest")
        & volume["feature_set"].eq("combined")
        & volume["information_budget"].eq("measurement_free_estimation")
    ].set_index("volume_quantile")
    assert len(selected_volume) == 4
    assert selected_volume.iloc[0]["mean_bias"] == pytest.approx(2.0)
    assert selected_volume.iloc[0]["overprediction_rate"] == pytest.approx(1.0)
    assert selected_volume.iloc[-1]["mean_bias"] == pytest.approx(-10.0)
    assert selected_volume.iloc[-1]["underprediction_rate"] == pytest.approx(1.0)


def test_volume_strata_reject_silently_collapsed_requested_quantiles() -> None:
    predictions = _predictions()
    target_by_observation = {
        observation_id: 0 if index < 6 else 100
        for index, observation_id in enumerate(
            sorted(predictions["observation_id"].unique())
        )
    }
    predictions["observed_vehicle_total"] = predictions["observation_id"].map(
        target_by_observation
    )

    with pytest.raises(ValueError, match="requested 4 distinct quantiles"):
        analysis.volume_quantile_summary(predictions, quantiles=4)


def test_whole_site_bootstrap_is_paired_deterministic_and_correctly_signed() -> None:
    paired = analysis.paired_prediction_rows(_predictions(), RF_COMBINED, HGB_COMBINED)

    first, first_samples = analysis.paired_site_bootstrap(
        paired, replicates=200, seed=71, confidence_level=0.95
    )
    second, second_samples = analysis.paired_site_bootstrap(
        paired.sample(frac=1, random_state=9),
        replicates=200,
        seed=71,
        confidence_level=0.95,
    )

    assert first == second
    pd.testing.assert_frame_equal(first_samples, second_samples)
    assert first["inferred_sites"] == 3
    assert first["delta_mae_left_minus_right"] == pytest.approx(-4.75)
    assert first["delta_mae_ci_low"] == pytest.approx(-4.75)
    assert first["delta_mae_ci_high"] == pytest.approx(-4.75)


def test_leave_one_site_out_reports_each_cluster_once() -> None:
    paired = analysis.paired_prediction_rows(_predictions(), RF_COMBINED, RF_SPATIAL)
    influence = analysis.leave_one_site_out_influence(paired)

    assert len(influence) == 3
    assert influence["inferred_site_id"].is_unique
    assert influence["remaining_sites"].eq(2).all()
    assert influence["remaining_rows"].eq(8).all()
    assert influence["mae_influence"].eq(0.0).all()


def test_cv_holdout_comparison_keeps_training_folds_separate() -> None:
    comparison = analysis.cv_holdout_comparison(_cv_metrics(), _holdout_metrics())

    assert len(comparison) == 4
    assert comparison["cv_folds"].eq(5).all()
    selected = comparison.loc[
        comparison["model"].eq("random_forest")
        & comparison["feature_set"].eq("combined")
    ].iloc[0]
    assert selected["cv_mean_mae"] == pytest.approx(7.0)
    assert selected["holdout_mae"] == pytest.approx(4.0)
    assert selected["holdout_minus_cv_mae"] == pytest.approx(-3.0)


@pytest.mark.parametrize(
    "operation, message",
    [
        (
            lambda frame: analysis.site_error_summary(frame.drop(columns="prediction")),
            "missing columns",
        ),
        (
            lambda frame: analysis.site_error_summary(frame.assign(prediction=-1)),
            "invalid non-negative",
        ),
        (
            lambda frame: analysis.volume_quantile_summary(
                frame.assign(observed_vehicle_total=10), 4
            ),
            "at least two distinct totals",
        ),
        (
            lambda frame: analysis.paired_site_bootstrap(
                analysis.paired_prediction_rows(
                    frame.loc[frame["inferred_site_id"].eq("restricted-site-0")],
                    RF_COMBINED,
                    HGB_COMBINED,
                ),
                replicates=10,
                seed=1,
                confidence_level=0.95,
            ),
            "at least two inferred sites",
        ),
    ],
)
def test_diagnostics_reject_missing_corrupt_or_insufficient_values(operation, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        operation(_predictions())


def _fake_figures(directory: Path, *_args) -> list[Path]:
    figure_dir = directory / "figures"
    figure_dir.mkdir()
    paths = []
    for name in (
        "site_mae_private.png",
        "volume_error_private.png",
        "paired_uncertainty_private.png",
        "cv_holdout_private.png",
    ):
        path = figure_dir / name
        path.write_bytes(b"synthetic test figure")
        paths.append(path)
    return paths


def test_figure_renderer_creates_complete_private_diagnostics(tmp_path: Path) -> None:
    predictions = _predictions()
    site_errors = analysis._anonymize_sites(analysis.site_error_summary(predictions))
    volume_errors = analysis.volume_quantile_summary(predictions, quantiles=4)
    paired_rows = analysis.paired_prediction_rows(
        predictions, RF_COMBINED, HGB_COMBINED
    )
    point, _ = analysis.paired_site_bootstrap(
        paired_rows, replicates=20, seed=7, confidence_level=0.95
    )
    paired = pd.DataFrame(
        [
            {
                "comparison_id": "rf_combined_vs_histgb_combined",
                "left_label": "random_forest / combined",
                "right_label": "histogram_gradient_boosting / combined",
                **point,
            }
        ]
    )
    cv_holdout = analysis.cv_holdout_comparison(_cv_metrics(), _holdout_metrics())

    paths = analysis._write_figures(
        tmp_path,
        site_errors,
        volume_errors,
        paired,
        cv_holdout,
        RF_COMBINED,
    )

    assert {path.name for path in paths} == {
        "site_mae_private.png",
        "volume_error_private.png",
        "paired_uncertainty_private.png",
        "cv_holdout_private.png",
    }
    for path in paths:
        assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert path.stat().st_size > 1_000


def test_private_runner_is_atomic_hashed_and_identifier_free(tmp_path: Path, monkeypatch) -> None:
    config = _write_config(tmp_path / "analysis.yaml")
    bundle = tmp_path / "upstream"
    bundle.mkdir()
    (bundle / "provenance.json").write_text('{"synthetic":"test-only"}\n', encoding="utf-8")
    destination = tmp_path / ".urbantwin" / "private" / "private-analysis"
    verified = _verified_bundle()
    monkeypatch.setattr(analysis, "validate_observed_artifacts", lambda *_args, **_kwargs: verified)
    monkeypatch.setattr(analysis, "_write_figures", _fake_figures)

    result = analysis.run_observed_error_analysis(
        tmp_path,
        bundle,
        destination,
        config,
        validation_root=tmp_path / "historical-validation-root",
        source_dir=tmp_path / "source",
    )

    assert result["output_dir"] == destination.resolve()
    assert destination.is_dir()
    assert not list(destination.parent.glob(".private-analysis.*"))
    manifest = json.loads((destination / "analysis_manifest.json").read_text(encoding="utf-8"))
    assert manifest["publication_status"] == analysis.PRIVATE_STATUS
    assert manifest["upstream_provenance_sha256"] == sha256_file(
        bundle / "provenance.json"
    )
    assert manifest["holdout_inferred_sites"] == 3
    for relative, expected in manifest["artifact_hashes"].items():
        assert sha256_file(destination / relative) == expected
    for path in destination.glob("*.csv"):
        columns = set(pd.read_csv(path, nrows=0).columns)
        assert not columns & analysis._IDENTIFIER_COLUMNS
    validated = analysis.validate_error_analysis_bundle(
        destination,
        config,
        upstream_provenance_sha256=manifest["upstream_provenance_sha256"],
    )
    assert validated["manifest"]["bootstrap_unit"] == "whole_inferred_site"

    with pytest.raises(FileExistsError, match="already exists"):
        analysis.run_observed_error_analysis(tmp_path, bundle, destination, config)


def test_private_runner_cleans_staging_after_render_failure(tmp_path: Path, monkeypatch) -> None:
    config = _write_config(tmp_path / "analysis.yaml")
    bundle = tmp_path / "upstream"
    bundle.mkdir()
    (bundle / "provenance.json").write_text('{"synthetic":"test-only"}\n', encoding="utf-8")
    destination = tmp_path / ".urbantwin" / "private" / "failed-analysis"
    monkeypatch.setattr(
        analysis,
        "validate_observed_artifacts",
        lambda *_args, **_kwargs: _verified_bundle(),
    )

    def fail_render(*_args):
        raise RuntimeError("synthetic rendering failure")

    monkeypatch.setattr(analysis, "_write_figures", fail_render)
    with pytest.raises(RuntimeError, match="synthetic rendering failure"):
        analysis.run_observed_error_analysis(tmp_path, bundle, destination, config)

    assert not destination.exists()
    assert not list(destination.parent.glob(".failed-analysis.*"))


def test_private_runner_rejects_output_outside_private_tree(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="under .urbantwin/private"):
        analysis.run_observed_error_analysis(
            tmp_path,
            tmp_path / "bundle",
            tmp_path / "public-analysis",
            tmp_path / "analysis.yaml",
        )


def test_private_validator_rejects_corruption(tmp_path: Path, monkeypatch) -> None:
    config = _write_config(tmp_path / "analysis.yaml")
    bundle = tmp_path / "upstream"
    bundle.mkdir()
    (bundle / "provenance.json").write_text('{"synthetic":"test-only"}\n', encoding="utf-8")
    destination = tmp_path / ".urbantwin" / "private" / "private-analysis"
    monkeypatch.setattr(
        analysis,
        "validate_observed_artifacts",
        lambda *_args, **_kwargs: _verified_bundle(),
    )
    monkeypatch.setattr(analysis, "_write_figures", _fake_figures)
    analysis.run_observed_error_analysis(tmp_path, bundle, destination, config)
    (destination / "site_errors_private.csv").write_text("corrupt\n", encoding="utf-8")

    with pytest.raises(ObservedArtifactIntegrityError, match="hash-mismatched"):
        analysis.validate_error_analysis_bundle(destination, config)


def test_private_validator_rejects_malformed_revision_provenance(
    tmp_path: Path, monkeypatch
) -> None:
    config = _write_config(tmp_path / "analysis.yaml")
    bundle = tmp_path / "upstream"
    bundle.mkdir()
    (bundle / "provenance.json").write_text(
        '{"synthetic":"test-only"}\n', encoding="utf-8"
    )
    destination = tmp_path / ".urbantwin" / "private" / "private-analysis"
    monkeypatch.setattr(
        analysis,
        "validate_observed_artifacts",
        lambda *_args, **_kwargs: _verified_bundle(),
    )
    monkeypatch.setattr(analysis, "_write_figures", _fake_figures)
    analysis.run_observed_error_analysis(tmp_path, bundle, destination, config)
    manifest_path = destination / "analysis_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["analysis_git"] = {}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ObservedArtifactIntegrityError, match="malformed"):
        analysis.validate_error_analysis_bundle(destination, config)
