from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import mean_absolute_error

from urbantwin.common import sha256_file
from urbantwin.data.observed_traffic import ObservedTrafficValidationError
from urbantwin.observed import features
from urbantwin.observed.benchmark import (
    _add_previous_interval,
    _bootstrap_interval,
    _feature_columns,
    _historical_mean,
    _pipeline,
    _split_protocol,
    _training_cv_selection,
)


def _split_frame() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    return pd.DataFrame(
        {
            "observation_id": [f"observation-{index}" for index in range(5)],
            "observed_date": dates,
            "inferred_site_id": [f"site-{index}" for index in range(5)],
            "eligible_for_spatial_benchmark": True,
        }
    )


def test_frozen_holdout_is_chronological_site_disjoint_and_order_invariant() -> None:
    frame = _split_frame()

    train, holdout, manifest = _split_protocol(frame, test_fraction=0.4)
    shuffled_train, shuffled_holdout, shuffled_manifest = _split_protocol(
        frame.sample(frac=1.0, random_state=19), test_fraction=0.4
    )

    assert set(train["observation_id"]) == {
        "observation-0",
        "observation-1",
        "observation-2",
    }
    assert set(holdout["observation_id"]) == {"observation-3", "observation-4"}
    assert set(train["observed_date"]).isdisjoint(holdout["observed_date"])
    assert set(train["inferred_site_id"]).isdisjoint(holdout["inferred_site_id"])
    assert set(shuffled_train["observation_id"]) == set(train["observation_id"])
    assert set(shuffled_holdout["observation_id"]) == set(holdout["observation_id"])
    assert shuffled_manifest == manifest
    assert manifest["protocol"] == "future_date_and_unseen_inferred_site_holdout"
    assert manifest["site_overlap"] == []
    assert "confounded" in manifest["warning"]


def test_frozen_holdout_rejects_site_overlap() -> None:
    frame = _split_frame()
    frame.loc[4, "inferred_site_id"] = frame.loc[0, "inferred_site_id"]

    with pytest.raises(ValueError, match="not site-disjoint"):
        _split_protocol(frame, test_fraction=0.4)


def test_model_selection_uses_training_cv_scores_only() -> None:
    cv_metrics = pd.DataFrame(
        [
            {
                "fold": fold,
                "model": model,
                "feature_set": "combined",
                "mae": mae,
                "rmse": rmse,
                # This deliberately contradictory diagnostic is not a selector input.
                "holdout_mae": holdout_mae,
            }
            for fold in (1, 2, 3)
            for model, mae, rmse, holdout_mae in (
                ("random_forest", 10.0 + fold, 12.0 + fold, 10_000.0),
                ("histogram_gradient_boosting", 20.0 + fold, 22.0 + fold, 0.0),
            )
        ]
    )

    selected = _training_cv_selection(cv_metrics)
    selected_after_shuffle = _training_cv_selection(
        cv_metrics.sample(frac=1.0, random_state=7)
    )

    assert selected == selected_after_shuffle
    assert selected["model"] == "random_forest"
    assert selected["selection_partition"] == "training_only"
    assert selected["holdout_scores_used_for_selection"] is False
    assert selected["candidate_count"] == 2


def test_historical_mean_is_fitted_only_from_training_records() -> None:
    train = pd.DataFrame(
        {
            "day_of_week": [0, 0, 1, 0],
            "minute_of_day": [480, 480, 480, 500],
            "observed_vehicle_total": [10, 14, 30, 50],
        }
    )
    holdout = pd.DataFrame(
        {
            "day_of_week": [0, 2, 2],
            "minute_of_day": [480, 480, 600],
            "observed_vehicle_total": [1, 2, 3],
        }
    )

    prediction = _historical_mean(train, holdout)
    changed_holdout = holdout.assign(observed_vehicle_total=[1_000, 2_000, 3_000])

    assert prediction == pytest.approx([12.0, 18.0, 26.0])
    assert _historical_mean(train, changed_holdout) == pytest.approx(prediction)


def test_persistence_requires_exactly_adjacent_twenty_minute_intervals() -> None:
    frame = pd.DataFrame(
        {
            "observation_id": ["a", "b", "c", "d", "e", "f"],
            "segment_id": ["one", "one", "one", "one", "two", "two"],
            "observed_date": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-01",
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-01",
                    "2024-01-01",
                ]
            ),
            "minute_of_day": [480, 500, 540, 480, 500, 520],
            "observed_vehicle_total": [10, 20, 30, 40, 50, 60],
        }
    )

    lagged = _add_previous_interval(frame).set_index("observation_id")

    assert lagged.index[lagged["persistence_eligible"]].tolist() == ["b", "f"]
    assert lagged.loc["b", "previous_observed_total"] == 10
    assert lagged.loc["f", "previous_observed_total"] == 50
    assert not bool(lagged.loc["c", "persistence_eligible"])
    assert not bool(lagged.loc["d", "persistence_eligible"])


def test_site_bootstrap_is_clustered_reproducible_and_order_invariant() -> None:
    y_true = np.asarray([11, 12, 21, 22, 23, 31], dtype=float)
    y_pred = np.zeros_like(y_true)
    groups = np.asarray(["a", "a", "b", "b", "b", "c"])
    calls = 0

    def checked_mae(actual: np.ndarray, estimate: np.ndarray) -> float:
        nonlocal calls
        calls += 1
        # Every selected site contributes its complete row cluster, possibly repeated.
        for site_values in ({11.0, 12.0}, {21.0, 22.0, 23.0}, {31.0}):
            counts = [int(np.count_nonzero(actual == value)) for value in site_values]
            assert len(set(counts)) == 1
        return float(mean_absolute_error(actual, estimate))

    first = _bootstrap_interval(
        y_true,
        y_pred,
        groups,
        checked_mae,
        replicates=200,
        seed=42,
    )
    second = _bootstrap_interval(
        y_true,
        y_pred,
        groups,
        checked_mae,
        replicates=200,
        seed=42,
    )
    order = np.asarray([5, 1, 3, 0, 4, 2])
    permuted = _bootstrap_interval(
        y_true[order],
        y_pred[order],
        groups[order],
        checked_mae,
        replicates=200,
        seed=42,
    )

    assert calls == 600
    assert first == pytest.approx((11.5, 25.735))
    assert first == pytest.approx(second)
    assert first == pytest.approx(permuted)
    assert first[0] <= first[1]


def test_site_bootstrap_requires_multiple_sites_for_interval() -> None:
    bounds = _bootstrap_interval(
        np.asarray([1.0, 2.0]),
        np.asarray([1.0, 2.0]),
        np.asarray(["only-site", "only-site"]),
        mean_absolute_error,
        replicates=20,
        seed=42,
    )

    assert all(np.isnan(value) for value in bounds)


@dataclass(frozen=True)
class _PointOnAxis:
    metres: float

    def distance(self, other: _PointOnAxis) -> float:
        return abs(self.metres - other.metres)


def test_inferred_sites_use_transitive_250_metre_components() -> None:
    geometry_by_segment = {
        "beta": _PointOnAxis(0.0),
        "alpha": _PointOnAxis(250.0),
        "gamma": _PointOnAxis(500.0),
        "zeta": _PointOnAxis(1_000.0),
    }
    segment_ids = list(geometry_by_segment)
    labels = features._component_labels(
        [geometry_by_segment[segment] for segment in segment_ids],
        segment_ids,
        distance_m=250.0,
    )
    mapping = dict(zip(segment_ids, labels, strict=True))
    reversed_ids = list(reversed(segment_ids))
    reversed_labels = features._component_labels(
        [geometry_by_segment[segment] for segment in reversed_ids],
        reversed_ids,
        distance_m=250.0,
    )

    assert mapping["alpha"] == mapping["beta"] == mapping["gamma"]
    assert mapping["gamma"] != mapping["zeta"]
    assert dict(zip(reversed_ids, reversed_labels, strict=True)) == mapping


def _analysis_counts() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "observation_id": ["observation-a", "observation-b"],
            "segment_id": ["segment-a", "segment-b"],
            "direction_code": ["N", "S"],
            "observed_date": ["2024-01-01", "2024-01-02"],
            "interval_start_local": ["08:00", "08:20"],
            "observed_vehicle_total": [10, 20],
        }
    )


def _analysis_segments() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "segment_id": ["segment-a", "segment-b"],
            "geometry_osm_id": ["osm-a", "osm-b"],
            "geometry_direction_code": ["N", "S"],
            "spatial_match_status": [
                "eligible_segment_join",
                "excluded_ambiguous_or_unobserved",
            ],
            "inferred_site_id": ["site-a", "site-b"],
            "geometry": [None, None],
        }
    )


def test_ambiguous_spatial_join_is_retained_but_excluded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(features, "load_normalized_counts", lambda _: _analysis_counts())
    monkeypatch.setattr(
        features,
        "spatial_segment_features",
        lambda *_args, **_kwargs: _analysis_segments(),
    )

    frame = features.build_analysis_table(tmp_path)

    assert frame["segment_id"].tolist() == ["segment-a", "segment-b"]
    assert frame["eligible_for_spatial_benchmark"].tolist() == [True, False]
    assert frame.loc[1, "spatial_match_status"] == "excluded_ambiguous_or_unobserved"


def test_count_geometry_direction_disagreement_is_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    segments = _analysis_segments()
    segments.loc[0, "geometry_direction_code"] = "E"
    monkeypatch.setattr(features, "load_normalized_counts", lambda _: _analysis_counts())
    monkeypatch.setattr(
        features,
        "spatial_segment_features",
        lambda *_args, **_kwargs: segments,
    )

    with pytest.raises(ObservedTrafficValidationError, match="direction codes disagree"):
        features.build_analysis_table(tmp_path)


def test_model_feature_whitelist_drops_targets_ids_and_retrospective_eo() -> None:
    forbidden = {
        "observed_vehicle_total",
        "car",
        "segment_id",
        "inferred_site_id",
        "geometry_osm_id",
        "observed_date",
        "opposite_direction_total",
        "previous_observed_total",
        "ndvi_mean",
        "ndbi_mean",
    }
    for feature_set in ("time_only", "spatial_only", "combined"):
        numeric, categorical = _feature_columns(feature_set)
        assert forbidden.isdisjoint([*numeric, *categorical])

    fit = pd.DataFrame(
        {
            "time_sin": [0.0, 1.0, -1.0],
            "time_cos": [1.0, 0.0, 0.0],
            "day_of_week": [0, 1, 2],
            "geometry_length_m": [100.0, 200.0, 300.0],
            "centroid_easting_m": [330_000.0, 330_100.0, 330_200.0],
            "centroid_northing_m": [3_320_000.0, 3_320_100.0, 3_320_200.0],
            "direction_code": ["N", "S", "E"],
            "highway": ["primary", "primary", "secondary"],
            "oneway": ["yes", "yes", "no"],
            **{column: [1, 2, 3] for column in forbidden},
        }
    )
    pipeline = _pipeline(
        "poisson",
        "combined",
        {
            "random_forest_trees": 10,
            "histogram_gradient_boosting_iterations": 10,
        },
        seed=42,
    )
    preprocessor = pipeline.named_steps["features"]
    transformed = preprocessor.fit_transform(fit)
    changed = fit.copy()
    for column in forbidden:
        changed[column] = [10_000, 20_000, 30_000]

    assert preprocessor.remainder == "drop"
    assert transformed == pytest.approx(preprocessor.transform(changed))
    scaler = preprocessor.named_transformers_["numeric"]
    allowed_numeric, _ = _feature_columns("combined")
    assert scaler.mean_ == pytest.approx(fit[allowed_numeric].mean().to_numpy())


def _write_artifact_manifest(study_dir: Path, filename: str, checksum: str) -> None:
    manifest = {
        "data_kind": "real_observed_traffic",
        "artifacts": {
            "survey_segments": {
                "filename": filename,
                "sha256": checksum,
            }
        },
    }
    (study_dir / "source_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )


def test_missing_observed_source_manifest_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="manifest is missing"):
        features._verified_artifact(tmp_path, "survey_segments")


@pytest.mark.parametrize(
    "manifest_text, message",
    [
        ("{not-json", "Invalid observed-study manifest"),
        (json.dumps({"data_kind": "synthetic"}), "not real traffic data"),
    ],
)
def test_invalid_observed_source_manifest_is_rejected(
    tmp_path: Path, manifest_text: str, message: str
) -> None:
    (tmp_path / "source_manifest.json").write_text(manifest_text, encoding="utf-8")

    with pytest.raises(ObservedTrafficValidationError, match=message):
        features._verified_artifact(tmp_path, "survey_segments")


def test_missing_or_corrupted_observed_artifact_is_rejected(tmp_path: Path) -> None:
    artifact = tmp_path / "survey_segments.geojson"
    _write_artifact_manifest(tmp_path, artifact.name, "0" * 64)

    with pytest.raises(FileNotFoundError, match="artifact is missing"):
        features._verified_artifact(tmp_path, "survey_segments")

    artifact.write_text('{"type":"FeatureCollection","features":[]}', encoding="utf-8")
    with pytest.raises(ObservedTrafficValidationError, match="hash mismatch"):
        features._verified_artifact(tmp_path, "survey_segments")

    _write_artifact_manifest(tmp_path, artifact.name, sha256_file(artifact))
    assert features._verified_artifact(tmp_path, "survey_segments") == artifact.resolve()


def test_observed_artifact_cannot_escape_its_bundle(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside-survey-segments.geojson"
    outside.write_text("{}", encoding="utf-8")
    _write_artifact_manifest(tmp_path, f"../{outside.name}", sha256_file(outside))

    with pytest.raises(ObservedTrafficValidationError, match="escapes its bundle"):
        features._verified_artifact(tmp_path, "survey_segments")
