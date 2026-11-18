from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from urbantwin.geoai.error_analysis import (
    analyze_geoai_details,
    confusion_metrics,
    normalize_confusion,
    paired_feature_ablations,
    training_majority_baseline,
)


def test_normalized_confusion_preserves_unsupported_rows() -> None:
    normalized = normalize_confusion([[3, 1], [0, 0]])

    np.testing.assert_allclose(normalized, [[0.75, 0.25], [0.0, 0.0]])


def test_confusion_metrics_report_minority_failures_and_balanced_accuracy() -> None:
    per_class, aggregate = confusion_metrics(
        [[90, 10], [8, 2]],
        [10, 50],
        class_names=["minority", "dominant"],
    )

    assert per_class["support"].tolist() == [100, 10]
    assert per_class.loc[1, "recall"] == pytest.approx(0.2)
    assert aggregate["accuracy"] == pytest.approx(92 / 110)
    assert aggregate["balanced_accuracy"] == pytest.approx(0.55)
    assert aggregate["macro_f1"] < aggregate["weighted_f1"]


def test_training_majority_selection_is_independent_of_test_distribution() -> None:
    train = {10: 4, 20: 9}
    first = training_majority_baseline(train, {10: 100, 20: 1}, [10, 20])
    second = training_majority_baseline(train, {10: 1, 20: 100}, [10, 20])

    assert first["predicted_class_id"] == second["predicted_class_id"] == 20
    assert first["accuracy"] == pytest.approx(1 / 101)
    assert second["accuracy"] == pytest.approx(100 / 101)


def test_training_majority_tie_break_is_deterministic() -> None:
    baseline = training_majority_baseline({20: 5, 10: 5}, {10: 2, 20: 2}, [20, 10])

    assert baseline["predicted_class_id"] == 10


def test_paired_feature_ablations_require_complete_unique_pairs() -> None:
    frame = pd.DataFrame(
        [
            {
                "method": "rf",
                "features": feature,
                "split_seed": 7,
                "macro_f1": macro,
                "weighted_f1": 0.8,
                "accuracy": 0.75,
                "balanced_accuracy": 0.6,
            }
            for feature, macro in (("bands_only", 0.5), ("bands_plus_indices", 0.48))
        ]
    )

    paired = paired_feature_ablations(frame)
    assert paired.loc[0, "delta_macro_f1__indices_minus_bands"] == pytest.approx(-0.02)

    with pytest.raises(ValueError, match="complete"):
        paired_feature_ablations(frame.iloc[:1])


def _details_fixture() -> dict:
    experiments = {}
    for features, matrix in (
        ("bands_only", [[8, 2], [1, 9]]),
        ("bands_plus_indices", [[7, 3], [2, 8]]),
    ):
        per_class, aggregate = confusion_metrics(matrix, [10, 50])
        experiments[f"rf__{features}"] = {
            "row": {"method": "rf", "features": features},
            "runs": [
                {
                    "seed": 3,
                    "eligible_class_ids": [10, 50],
                    "confusion_matrix": matrix,
                    "row": {
                        "macro_f1": aggregate["macro_f1"],
                        "weighted_f1": aggregate["weighted_f1"],
                        "accuracy_vs_worldcover": aggregate["accuracy"],
                    },
                    "per_class": per_class.to_dict(orient="records"),
                }
            ],
        }
    return {
        "class_ids": [10, 50],
        "class_names": ["Tree", "Built-up"],
        "split_diagnostics": [
            {
                "seed": 3,
                "eligible_class_ids": [10, 50],
                "raw_train_class_support": {"10": 12, "50": 30},
            }
        ],
        "experiments": experiments,
    }


def test_analyze_geoai_details_uses_stored_counts_without_row_reconstruction() -> None:
    analysis = analyze_geoai_details(_details_fixture())

    assert len(analysis["per_class"]) == 4
    assert len(analysis["split_metrics"]) == 2
    assert set(analysis["majority_baselines"]["predicted_class_id"]) == {50}
    assert analysis["paired_ablations"].loc[
        0, "delta_balanced_accuracy__indices_minus_bands"
    ] == pytest.approx(-0.1)


@pytest.mark.parametrize(
    "matrix, error",
    [
        ([[1, 2, 3], [4, 5, 6]], "square"),
        ([[1, -1], [0, 1]], "non-negative"),
        ([[1.5, 0], [0, 1]], "integers"),
    ],
)
def test_confusion_validation_rejects_malformed_counts(matrix, error: str) -> None:
    with pytest.raises(ValueError, match=error):
        normalize_confusion(matrix)


@pytest.mark.parametrize(
    ("matrix", "class_ids", "class_names", "eligible", "error_type", "error"),
    [
        ([[1]], [], None, None, ValueError, "non-empty"),
        ([[1, 0], [0, 1]], [10, 10], None, None, ValueError, "unique"),
        ([[1]], [10], ["Tree", "Built-up"], None, ValueError, "align"),
        ([[1, 0, 0], [0, 1, 0]], [10, 50], None, None, ValueError, "shape"),
        ([["one", "zero"], ["zero", "one"]], [10, 50], None, None, TypeError, "numeric"),
        ([[1, 0], [0, 1]], [10, 50], None, [], ValueError, "non-empty"),
        ([[1, 0], [0, 1]], [10, 50], None, [10, 80], ValueError, "absent"),
        ([[0, 0], [0, 2]], [10, 50], None, [10], ValueError, "no held-out"),
    ],
)
def test_confusion_metrics_reject_invalid_axes_and_eligibility(
    matrix,
    class_ids,
    class_names,
    eligible,
    error_type,
    error: str,
) -> None:
    with pytest.raises(error_type, match=error):
        confusion_metrics(
            matrix,
            class_ids,
            class_names=class_names,
            eligible_class_ids=eligible,
        )


@pytest.mark.parametrize(
    ("training", "evaluation", "eligible", "error"),
    [
        ({10: 1}, {10: 1}, [], "non-empty"),
        ({10: 1}, {10: 1}, [10, 10], "unique"),
        ({10: True}, {10: 1}, [10], "non-negative integer"),
        ({10: 1.5}, {10: 1}, [10], "non-negative integer"),
        ({10: -1}, {10: 1}, [10], "non-negative integer"),
        ({10: 0}, {10: 1}, [10], "no training support"),
        ({10: 1}, {10: 0}, [10], "no evaluation support"),
    ],
)
def test_training_majority_rejects_invalid_support(
    training, evaluation, eligible, error: str
) -> None:
    with pytest.raises(ValueError, match=error):
        training_majority_baseline(training, evaluation, eligible)


def test_paired_feature_ablations_reject_missing_columns_and_duplicates() -> None:
    complete = pd.DataFrame(
        [
            {
                "method": "rf",
                "features": feature,
                "split_seed": 7,
                "macro_f1": 0.5,
                "weighted_f1": 0.8,
                "accuracy": 0.75,
                "balanced_accuracy": 0.6,
            }
            for feature in ("bands_only", "bands_plus_indices")
        ]
    )

    with pytest.raises(ValueError, match="missing columns: balanced_accuracy"):
        paired_feature_ablations(complete.drop(columns="balanced_accuracy"))

    duplicated = pd.concat([complete, complete.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        paired_feature_ablations(duplicated)


def _single_experiment(details: dict) -> dict:
    details["experiments"] = {"rf__bands_only": details["experiments"]["rf__bands_only"]}
    return details


@pytest.mark.parametrize(
    ("mutate", "error_type", "error"),
    [
        (
            lambda details: details.update(split_diagnostics=None),
            TypeError,
            "split_diagnostics and experiments",
        ),
        (
            lambda details: details.update(split_diagnostics=[{}]),
            TypeError,
            "diagnostic is malformed",
        ),
        (
            lambda details: details.update(
                split_diagnostics=[details["split_diagnostics"][0]] * 2
            ),
            ValueError,
            "Duplicate.*seed",
        ),
        (
            lambda details: details.update(experiments={"bad": None}),
            TypeError,
            "experiment is malformed",
        ),
        (
            lambda details: _single_experiment(details)["experiments"][
                "rf__bands_only"
            ].update(runs={}),
            TypeError,
            "runs must be a list",
        ),
        (
            lambda details: _single_experiment(details)["experiments"][
                "rf__bands_only"
            ].update(runs=[None]),
            TypeError,
            "run is malformed",
        ),
        (
            lambda details: _single_experiment(details)["experiments"][
                "rf__bands_only"
            ]["runs"][0].update(seed=99),
            ValueError,
            "Missing split diagnostic",
        ),
        (
            lambda details: _single_experiment(details)["experiments"][
                "rf__bands_only"
            ]["runs"][0].update(eligible_class_ids=[10]),
            ValueError,
            "Eligible classes disagree",
        ),
        (
            lambda details: _single_experiment(details)["experiments"][
                "rf__bands_only"
            ]["runs"][0].update(row=None),
            TypeError,
            "run row is malformed",
        ),
        (
            lambda details: _single_experiment(details)["experiments"][
                "rf__bands_only"
            ]["runs"][0]["row"].update(macro_f1=0.0),
            ValueError,
            "Stored macro_f1 disagrees",
        ),
    ],
)
def test_analyze_geoai_details_rejects_inconsistent_stored_artifacts(
    mutate, error_type, error: str
) -> None:
    details = deepcopy(_details_fixture())
    mutate(details)

    with pytest.raises(error_type, match=error):
        analyze_geoai_details(details)
