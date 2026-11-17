from __future__ import annotations

import inspect
import json
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image as PILImage
from reportlab.pdfbase import pdfmetrics

from tests.test_observed_artifacts import _observed_bundle
from urbantwin.cli import _parser
from urbantwin.common import sha256_file
from urbantwin.observed.artifacts import ObservedArtifactIntegrityError
from urbantwin.observed.report import (
    BODY_FONT,
    BOLD_FONT,
    _dataset_overview_rows,
    _date_span,
    _manifest_count,
    _register_pdf_fonts,
    _require_report_figures,
    _split_warning,
    _survey_map_image,
    make_observed_report,
)


def _overview_fixture() -> dict:
    return {
        "source_manifest": {
            "row_counts": {"counts_wide": 8, "survey_segments": 9},
            "normalization": {"timezone_status": "unspecified_by_publisher"},
            "license": {"status": "unspecified_by_publisher"},
        },
        "split_manifest": {
            "train_rows": 4,
            "test_rows": 3,
            "excluded_ambiguous_rows": 1,
            "train_inferred_sites": 3,
            "test_inferred_sites": 2,
            "train_dates": ["2024-01-01", "2024-01-02"],
            "test_dates": ["2024-02-01"],
        },
        "vehicle_classes": pd.DataFrame(
            [
                {"vehicle_class": "a", "observed_vehicles": 10},
                {"vehicle_class": "b", "observed_vehicles": 20},
            ]
        ),
    }


def test_report_defaults_to_full_private_mode() -> None:
    parameter = inspect.signature(make_observed_report).parameters["publication_safe"]

    assert parameter.default is False


def test_observed_report_cli_exposes_explicit_publication_safe_mode() -> None:
    arguments = _parser().parse_args(["observed-report", "--publication-safe"])

    assert arguments.publication_safe is True


def test_dataset_overview_is_derived_from_validated_metadata() -> None:
    verified = _overview_fixture()

    rows = {row[0]: row[1:] for row in _dataset_overview_rows(verified)[1:]}

    assert rows["Directional intervals"][0] == "8"
    assert rows["Observed vehicles"][0] == "30"
    assert rows["Survey dates"][0] == "3 (1 Jan–1 Feb 2024)"
    assert rows["Released survey lines"][0] == "9"
    assert rows["Inferred analytical sites"][0] == "5"
    assert rows["Ambiguous OSM rows excluded"][0] == "1"
    assert rows["Timezone"][0] == "Unspecified by publisher"


@pytest.mark.parametrize("value", [True, -1, 1.5, "1"])
def test_manifest_counts_reject_non_integer_or_negative_values(value: object) -> None:
    with pytest.raises(ValueError, match="invalid synthetic row count"):
        _manifest_count(value, "synthetic row count")


@pytest.mark.parametrize(
    ("dates", "expected"),
    [
        (["2024-01-01"], "1 (1 Jan 2024)"),
        (["2024-01-01", "2024-01-20"], "2 (1–20 Jan 2024)"),
        (["2023-12-31", "2024-01-01"], "2 (31 Dec 2023–1 Jan 2024)"),
    ],
)
def test_survey_date_span_formats_single_month_and_cross_year_ranges(
    dates: list[str], expected: str
) -> None:
    assert _date_span(dates) == expected


@pytest.mark.parametrize(
    ("dates", "message"),
    [
        ([], "contains no survey dates"),
        (["2024-02-30"], "contains an invalid ISO date"),
        (["not-a-date"], "contains an invalid ISO date"),
    ],
)
def test_survey_date_span_rejects_empty_or_malformed_values(
    dates: list[str], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _date_span(dates)


@pytest.mark.parametrize(
    ("field", "value", "error", "message"),
    [
        ("row_counts", None, TypeError, "no row_counts object"),
        ("normalization", None, TypeError, "no normalization metadata"),
        ("timezone_status", "  ", ValueError, "no timezone status"),
        ("license", None, TypeError, "no license metadata"),
        ("license_status", "", ValueError, "no license status"),
    ],
)
def test_dataset_overview_rejects_missing_or_invalid_manifest_metadata(
    field: str,
    value: object,
    error: type[Exception],
    message: str,
) -> None:
    verified = _overview_fixture()
    source = verified["source_manifest"]
    if field in {"row_counts", "normalization", "license"}:
        source[field] = value
    elif field == "timezone_status":
        source["normalization"]["timezone_status"] = value
    else:
        source["license"]["status"] = value

    with pytest.raises(error, match=message):
        _dataset_overview_rows(verified)


@pytest.mark.parametrize(
    "totals",
    [
        [],
        [True],
        ["not-a-number"],
        [float("nan")],
        [1e308, 1e308],
        [-1],
        [10, -10],
        [1.5],
    ],
)
def test_dataset_overview_rejects_invalid_vehicle_totals(totals: list[object]) -> None:
    verified = _overview_fixture()
    verified["vehicle_classes"] = pd.DataFrame({"observed_vehicles": totals})

    with pytest.raises(ValueError, match="invalid total"):
        _dataset_overview_rows(verified)


def test_dataset_overview_rejects_manifest_count_disagreement() -> None:
    verified = {
        "source_manifest": {
            "row_counts": {"counts_wide": 9, "survey_segments": 1},
            "normalization": {"timezone_status": "unspecified"},
            "license": {"status": "unspecified"},
        },
        "split_manifest": {
            "train_rows": 4,
            "test_rows": 3,
            "excluded_ambiguous_rows": 1,
            "train_inferred_sites": 1,
            "test_inferred_sites": 1,
            "train_dates": ["2024-01-01"],
            "test_dates": ["2024-02-01"],
        },
        "vehicle_classes": pd.DataFrame([{"observed_vehicles": 1}]),
    }

    with pytest.raises(ValueError, match="disagree on directional interval counts"):
        _dataset_overview_rows(verified)


def test_split_warning_preserves_manifest_text_and_has_safe_fallback() -> None:
    assert _split_warning({"warning": "  Audited split warning.  "}) == ("Audited split warning.")
    fallback = _split_warning({})
    assert "future dates and unseen inferred sites" in fallback
    assert "separate temporal or geographic validation" in fallback


def test_publication_safe_report_generates_pdf_and_bound_sidecar(tmp_path: Path) -> None:
    observed = _observed_bundle(tmp_path)
    output = tmp_path / "reports" / "observed_methods.pdf"

    result = make_observed_report(
        tmp_path,
        output,
        observed_output=observed,
        publication_safe=True,
    )

    assert result == output.resolve()
    assert output.read_bytes().startswith(b"%PDF-")
    assert output.stat().st_size > 1_000
    sidecar = json.loads(output.with_suffix(".provenance.json").read_text(encoding="utf-8"))
    assert sidecar["schema_version"] == "urbantwin.observed-report/v1"
    assert sidecar["publication_safe"] is True
    assert sidecar["data_kind"] == "publication_safe_methods_summary"
    assert sidecar["config_sha256"] == sha256_file(
        tmp_path / "configs" / "observed_traffic.yaml"
    )
    assert sidecar["code_revision"] == "c" * 40
    assert sidecar["report_sha256"] == sha256_file(output)


def test_full_report_generates_from_provenance_bound_figures(tmp_path: Path) -> None:
    observed = _observed_bundle(tmp_path)
    output = tmp_path / "reports" / "observed_private.pdf"

    result = make_observed_report(tmp_path, output, observed_output=observed)

    assert result == output.resolve()
    assert output.read_bytes().startswith(b"%PDF-")
    assert output.stat().st_size > 5_000
    sidecar = json.loads(output.with_suffix(".provenance.json").read_text(encoding="utf-8"))
    assert sidecar["schema_version"] == "urbantwin.observed-report/v1"
    assert sidecar["publication_safe"] is False
    assert sidecar["data_kind"] == "research_report_mixed_sources_explicitly_separated"
    assert sidecar["observed_provenance_sha256"] == sha256_file(
        observed / "provenance.json"
    )
    assert "configs/observed_traffic.yaml" in sidecar["input_hashes"]
    assert "results/observed_traffic/observed_metrics.csv" in sidecar["input_hashes"]
    assert sidecar["report_sha256"] == sha256_file(output)
    assert sidecar["input_hashes"]


def test_full_report_rejects_unbound_required_figure(tmp_path: Path) -> None:
    observed = _observed_bundle(tmp_path)
    provenance_path = observed / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["artifact_hashes"].pop("figures/observed_model_comparison.png")
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    safe_output = tmp_path / "reports" / "safe-without-restricted-figures.pdf"
    output = tmp_path / "reports" / "unbound.pdf"

    assert make_observed_report(
        tmp_path,
        safe_output,
        observed_output=observed,
        publication_safe=True,
    ) == safe_output.resolve()
    assert safe_output.read_bytes().startswith(b"%PDF-")

    with pytest.raises(
        ObservedArtifactIntegrityError,
        match="lacks provenance-bound report figures",
    ):
        make_observed_report(tmp_path, output, observed_output=observed)

    assert not output.exists()
    assert not output.with_suffix(".provenance.json").exists()


@pytest.mark.parametrize("damage", ["missing", "corrupt"])
def test_full_report_rejects_missing_or_corrupted_figure(
    tmp_path: Path, damage: str
) -> None:
    observed = _observed_bundle(tmp_path)
    figure = observed / "figures" / "observed_model_comparison.png"
    if damage == "missing":
        figure.unlink()
    else:
        figure.write_bytes(b"not a valid provenance-bound image")

    output = tmp_path / "reports" / f"{damage}-figure.pdf"
    with pytest.raises(
        ObservedArtifactIntegrityError,
        match="missing or hash-mismatched",
    ):
        make_observed_report(tmp_path, output, observed_output=observed)

    assert not output.exists()


def test_optional_context_figures_must_be_provenance_bound() -> None:
    core_figures = {
        "figures/survey_segments_measured_flow.png",
        "figures/selected_link_measured_flow.png",
        "figures/observed_model_comparison.png",
        "figures/observed_vs_estimated.png",
        "figures/online_persistence_diagnostic.png",
    }
    verified = {"hashes": {name: "synthetic-hash" for name in core_figures}}

    _require_report_figures(
        verified,
        has_eo_context=False,
        has_road_context=False,
    )
    with pytest.raises(ObservedArtifactIntegrityError) as error:
        _require_report_figures(
            verified,
            has_eo_context=True,
            has_road_context=True,
        )

    assert "selected_link_historic_eo_context.png" in str(error.value)
    assert "selected_link_current_osm_context.png" in str(error.value)


def test_report_rejects_missing_or_corrupted_observed_inputs(tmp_path: Path) -> None:
    missing_output = tmp_path / "reports" / "missing.pdf"
    with pytest.raises(FileNotFoundError, match="No validated observed-traffic study"):
        make_observed_report(
            tmp_path,
            missing_output,
            observed_output=tmp_path / "missing-observed-results",
            publication_safe=True,
        )
    assert not missing_output.exists()

    corrupted_root = tmp_path / "corrupted"
    observed = _observed_bundle(corrupted_root)
    metrics = observed / "observed_metrics.csv"
    metrics.write_text(
        metrics.read_text(encoding="utf-8") + "corrupt\n",
        encoding="utf-8",
    )
    corrupted_output = tmp_path / "reports" / "corrupted.pdf"
    with pytest.raises(ObservedArtifactIntegrityError, match="hash-mismatched"):
        make_observed_report(
            corrupted_root,
            corrupted_output,
            observed_output=observed,
            publication_safe=True,
        )
    assert not corrupted_output.exists()
    assert not corrupted_output.with_suffix(".provenance.json").exists()


def test_report_embeds_portable_fonts_and_enhances_map_without_overwriting(
    tmp_path: Path,
) -> None:
    _register_pdf_fonts()
    assert BODY_FONT in pdfmetrics.getRegisteredFontNames()
    assert BOLD_FONT in pdfmetrics.getRegisteredFontNames()

    source = tmp_path / "survey.png"
    PILImage.new("RGB", (240, 180), "white").save(source)
    before = source.read_bytes()
    image = _survey_map_image(source, 240, 180)

    assert image.drawWidth > 0
    assert image.drawHeight > 0
    assert source.read_bytes() == before
