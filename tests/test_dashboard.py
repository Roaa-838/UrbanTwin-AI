from __future__ import annotations

import ast
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from tests.test_observed_artifacts import _observed_bundle
from urbantwin.common import sha256_file
from urbantwin.observed.artifacts import (
    ObservedArtifactIntegrityError,
    validate_observed_artifacts,
)
from urbantwin.visualization import dashboard

DASHBOARD = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "urbantwin"
    / "visualization"
    / "dashboard.py"
)


def _streamlit_calls() -> list[ast.Call]:
    tree = ast.parse(DASHBOARD.read_text(encoding="utf-8"), filename=str(DASHBOARD))
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "st"
    ]


def test_dashboard_display_calls_use_current_accessible_parameters() -> None:
    calls = _streamlit_calls()
    display_names = {
        "altair_chart",
        "dataframe",
        "image",
        "pydeck_chart",
        "table",
    }
    stretchable_names = display_names - {"table"}
    display_calls = [call for call in calls if call.func.attr in display_names]

    assert display_calls
    assert all(
        keyword.arg != "use_container_width"
        for call in calls
        for keyword in call.keywords
    )
    for call in display_calls:
        keywords = {keyword.arg: keyword.value for keyword in call.keywords}
        assert "alt" in keywords, f"st.{call.func.attr} is missing alt text"
        if call.func.attr in stretchable_names:
            assert "width" in keywords, f"st.{call.func.attr} is missing an explicit width"


def _write_raster_bundle(
    directory: Path,
    *,
    labels_shape: tuple[int, int] = (2, 3),
    recorded_hash: str | None = None,
    band_order: list[str] | None = None,
) -> None:
    directory.mkdir(parents=True)
    bands = np.arange(30, dtype=np.float32).reshape(5, 2, 3)
    labels = np.ones(labels_shape, dtype=np.uint8)
    stack = directory / "eo_stack.npz"
    np.savez_compressed(stack, bands=bands, labels=labels)
    provenance = {
        "data_kind": "synthetic",
        "eo_stack_sha256": recorded_hash or sha256_file(stack),
        "band_order": band_order or ["B02", "B03", "B04", "B08", "B11"],
        "height": labels_shape[0],
        "width": labels_shape[1],
    }
    (directory / "provenance.json").write_text(
        json.dumps(provenance), encoding="utf-8"
    )


def test_dashboard_raster_loader_accepts_bound_data_and_rejects_invalid_bundles(
    tmp_path: Path,
) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    assert dashboard._load_raster(str(empty), "synthetic") is None

    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    np.savez_compressed(
        incomplete / "eo_stack.npz",
        bands=np.zeros((5, 2, 3), dtype=np.float32),
        labels=np.ones((2, 3), dtype=np.uint8),
    )
    with pytest.raises(ValueError, match="generated together"):
        dashboard._load_raster(str(incomplete), "synthetic")

    valid = tmp_path / "valid"
    _write_raster_bundle(valid)
    bands, labels, provenance = dashboard._load_raster(str(valid), "synthetic")
    assert bands.shape == (5, 2, 3)
    assert labels.shape == (2, 3)
    assert provenance["data_kind"] == "synthetic"

    bad_hash = tmp_path / "bad-hash"
    _write_raster_bundle(bad_hash, recorded_hash="0" * 64)
    with pytest.raises(ValueError, match="recorded provenance hash"):
        dashboard._load_raster(str(bad_hash), "synthetic")

    misaligned = tmp_path / "misaligned"
    _write_raster_bundle(misaligned, labels_shape=(3, 2))
    with pytest.raises(ValueError, match="not aligned"):
        dashboard._load_raster(str(misaligned), "synthetic")

    wrong_order = tmp_path / "wrong-order"
    _write_raster_bundle(wrong_order, band_order=["B03", "B02", "B04", "B08", "B11"])
    with pytest.raises(ValueError, match="unsupported band order"):
        dashboard._load_raster(str(wrong_order), "synthetic")


def test_dashboard_reports_rejected_observed_artifacts() -> None:
    with (
        patch.object(
            dashboard,
            "_observed_bundle",
            side_effect=ObservedArtifactIntegrityError("synthetic corruption"),
        ),
        patch.object(dashboard.st, "error") as error,
    ):
        assert dashboard._load_verified_observed() is None

    error.assert_called_once()
    assert "artifacts rejected" in error.call_args.args[0]


def _render_section(section: str) -> AppTest:
    with (
        patch(
            "urbantwin.observed.artifacts.validate_observed_artifacts",
            return_value=None,
        ),
        patch(
            "urbantwin.observed.roads.validate_observed_road_bundle",
            return_value=None,
        ),
        patch(
            "urbantwin.visualization.report_writer.validate_geoai_artifacts",
            return_value=None,
        ),
        patch(
            "urbantwin.visualization.report_writer.validate_traffic_artifacts",
            return_value=None,
        ),
    ):
        app = AppTest.from_file(DASHBOARD, default_timeout=20)
        app.session_state["research_section"] = section
        return app.run()


def _dashboard_script() -> None:
    from urbantwin.visualization.dashboard import main

    main()


def _synthetic_raster() -> tuple[np.ndarray, np.ndarray, dict]:
    labels = np.ones((2, 2), dtype=np.uint8)
    provenance = {
        "valid_pixel_count": 4,
        "grid_pixel_count": 4,
        "configuration": {"sentinel_period": ["2024-10-24", "2024-11-03"]},
        "width": 2,
        "height": 2,
        "crs": "EPSG:32636",
    }
    return np.zeros((5, 2, 2), dtype=np.float32), labels, provenance


def test_dashboard_renders_scientific_gate_without_local_artifacts() -> None:
    app = _render_section("Measured-demand simulation")

    assert not app.exception
    assert app.title[0].value == "UrbanTwin-AI research evidence"
    assert app.header[0].value == "Measured-demand simulation"
    assert any("scientifically blocked" in warning.value for warning in app.warning)
    assert any("No traffic calibration" in caption.value for caption in app.caption)
    control = app.get("segmented_control")[0]
    assert control.value == "Measured-demand simulation"


def test_dashboard_missing_artifacts_render_actionable_states() -> None:
    measured = _render_section("Measured Cairo traffic")
    abstract = _render_section("Abstract simulation")

    assert not measured.exception
    assert measured.header[0].value == "Measured Cairo traffic"
    assert any("Publication-safe mode omits" in item.value for item in measured.info)

    assert not abstract.exception
    assert abstract.header[0].value == "Existing abstract SUMO experiment"
    assert any("No complete provenance-verified" in item.value for item in abstract.warning)


def test_non_public_measured_section_renders_missing_bundle_state() -> None:
    with (
        patch.object(dashboard, "PUBLICATION_SAFE", False),
        patch.object(dashboard, "_load_raster", return_value=None),
        patch.object(dashboard, "_load_verified_observed", return_value=None),
    ):
        app = AppTest.from_function(_dashboard_script, default_timeout=20).run()
        app.get_by_key("research_section").set_value("Measured Cairo traffic").run()

    assert not app.exception
    assert app.header[0].value == "Measured Cairo traffic"
    assert any("No complete provenance-verified" in item.value for item in app.warning)


def test_non_public_measured_section_renders_verified_synthetic_bundle(
    tmp_path: Path,
) -> None:
    output = _observed_bundle(tmp_path)
    bundle = validate_observed_artifacts(tmp_path, output)
    assert bundle is not None
    with (
        patch.object(dashboard, "PUBLICATION_SAFE", False),
        patch.object(dashboard, "_load_raster", return_value=None),
        patch.object(dashboard, "_load_verified_observed", return_value=bundle),
    ):
        app = AppTest.from_function(_dashboard_script, default_timeout=20).run()
        app.get_by_key("research_section").set_value("Measured Cairo traffic").run()

    assert not app.exception
    assert app.header[0].value == "Measured Cairo traffic"
    metrics = {item.label: item.value for item in app.metric}
    assert metrics["Measured intervals"] == "7"
    assert metrics["Observed vehicles"] == "78"
    assert metrics["Inferred site groups"] == "6"
    assert metrics["Eligible intervals"] == "1"
    assert app.table
    assert app.dataframe
    assert len(app.image) == 2


def test_dashboard_section_control_navigates_and_renders_each_section() -> None:
    with (
        patch.object(dashboard, "PUBLICATION_SAFE", True),
        patch.object(dashboard, "_load_raster", return_value=None),
        patch.object(dashboard, "_load_verified_observed", return_value=None),
        patch.object(dashboard, "validate_observed_road_bundle", return_value=None),
        patch.object(dashboard, "validate_traffic_artifacts", return_value=None),
    ):
        app = AppTest.from_function(_dashboard_script, default_timeout=20).run()
        assert not app.exception
        assert app.header[0].value == "Observed Earth observation"

        expected_headers = {
            "Measured Cairo traffic": "Measured Cairo traffic",
            "Measured-demand simulation": "Measured-demand simulation",
            "Abstract simulation": "Existing abstract SUMO experiment",
            "Observed EO": "Observed Earth observation",
        }
        for section, expected_header in expected_headers.items():
            app.get_by_key("research_section").set_value(section).run()
            assert not app.exception
            assert app.header[0].value == expected_header


def test_publication_safe_eo_section_never_loads_restricted_observed_bundle() -> None:
    with (
        patch.object(dashboard, "PUBLICATION_SAFE", True),
        patch.object(dashboard, "_load_raster", return_value=_synthetic_raster()),
        patch.object(dashboard, "_render_eo_map"),
        patch.object(dashboard, "validate_geoai_artifacts", return_value=None),
        patch.object(
            dashboard,
            "_load_verified_observed",
            side_effect=AssertionError("restricted observed bundle was loaded"),
        ) as observed_loader,
    ):
        app = AppTest.from_function(_dashboard_script, default_timeout=20).run()
        assert not app.exception
        app.get_by_key("research_section").set_value("Measured Cairo traffic").run()

    assert not app.exception
    assert app.header[0].value == "Measured Cairo traffic"
    assert any("Publication-safe mode omits" in item.value for item in app.info)
    observed_loader.assert_not_called()


def test_non_public_eo_section_renders_valid_observed_context(tmp_path: Path) -> None:
    context = [
        {
            "buffer_m": 50,
            "valid_pixel_coverage": 0.9,
            "mean_ndvi": 0.1,
            "mean_ndbi": 0.2,
            "worldcover_built_up_fraction": 0.7,
            "worldcover_vegetated_fraction": 0.1,
        }
    ]
    (tmp_path / "historic_eo_context.json").write_text(
        json.dumps(context), encoding="utf-8"
    )
    with (
        patch.object(dashboard, "PUBLICATION_SAFE", False),
        patch.object(dashboard, "_load_raster", return_value=_synthetic_raster()),
        patch.object(dashboard, "_render_eo_map"),
        patch.object(dashboard, "validate_geoai_artifacts", return_value=None),
        patch.object(
            dashboard,
            "_load_verified_observed",
            return_value={"output_dir": str(tmp_path)},
        ),
    ):
        app = AppTest.from_function(_dashboard_script, default_timeout=20).run()

    assert not app.exception
    assert any(item.value == "Selected observed-link context" for item in app.subheader)
    assert len(app.dataframe) == 1
    assert app.dataframe[0].value["buffer_m"].tolist() == [50]


@pytest.mark.parametrize(
    ("contents", "element", "message"),
    [
        (None, "error", "artifact is unavailable"),
        ("{not-json", "error", "artifact is invalid JSON"),
        (json.dumps({"buffer_m": 50}), "error", "must contain a JSON list"),
        ("[]", "info", "No retrospective observed-link EO context"),
        (json.dumps([{"buffer_m": 50}]), "error", "lacks required columns"),
    ],
)
def test_non_public_eo_section_handles_missing_or_invalid_context_artifact(
    tmp_path: Path,
    contents: str | None,
    element: str,
    message: str,
) -> None:
    if contents is not None:
        (tmp_path / "historic_eo_context.json").write_text(contents, encoding="utf-8")
    with (
        patch.object(dashboard, "PUBLICATION_SAFE", False),
        patch.object(dashboard, "_load_raster", return_value=_synthetic_raster()),
        patch.object(dashboard, "_render_eo_map"),
        patch.object(dashboard, "validate_geoai_artifacts", return_value=None),
        patch.object(
            dashboard,
            "_load_verified_observed",
            return_value={"output_dir": str(tmp_path)},
        ),
    ):
        app = AppTest.from_function(_dashboard_script, default_timeout=20).run()

    assert not app.exception
    rendered = getattr(app, element)
    assert any(message in item.value for item in rendered)


def test_non_public_eo_section_renders_observed_validation_failure() -> None:
    with (
        patch.object(dashboard, "PUBLICATION_SAFE", False),
        patch.object(dashboard, "_load_raster", return_value=_synthetic_raster()),
        patch.object(dashboard, "_render_eo_map"),
        patch.object(dashboard, "validate_geoai_artifacts", return_value=None),
        patch.object(
            dashboard,
            "_observed_bundle",
            side_effect=ObservedArtifactIntegrityError("synthetic checksum mismatch"),
        ),
    ):
        app = AppTest.from_function(_dashboard_script, default_timeout=20).run()

    assert not app.exception
    assert any("artifacts rejected" in item.value for item in app.error)
    assert any("synthetic checksum mismatch" in item.value for item in app.error)
