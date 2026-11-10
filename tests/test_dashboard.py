from pathlib import Path

DASHBOARD = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "urbantwin"
    / "visualization"
    / "dashboard.py"
)


def test_dashboard_source_uses_modern_width_and_accessibility_parameters() -> None:
    source = DASHBOARD.read_text(encoding="utf-8")

    assert "use_container_width" not in source
    assert 'alt="' in source
    assert 'width="stretch"' in source


def test_dashboard_keeps_evidence_sections_and_validators_separate() -> None:
    source = DASHBOARD.read_text(encoding="utf-8")
    compile(source, str(DASHBOARD), "exec")

    for label in (
        "Observed EO",
        "Observed TfC traffic",
        "Measured-demand simulation",
        "Abstract simulation",
    ):
        assert label in source
    assert "validate_observed_artifacts" in source
    assert "validate_geoai_artifacts" in source
    assert "validate_traffic_artifacts" in source
    assert "scientifically blocked" in source
    assert "No traffic calibration" in source
