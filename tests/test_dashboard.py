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
        "Measured Cairo traffic",
        "Measured-demand simulation",
        "Abstract simulation",
    ):
        assert label in source
    assert "validate_observed_artifacts" in source
    assert "validate_geoai_artifacts" in source
    assert "validate_traffic_artifacts" in source
    assert "scientifically blocked" in source
    assert "No traffic calibration" in source
    assert "URBANTWIN_PUBLICATION_SAFE" in source
    assert "Random forest/combined is selected from training CV" in source
    assert "No model ranking, exact score, or holdout outcome is reproduced" in source
    assert "EO variables are not included" in source
    assert "Contains modified Copernicus Sentinel data 2024" in source
    assert "© ESA WorldCover project 2021" in source
    assert "Near-survey" not in source
    assert "Retrospective Sentinel-2 context (2024)" in source
