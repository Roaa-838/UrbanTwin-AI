from __future__ import annotations

import inspect
from pathlib import Path

from PIL import Image as PILImage
from reportlab.pdfbase import pdfmetrics

from urbantwin.cli import _parser
from urbantwin.observed.report import (
    BODY_FONT,
    BOLD_FONT,
    _register_pdf_fonts,
    _survey_map_image,
    make_observed_report,
)

REPORT = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "urbantwin"
    / "observed"
    / "report.py"
)


def test_report_defaults_to_full_private_mode_and_supports_safe_publication() -> None:
    parameter = inspect.signature(make_observed_report).parameters["publication_safe"]

    assert parameter.default is False
    source = REPORT.read_text(encoding="utf-8")
    assert "Measured Greater Cairo Traffic" in source
    assert "Selected using training-only grouped CV" in source
    assert "Lowest observed final-holdout point estimate" in source
    assert "EO variables are absent from the multi-site prediction benchmark" in source
    assert "Contains modified Copernicus Sentinel data 2024" in source
    assert "© ESA WorldCover project 2021" in source
    assert "publication-safe methods summary" in source.lower()
    assert "No model ranking or " in source
    assert "holdout outcome is reproduced" in source
    assert "https://www.openstreetmap.org/copyright" in source


def test_observed_report_cli_exposes_explicit_publication_safe_mode() -> None:
    arguments = _parser().parse_args(["observed-report", "--publication-safe"])

    assert arguments.publication_safe is True


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
