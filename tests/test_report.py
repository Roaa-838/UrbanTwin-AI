import pytest
from urbantwin.visualization.report_writer import make_report


def test_draft_report_renders(tmp_path):
    pdf=tmp_path/"report.pdf"
    make_report(pdf,include_draft=True)
    assert pdf.exists() and pdf.stat().st_size>1500
