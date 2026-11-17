from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = {
    "01_geospatial_analysis.ipynb": "validate_geoai_artifacts",
    "02_experiment_results.ipynb": "validate_traffic_artifacts",
    "03_observed_traffic_measurements.ipynb": "load_normalized_counts",
    "04_observed_traffic_benchmark.ipynb": "validate_observed_artifacts",
}


def _load_notebook(name: str) -> dict:
    return json.loads((ROOT / "notebooks" / name).read_text(encoding="utf-8"))


@pytest.mark.parametrize(("name", "required_call"), NOTEBOOKS.items())
def test_research_notebooks_are_clean_reproducible_sources(
    name: str,
    required_call: str,
) -> None:
    notebook = _load_notebook(name)

    assert notebook["nbformat"] == 4
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert code_cells
    assert all(cell.get("execution_count") is None for cell in code_cells)
    assert all(cell.get("outputs") == [] for cell in code_cells)

    code = "\n\n".join("".join(cell["source"]) for cell in code_cells)
    tree = ast.parse(code, filename=str(ROOT / "notebooks" / name))
    called_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert required_call in called_names
    assert "write_demo" not in called_names


def test_simulation_notebook_enforces_abstract_source_label() -> None:
    notebook = _load_notebook("02_experiment_results.ipynb")
    code = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    constants = {
        node.value
        for node in ast.walk(ast.parse(code))
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }

    assert "simulation_abstract" in constants
