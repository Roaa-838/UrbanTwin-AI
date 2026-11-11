"""Offline release-integrity checks; this does not verify online data or SUMO runs."""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 compatibility; supplied by the dev/all extras.
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = [
    "README.md",
    "LICENSE",
    "configs/study_area.yaml",
    "configs/experiments.yaml",
    "configs/observed_traffic.yaml",
    "configs/observed_eo_2024.yaml",
    "src/urbantwin/cli.py",
    "src/urbantwin/data/__init__.py",
    "src/urbantwin/data/earth_observation.py",
    "src/urbantwin/data/spatial_features.py",
    "src/urbantwin/data/demo.py",
    "src/urbantwin/data/observed_traffic.py",
    "src/urbantwin/data/observed_study.py",
    "src/urbantwin/geoai/train.py",
    "src/urbantwin/observed/artifacts.py",
    "src/urbantwin/observed/benchmark.py",
    "src/urbantwin/observed/features.py",
    "src/urbantwin/observed/mlflow_tracking.py",
    "src/urbantwin/observed/roads.py",
    "src/urbantwin/observed/report.py",
    "src/urbantwin/traffic/benchmark.py",
    "src/urbantwin/traffic/checkpoints.py",
    "src/urbantwin/traffic/mlflow_tracking.py",
    "src/urbantwin/visualization/dashboard.py",
    "docs/real_data_feasibility.md",
    "docs/observed_traffic_case_study.md",
    "docs/publication_scope.md",
    "docs/portfolio_summary.md",
    "notebooks/03_observed_traffic_measurements.ipynb",
    "notebooks/04_observed_traffic_benchmark.ipynb",
    "reports/research_report.pdf",
]
TEXT_SUFFIXES = {
    ".cfg",
    ".csv",
    ".env",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".rst",
    ".toml",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}
SIGNED_URL_PATTERNS = (
    re.compile(r"[?&](?:sig|signature|se|sp|sv|st)=", re.IGNORECASE),
    re.compile(r"X-Amz-(?:Credential|Signature|Security-Token)=", re.IGNORECASE),
)
SECRET_PATTERNS = (
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"(?i)(?:api[_-]?key|client[_-]?secret|access[_-]?token)\s*[=:]\s*['\"][^'\"]{12,}"),
)


def fail(message: str) -> None:
    raise SystemExit(message)


def tracked_files() -> list[Path]:
    try:
        result = subprocess.run(
            ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError):
        # Source archives do not necessarily include .git. Limit fallback scanning
        # to repository content and avoid local environments/caches.
        ignored = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".ruff_cache"}
        return [
            path.relative_to(ROOT)
            for path in ROOT.rglob("*")
            if path.is_file() and not any(part in ignored for part in path.parts)
        ]
    return [Path(raw.decode("utf-8")) for raw in result.stdout.split(b"\0") if raw]


missing = [relative for relative in EXPECTED if not (ROOT / relative).is_file()]
if missing:
    fail(f"Missing required files: {missing}")

for folder in (ROOT / "src", ROOT / "scripts", ROOT / "tests"):
    for candidate in folder.rglob("*.py"):
        try:
            compile(candidate.read_text(encoding="utf-8"), str(candidate), "exec")
        except (OSError, UnicodeError, SyntaxError) as exc:
            fail(f"Python source verification failed for {candidate}: {exc}")

project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
metadata = project.get("project", {})
if metadata.get("license") != "MIT" or "LICENSE" not in metadata.get("license-files", []):
    fail("pyproject.toml must use a PEP 639 SPDX license expression and license file")
for extra in ("app", "all"):
    dependencies = metadata.get("optional-dependencies", {}).get(extra, [])
    if not any(dependency.lower().startswith("pydeck") for dependency in dependencies):
        fail(f"The {extra!r} dependency group must declare pydeck explicitly")
optional = metadata.get("optional-dependencies", {})
mlflow_dependencies = optional.get("mlflow", [])
if not any(dependency.lower().startswith("mlflow") for dependency in mlflow_dependencies):
    fail("The optional 'mlflow' dependency group must declare MLflow")
if any(dependency.lower().startswith("mlflow") for dependency in metadata.get("dependencies", [])):
    fail("MLflow must remain optional and must not be a core dependency")

tracked = tracked_files()
large_data_suffixes = {".jp2", ".npy", ".npz", ".tif", ".tiff", ".zarr"}
for relative in tracked:
    path = ROOT / relative
    if not path.is_file():
        continue
    if path.suffix.lower() in large_data_suffixes:
        fail(f"Generated raster/array asset must not be committed: {relative}")
    if path.stat().st_size > 5 * 1024 * 1024:
        fail(f"Tracked file exceeds the 5 MiB release limit: {relative}")
    if path.suffix.lower() not in TEXT_SUFFIXES:
        continue
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeError:
        continue
    for pattern in SIGNED_URL_PATTERNS:
        if pattern.search(text):
            fail(f"Possible signed asset URL in tracked file: {relative}")
    for pattern in SECRET_PATTERNS:
        if pattern.search(text):
            fail(f"Possible secret in tracked file: {relative}")

dashboard = (ROOT / "src/urbantwin/visualization/dashboard.py").read_text(encoding="utf-8")
deprecated_name = "use" + "_container_width"
if deprecated_name in dashboard:
    fail("Dashboard still uses Streamlit's deprecated container-width argument")

# Static documents must distinguish executed evidence from deliberately blocked
# phases. Generated row-level/result bundles remain ignored.
case_study = (ROOT / "docs/research_case_study.md").read_text(encoding="utf-8").lower()
report = (ROOT / "reports/research_report.md").read_text(encoding="utf-8").lower()
readme = (ROOT / "README.md").read_text(encoding="utf-8").lower()
required_case_phrases = (
    "measured-demand signal-control simulation was **not run**",
    "198 logical sumo requests",
    "different task",
    "not a deployed",
)
if any(phrase not in case_study for phrase in required_case_phrases):
    fail("Research case study does not preserve executed/blocked evidence boundaries")
required_readme_phrases = (
    "real_observed_traffic_analysis",
    "measured-demand signal",
    "not a calibrated",
    "urbantwin-observed-traffic",
)
if any(phrase not in readme for phrase in required_readme_phrases):
    fail("README does not document the observed study and its scientific gate")
if "not yet generated" not in report:
    fail("Tracked methodology report must remain explicit about its static status")

print(
    json.dumps(
        {
            "status": "PASSED",
            "scope": "offline repository structure, syntax, packaging, and tracked-file hygiene",
            "not_verified": [
                "live network downloads in this offline verifier",
                "fresh live reproduction of current OSM retrieval (local bundle separately validated)",
                "measured-demand signal SUMO (not run by scientific design)",
                "external reproduction of local Git-ignored experiment artifacts",
            ],
        },
        indent=2,
    )
)
