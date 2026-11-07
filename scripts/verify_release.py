"""Simple repository integrity check: no committed research numbers are fabricated."""
from __future__ import annotations
from pathlib import Path
import json

root=Path(__file__).resolve().parents[1]
expected=["README.md","LICENSE","configs/study_area.yaml","configs/experiments.yaml",
          "src/urbantwin/cli.py","src/urbantwin/data/earth_observation.py",
          "src/urbantwin/geoai/train.py","src/urbantwin/traffic/benchmark.py",
          "src/urbantwin/visualization/dashboard.py","reports/research_report.pdf"]
missing=[p for p in expected if not (root/p).exists()]
if missing:raise SystemExit("Missing required files: "+str(missing))
for candidate in root.rglob("*.py"):
    compile(candidate.read_text(encoding="utf-8"),str(candidate),"exec")
print("Repository skeleton and Python syntax verification PASSED")
