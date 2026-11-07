# Results and evaluation integrity

No research results have been pre-populated in this repository.

Run:

1. `urbantwin eo-fetch`
2. `urbantwin geo-train` -> `results/geoai/geoai_metrics.csv` and per-class measurements.
3. `urbantwin roads` -> `results/roads.json`.
4. `urbantwin traffic-run` -> `results/traffic/traffic_metrics.csv` and summary.
5. `urbantwin report` -> `reports/generated_research_report.pdf`.

The generated outputs are ignored by Git by default because they may be large, machine-specific or contain licensed data. **After verifying genuine measurements and checking data sharing terms**, explicitly add only aggregate CSVs, figures, and a redacted provenance report (without signed asset URLs), e.g. `git add -f results/geoai/geoai_metrics.csv`.

`results/demo/` contains SYNTHETIC fixtures and synthetic benchmark outputs. Never use those scores as authentic research evidence. `results/traffic_quick/` contains low-budget quick simulations and should not be presented as a full study.

SUMO simulation does not automatically imply real-world improvement; the generated network is abstract.
