# Results and evaluation integrity

No research results have been pre-populated in this repository.

Run:

1. `urbantwin eo-fetch`
2. `urbantwin geo-train` -> aggregate/split GeoAI metrics, per-class confusion matrices, unsupported-class reporting, and input hashes.
3. `urbantwin roads` -> `results/roads.json`.
4. `urbantwin traffic-run` -> metrics, paired-seed summaries, and simulator provenance under `results/traffic/`.
5. `urbantwin report` -> `reports/generated_research_report.pdf` plus its provenance sidecar.

The generated outputs are ignored by Git by default because they may be large, machine-specific or contain licensed data. Outputs retain the exact `data_kind` labels `real`, `synthetic`, `simulation_abstract`, or `simulation_external_user_network`. **After verifying genuine measurements and checking data sharing terms**, explicitly add only aggregate CSVs, figures, and a redacted provenance report (without signed asset URLs), e.g. `git add -f results/geoai/geoai_metrics.csv`.

`results/demo/` contains SYNTHETIC fixtures and synthetic benchmark outputs. Never use those scores as authentic research evidence. `results/traffic_quick/` contains low-budget quick simulations and should not be presented as a full study.

SUMO simulation does not automatically imply real-world improvement; the generated network is abstract.
