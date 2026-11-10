# Results and evaluation integrity

Generated experiment outputs are ignored by Git by default. The current validated local working
copy contains three scientifically distinct result families:

1. `results/geoai/`: real Sentinel-2/WorldCover 2021 agreement benchmarks (`data_kind: real`).
2. `results/observed_traffic/`: measured TfC 2024 directional-link characterization and the frozen
   joint future-date/unseen-inferred-site benchmark (`data_kind: real_observed_traffic_analysis`).
3. `results/traffic/`: the completed 198-request abstract SUMO comparison
   (`data_kind: simulation_abstract`). It is not measured Cairo demand.

The observed study is reproduced with:

```powershell
urbantwin observed-study --resume --mlflow
urbantwin observed-report
```

Its strict validator binds the official source hashes, normalized manifest, configuration, split,
predictions, aggregate metrics, figures and source labels. Measurement-free volume estimation and
online persistence are different information budgets and must not be compared as if they used the
same inputs. The measured-demand signal-control phase is **NOT RUN**: the release lacks turning
movements, complete routes and observed signal timing.

`results/demo/` contains synthetic software fixtures. `results/traffic_quick/` contains low-budget
simulator smoke results. Neither is empirical evidence. The full `results/traffic/` bundle is also
simulator-only: modeled CO₂ is not an environmental measurement, and no abstract SUMO result is a
calibrated Greater Cairo controller result.

TfC did not publish an explicit reuse license with the audited records. Keep raw and row-level
derived traffic data, measured maps and reports local until written reuse terms are confirmed.
Only redistribution-permitted compact aggregates and redacted provenance should ever be forced into
Git. Do not commit large rasters, source archives, local MLflow SQLite files, caches, secrets or
signed STAC asset URLs.
