# Results and evaluation integrity

Generated experiment outputs are ignored by Git by default. The current validated local working
copy contains four scientifically distinct result families:

1. `results/geoai/`: real Sentinel-2/WorldCover 2021 agreement benchmarks (`data_kind: real`).
2. `results/observed_traffic/`: measured TfC 2024 directional-link characterization and the frozen
   joint future-date/unseen-inferred-site benchmark (`data_kind: real_observed_traffic_analysis`).
3. `results/traffic/`: the completed 198-request abstract SUMO comparison
   (`data_kind: simulation_abstract`). It is not measured Cairo demand.
4. `results/geoai_spatial_validation_v1/`: the separate buffered, three-class GeoAI robustness
   study (`data_kind: real_eo_with_historic_worldcover_pseudo_labels`). It does not replace the
   six-class frozen GeoAI benchmark and is not independent field accuracy.

The completed abstract benchmark comprised 198 logical simulator requests: 162 physical executions
and 36 validated cache hits under SUMO 1.26.0. These execution counts describe the archived local
run, not measured traffic or a calibrated city model. Exact reproduction commands for every study
are maintained in the [reproducibility guide](../docs/REPRODUCIBILITY.md).

Its strict validator binds the official source hashes, normalized manifest, configuration, split,
predictions, aggregate metrics, figures and source labels. Measurement-free volume estimation and
online persistence are different information budgets and must not be compared as if they used the
same inputs. The measured-demand signal-control phase is **NOT RUN**: the release lacks turning
movements, complete routes and observed signal timing.

`results/demo/` contains synthetic software fixtures. `results/traffic_quick/` contains low-budget
simulator smoke results. Neither is empirical evidence. The full `results/traffic/` bundle is also
simulator-only: modeled CO₂ is not an environmental measurement, and no abstract SUMO result is a
calibrated Greater Cairo controller result.

TfC did not publish an explicit record-level license with the audited records. Public access is not
treated as an open-license grant. Keep raw and row-level traffic data, measured maps and reports local
unless documented publisher terms provide a legal basis for the intended release. Only expressly
permitted compact aggregates and redacted provenance should ever be forced into Git. Do not commit
large rasters, source archives, local MLflow SQLite files, caches, secrets or signed STAC asset URLs.
