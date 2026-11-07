# UrbanTwin-AI — Research case study (initial implementation)

**Research status (November 2026):** Source code and offline synthetic fixtures are implemented. Real Earth-observation/OSM acquisition still requires network execution, and full-budget traffic outcomes still require a reviewed run with Eclipse SUMO installed. **No performance claims are being made yet.**

## Motivation

Urban research frequently combines Earth-observation mapping and traffic models, but it is easy to conflate city imagery with observed mobility conditions. This prototype separates a geospatial description of a Cairo study area from a reproducible and explicitly synthetic traffic signal optimization experiment.

## Problem statement

How can an Earth-observation and traffic-control workflow expose source provenance, spatial validation, simulator assumptions and failure modes so that AI-driven urban analysis is testable rather than just a visualization?

## Individual contribution

Developed the integrated Python research framework; public Sentinel-2 and ESA WorldCover STAC acquisition and geospatial alignment; spatially blocked land-cover reference-label benchmarks; deterministic abstract SUMO scenario generator; GA, PSO, and equal-budget random-search signal-timing optimizers; evaluation scripts, CI tests, dashboard, and research-report generator. If significant contributions from others are incorporated later, credit them explicitly.

## Method

1. A small WGS84-defined Cairo AOI is projected to EPSG:32636 and sampled at 10m.
2. A cloud-screened multi-scene Sentinel-2 L2A median is aligned with the 2021 ESA WorldCover class raster.
3. NDVI/NDBI and road-network context are recorded with data sources, STAC item IDs, CRS, and limitations.
4. Random Forest and histogram gradient boosting are tested over repeated group-disjoint spatial splits, with and without two spectral indices. Eligible labels are selected from training support alone; unsupported classes and every per-class confusion matrix are retained. **Evaluation is against WorldCover pseudo-labels, not independent surveyed ground truth**.
5. A generic four-way SUMO junction receives low/medium/high generated demand. Fixed green durations are compared against GA-, PSO-, and equal-budget random-search timings. Optimization seeds are separate from paired evaluation seeds.
6. The dashboard shows real EO results and simulator-only SUMO results in separately labeled panels after artifact hashes and provenance agree.

## Observed results

**Not yet available.** Run `urbantwin eo-fetch`, `urbantwin geo-train`, `urbantwin traffic-run` and `urbantwin report` locally, then replace this paragraph with measured findings and reference the result files.

## Limitations and responsible interpretation

- One geographic AOI and block-based tests cannot establish cross-region generalization.
- WorldCover-label agreement does not mean ground-truth classification accuracy.
- The generated SUMO network is not Cairo geometry and demand is not observed demand; improvements are simulation-only.
- The 2021 EO labels, composited optical scenes and contemporary OSM network differ in temporal origin.
- Completed-trip waiting times must be interpreted alongside generated, inserted, completed, unfinished, and pending-to-insert counts to avoid survivorship bias.
- Simulator-estimated CO₂ depends on the recorded SUMO/HBEFA emissions-class assumption and is not measured Cairo air pollution.
- The project does not derive or verify a causal relationship between satellite land cover and intersection signal performance.

## Reproducibility

The README supplies exact Windows PowerShell commands. Outputs preserve the exact `data_kind` labels `real`, `synthetic`, `simulation_abstract`, or `simulation_external_user_network`. Generated reports reject stale or mismatched metrics and write a SHA-256 provenance sidecar. The `demo` command uses synthetic fixtures and must not be cited as empirical evidence.

## Research relevance

- **SMACCs:** Mobility simulation, sustainability, interpretable limitations.
- **Copernicus Digital Earth / GeoTech:** EO/GIS processing, CRS, spectral indices, provenance and validation.
- **AISS:** Urban sustainability and responsible use of models.
- **Toulouse / Lille:** Algorithms, statistical evaluation, experimental comparability, ML ablations.
