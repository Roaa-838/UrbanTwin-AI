# UrbanTwin-AI — evidence-bounded Greater Cairo research case study

**Verified local status (November 2026):** real Earth-observation workflows, the measured TfC
traffic analysis and the existing full abstract SUMO benchmark have been executed locally and pass
strict provenance validation. A measured-demand signal-control simulation was **not run** because
the available observations do not support it. This is a research prototype, not a deployed
controller or a calibrated Cairo digital twin.

## Research question

How can real Earth observation, measured directional traffic demand and reproducible
counterfactual simulation be studied together without blurring measured, remotely sensed,
third-party mapped, inferred and entirely simulated variables?

## Completed evidence streams

### Real Earth observation

The project aligns Sentinel-2 L2A bands and ESA WorldCover to a 10 m EPSG:32636 grid with explicit
SCL masking, continuous/categorical resampling and source-item provenance. The 2021 GeoAI experiment
uses repeated grouped spatial evaluation and reports agreement with WorldCover pseudo-labels—not
independent land-cover accuracy. A separate retrospective-context bundle contains real Sentinel-2
L2A scenes from 24 October, 29 October and 3 November 2024. The 3 November acquisition postdates the
selected 31 October survey and was unavailable to a prospective forecast. WorldCover 2021 remains a
historic reference, not 2024 ground truth. EO is absent from the multi-site traffic benchmark, so no
EO accuracy gain was tested.

### Measured TfC traffic

The official release contains 6,618 directional 20-minute records, 12 vehicle classes and
2,961,483 observed vehicles across 142 counted directional segments. The normalized local bundle
preserves source values, missingness, undefined direction semantics, unknown timezone and absent
sample-validity flags. Five OSM-ID conflicts affecting 156 rows remain visible and are excluded from
the primary spatial benchmark.

The source has no official physical-site key. A declared 250 m projected line-distance connected-
component rule reproducibly gives 60 analytical groups; these are inferred sites, never TfC IDs.
The final six survey dates (27 October–5 November 2024) form a frozen joint future-date/unseen-inferred-site
holdout: 1,207 rows from ten inferred sites, with no site overlap with the 5,255-row training set.
Five-fold grouped validation stays inside training. Because each directional segment occurs on one
date only, date and site effects cannot be separated.

Training-only GroupKFold development ranked Random Forest with combined features first (mean CV MAE
283.42). Histogram gradient boosting with combined time/static spatial features had the lowest
observed final-holdout MAE point estimate: 348.25 (95% site-bootstrap interval 239.46–487.46), with
RMSE 496.34 (309.46–654.29) vehicles per 20 minutes. The latter is a descriptive holdout ordering,
not retrospective model selection. Broad overlapping intervals preclude a superiority or
significance claim. Online one-step persistence has MAE 63.93 and RMSE 144.04 on 1,170 exactly
adjacent intervals, but it consumes the previous measured count and is therefore a different task.
The static geometry descriptors come from TfC-released OSM-derived lines dated July 2026.

### Existing abstract SUMO experiment

The existing configured benchmark completed 198 logical SUMO requests (162 physical executions and
36 validated cache hits) under SUMO 1.26.0. It compares fixed time, GA, PSO and equal-budget random
search with paired held-out seeds on a synthetic one-junction network. Its demand, queues, time loss,
throughput and modeled CO₂ are simulator-only and are not relabeled as observed Cairo performance.

## Why measured-demand SUMO was not run

TfC provides directional link totals, not intersection turning movements. The files contain no
complete routes, signal/controller IDs, observed phase states or timing, lane-level demand,
pedestrian operation, queues, speeds or travel times. The selected observed link is the 6th of
October Bridge motorway pair rather than a verified signalized junction. One link count cannot
identify a complete turning-demand matrix. The scientific gate therefore blocks controller
calibration, GA/PSO comparisons on Cairo geometry and any real-world improvement claim.

## Provenance, licensing and ethics

- Count archive SHA-256: `d90703a022b2cad08e0533a4d9538236e45034c2fe260673a08c6995a56bbd7b`.
- Survey-location archive SHA-256: `d8b8e329be14b2c008460e7caed55c3dc820dc151119f4d0d37fa4a6070b42b1`.
- Local experiment associations are accepted only after source, configuration, split, artifact and
  code-provenance validation.
- TfC attribution is recorded. The audited portal metadata states no explicit record-level open
  license. Source and row-level derivatives remain Git-ignored unless documented publisher terms
  provide a legal basis for the intended redistribution.
- The release contains aggregate link counts, not identifiable trajectories. Results must not be
  used for automated enforcement, neighborhood ranking or controller deployment.
- EO/road descriptors are descriptive associations. No causal land-use effect is inferred.

The detailed data, split, model, uncertainty and reproduction record is in
[observed_traffic_case_study.md](observed_traffic_case_study.md); source feasibility and legal gaps
are in [real_data_feasibility.md](real_data_feasibility.md).

## Reproducibility

The README provides exact Windows PowerShell 5.1/Python 3.12 commands. Local artifacts are hash-bound
and strictly validated before dashboard display, PDF generation or MLflow logging. The dashboard
keeps observed EO, observed traffic, blocked measured-demand simulation and abstract simulation in
separate sections. Notebooks are output-free and reject missing or mismatched local evidence.
