# Measured Greater Cairo traffic: GIS, spatial features and retrospective Earth observation

**Research prototype status:** the observed-traffic analysis and a retrospective-context 2024 Sentinel-2 acquisition are complete locally. WorldCover 2021 remains a historic reference product. A measured-demand signal-control simulation was not run because the released evidence does not identify turning movements, signal phases, controller timing, or a complete route-demand matrix.

## Abstract

This study tests how far a public Greater Cairo traffic-count release can support a reproducible urban-data workflow without overstating what was measured. Transport for Cairo (TfC) published 6,618 directional 20-minute records covering 142 directional segments on 30 survey dates between 21 July and 5 November 2024. The records contain counts for 12 vehicle classes and sum to 2,961,483 observed vehicles. TfC also released 145 directional survey-line geometries derived from a July 2026 OpenStreetMap extract.

The implementation preserves the source fields and missing values, validates segment and direction joins, projects the released lines to EPSG:32636, and separates ambiguous OSM identifiers from reliable segment joins. A reproducible 250 m geometry-distance rule groups the 145 lines into 60 analytical site clusters, but these are explicitly **inferred groups**, not publisher location identifiers. A separately retrieved current OpenStreetMap drive graph supplies third-party road context for the Tahrir AOI; it does not supply 2024 observations, surveyed travel-direction semantics, or signal-control evidence. For directional volume estimation, the last six survey dates form a frozen joint future-date/unseen-inferred-site holdout. Models are developed using five-fold grouped validation on earlier inferred sites. Training-only GroupKFold development ranked Random Forest with combined features first (mean CV MAE 283.42). Histogram gradient boosting with combined features had the lowest observed final-holdout measurement-free MAE point estimate (348.25, 95% site-bootstrap interval 239.46-487.46; RMSE 496.34, interval 309.46-654.29 vehicles per 20 minutes). This holdout comparison is descriptive, not retrospective model selection. One-step persistence was much more accurate on exactly adjacent intervals (MAE 63.93), but it uses a previous measured count and is therefore a different, online information budget rather than a fair competitor to measurement-free estimation.

Three real Sentinel-2 Level-2A scenes acquired on 24 October, 29 October and 3 November 2024 provide descriptive retrospective context for the selected 6th of October Bridge links. The 3 November scene postdates the 31 October traffic survey and was not available to a prospective forecast. ESA WorldCover 2021 remains a historic reference product rather than 2024 ground truth. EO descriptors are not used as causal evidence or as multi-site prediction features; no EO prediction gain was evaluated. Because the public release lacks turning movements and observed signal operation, the project stops before claiming a calibrated Cairo traffic simulation or controller benefit.

## Research questions

1. Can the published TfC files be transformed into a lossless, provenance-bound table of directional 20-minute counts and mapped survey segments?
2. How well can directional 20-minute total volume be estimated at future-date, unseen inferred sites from clock-time and static released road descriptors?
3. How does measurement-free estimation compare descriptively with an online persistence forecast that has access to the immediately preceding observed interval?
4. What retrospective spectral context and historic land-cover reference are visible around a predeclared observed link near the existing Tahrir AOI?
5. Do the available observations support a measured-demand signal-control SUMO study?

The answer to question 5 is currently **no**. This is a scientific gate, not a missing software feature.

## Data and provenance

### Measured traffic

The primary measured inputs are the official TfC records established in the feasibility audit:

- [Traffic Counts, record 190](https://data.transportforcairo.com/catalogue/#/document/190), asset 252.
- [Traffic Survey Locations, record 193](https://data.transportforcairo.com/catalogue/#/dataset/193), asset 255.

The validated count archive has SHA-256 `d90703a022b2cad08e0533a4d9538236e45034c2fe260673a08c6995a56bbd7b`; the location archive has SHA-256 `d8b8e329be14b2c008460e7caed55c3dc820dc151119f4d0d37fa4a6070b42b1`. The normalized bundle contains 6,618 wide interval rows and 79,416 long class-count rows. Its `data_kind` is `real_observed_traffic`.

TfC names itself as the attribution party, and the portal record had no explicit license identifier or reuse terms at audit time. Public access is not treated as an open-source or open-data license. Source files, row-level normalized data, derived measured maps and generated reports therefore remain local and Git-ignored unless documented publisher terms provide a legal basis for the intended redistribution.

The source does not provide a timezone, an interval-level sample-validity flag, or a legend mapping direction codes `0` and `1` to inbound/outbound. The pipeline records these facts as `unspecified_by_publisher`, `not_provided_by_source`, and undefined direction semantics instead of inventing values.

### Geometry and third-party map context

The location file contains 145 WGS84 directional lines and reports OSM extract dates of 15-20 July 2026. Those geometries are later than the 2024 traffic observations. They are treated as TfC-released, OSM-derived spatial context, not as contemporaneous surveyed road geometry. Every counted segment joins to a released geometry after whitespace normalization, and all joined direction codes agree. Five segment joins, affecting 156 count rows, have conflicting count-versus-geometry OSM IDs; those rows are preserved for characterization but excluded from the primary spatial benchmark. Three geometry segments have no count rows.

A current OSM drive graph was retrieved through Overpass for the Tahrir AOI and exported in EPSG:32636 with OpenStreetMap attribution and ODbL 1.0 licensing. The validated bundle contains 849 nodes and 1,788 directed edges in three weakly connected components; the largest contains 847 nodes (99.76%). Both predeclared 6th of October Bridge survey lines pass the conservative OSM-ID-plus-projected-geometry match rule, while the other 143 released survey lines lie outside this deliberately small AOI. The bundle is useful for road context and conservative spatial alignment, but it is a **current** third-party map rather than a 2024 traffic observation.

TfC direction codes `0` and `1` have no published travel-direction legend, so neither matched line is assigned to a legal OSM routing direction. The road bundle also contains no observed turning movements, routes, signal phases, timing, or controller operation. It is therefore not a study-ready signal network and is not evidence of a calibrated Cairo simulation.

### Earth observation

The observed-study context bundle contains real Sentinel-2 L2A surface reflectance from 24 October, 29 October and 3 November 2024 around the selected 31 October traffic survey. The 3 November acquisition is post-survey retrospective context, not a feature that could have been available to a prospective 31 October forecast. The three STAC item IDs and scene-level cloud metadata are retained in `data/observed/eo_2024/provenance.json`. Assets are SCL-masked and aligned to the same 10 m EPSG:32636 Tahrir grid used by the project. ESA WorldCover 2021 remains a historic categorical reference/pseudo-label product, not 2024 ground truth.

Only the selected Tahrir-area link pair has this consistently processed EO context. EO is therefore excluded from the 60-site prediction benchmark rather than being filled or extrapolated to unsampled sites.

## Preprocessing and validation

The ingestion stage is local-only and fail-closed. It validates the exact published schemas, archive membership, UUID segment keys, 20-minute intervals, non-negative integer class counts, weekday/date agreement, WGS84 line geometry, duplicate keys, and source hashes. It writes a new directory atomically and refuses to overwrite an existing inspected bundle.

The normalized wide table retains source row numbers, source text, original and normalized segment/direction values, all 12 class counts, the derived class sum, missing road names, and explicit unknown metadata. A tidy long table retains one row per source interval and vehicle class. A separate join diagnostic keeps all OSM-ID conflicts visible. No missing observation is imputed.

Released lines are projected to EPSG:32636 for length and distance calculations. The spatial features used by the benchmark are:

- line length;
- centroid easting and northing;
- released highway class and one-way value, with missing values remaining an explicit category;
- direction code as an uninterpreted categorical value; and
- sine/cosine clock-time plus day of week for temporal features.

IDs, road names, exact dates, current-period counts, opposite-direction counts, and future observations are excluded from model features.

## Site grouping and predeclared site selection

The release states that observations came from 60 locations but contains no official physical `location_id`. To prevent paired or nearby lines leaking across spatial folds, the analysis connects released lines whose minimum EPSG:32636 geometry distance is at most 250 m and uses the connected components as analytical groups. This deterministic rule yields 60 groups: 46 with two lines, three with three, and 11 with four. The result is stable from 250 m through 750 m, but it remains an inference that TfC has not confirmed.

The 6th of October Bridge pair was selected before model evaluation because it intersects the existing Tahrir AOI and has full-day, same-day coverage in both direction codes on 31 October 2024. Each direction has 72 intervals. Their measured totals are 107,028 and 69,206 vehicles. The site is suitable for directional link-flow and retrospective EO context. It is a motorway bridge, not a candidate signalized junction.

## Prediction task and evaluation design

The target is the sum of the 12 published vehicle-class counts for one directional 20-minute interval. Two information budgets are kept separate:

1. **Measurement-free estimation:** uses clock-time and/or static released spatial descriptors, without a current or previous count.
2. **Online one-step forecasting:** persistence uses only the immediately preceding measured interval, and only when it is exactly 20 minutes earlier. Gaps are never bridged or filled.

Every directional segment appears on exactly one survey date. Pure same-site temporal generalization is therefore impossible. The last 20% of the 30 distinct dates were frozen as the final holdout: 27, 28, 29 and 31 October plus 4 and 5 November 2024. After excluding the five ambiguous OSM joins, training contains 5,255 rows from 50 inferred sites and the final holdout contains 1,207 rows from ten different inferred sites. Site overlap is zero. The test is thus a strict **joint future-date and unseen-inferred-site** evaluation; date and site effects cannot be disentangled.

Model development uses five-fold `GroupKFold` on the 50 training inferred sites. The frozen holdout is evaluated once. Uncertainty intervals use 500 bootstrap resamples of inferred sites, not individual rows, so correlated intervals are not treated as independent observations.

### Models and ablations

- Historical mean by day-of-week and 20-minute clock slot, with training-only fallbacks.
- L2-regularized Poisson regression as the simple count-model reference.
- Random Forest regression.
- Histogram gradient boosting with Poisson loss.
- Persistence for the separate online task.

The fitted measurement-free models use `time_only`, `spatial_only`, and `combined` feature sets. Because consistent EO features are unavailable for all sites, the completed ablation does not claim an EO contribution.

## Results

All errors below are vehicles per directional 20-minute interval. Intervals are 95% site-bootstrap intervals. These are point estimates with broad and overlapping uncertainty; no statistical-significance claim is made.

| Information budget | Model | Features | Rows | MAE (95% interval) | RMSE (95% interval) | Mean Poisson deviance |
|---|---|---:|---:|---:|---:|---:|
| Measurement-free | Historical mean | time | 1,207 | 468.13 (314.07-678.99) | 631.34 (401.82-830.47) | 781.75 |
| Measurement-free | Poisson | time | 1,207 | 472.09 (320.15-632.33) | 623.17 (387.71-827.02) | 728.62 |
| Measurement-free | Poisson | spatial | 1,207 | 541.83 (426.19-657.91) | 645.78 (494.39-779.51) | 761.05 |
| Measurement-free | Poisson | combined | 1,207 | 522.51 (372.26-684.32) | 662.21 (464.34-834.65) | 799.78 |
| Measurement-free | Random Forest | time | 1,207 | 470.05 (300.64-687.25) | 633.31 (377.78-817.64) | 795.07 |
| Measurement-free | Random Forest | spatial | 1,207 | 373.75 (260.92-492.79) | 507.59 (338.15-674.49) | 433.13 |
| Measurement-free | Random Forest | combined | 1,207 | 366.41 (273.05-479.15) | 489.38 (327.67-611.34) | 406.38 |
| Measurement-free | Histogram gradient boosting | time | 1,207 | 470.47 (291.69-673.36) | 633.38 (380.04-843.01) | 795.10 |
| Measurement-free | Histogram gradient boosting | spatial | 1,207 | 397.30 (264.13-547.85) | 542.91 (339.84-713.39) | 490.48 |
| Measurement-free | Histogram gradient boosting | combined | 1,207 | 348.25 (239.46-487.46) | 496.34 (309.46-654.29) | 385.66 |
| Prior measurement available | Persistence | previous interval | 1,170 | 63.93 (30.89-110.54) | 144.04 (68.24-200.70) | 17.32 |

Training-only GroupKFold development ranked Random Forest with combined features first (mean CV MAE 283.42), which is the defensible training-based selection result. Histogram gradient boosting with combined features has the lowest observed measurement-free MAE point estimate in the final holdout. Its interval overlaps those of other measurement-free models, so that post-evaluation ordering is descriptive and is neither retrospective model selection nor proof of superiority. The time-only models are close to the historical mean. Spatial and combined tree models have lower holdout point estimates than their time-only versions, but this ablation does not isolate a causal spatial contribution: the 2026 static road descriptors may encode site composition, and date/site are confounded in the joint holdout. EO is not part of any benchmark feature set.

Persistence is substantially more accurate on the 1,170 exactly adjacent test intervals. It is not a measurement-free baseline: it consumes a previous real observation at the same directional segment. It is reported separately to establish the difficulty and value of online measurement access, not as a direct win over the other estimators.

The complete local bundle retains per-fold validation metrics, per-site errors, row-level predictions, mean bias, median absolute error, and vehicle-class composition. Cars account for 59.49% of the observed vehicle total, followed by mini/microbuses at 11.25%; these are descriptive shares of the released observations, not estimates of the Greater Cairo fleet.

## Retrospective EO context at the selected link

For 50 m, 100 m and 250 m buffers around the selected released bridge lines, valid-pixel coverage in the 2024 Sentinel composite was 100.00% for all three buffers. Mean NDVI was -0.106, -0.139 and -0.036; mean NDBI was 0.256, 0.249 and 0.157. WorldCover 2021 built-up fractions were 40.00%, 33.49% and 53.80%, while represented historic vegetated-class fractions were 0.00%, 0.00% and 0.47%.

These numbers describe a raster neighborhood around an elevated urban road and must be interpreted cautiously. Buffers include surfaces below and beside the bridge, water and neighboring urban fabric. They do not measure traffic, road-level exposure, or causal land-use effects. Only the Sentinel spectral observations are near the 2024 survey date; the categorical reference remains from 2021.

## Why measured-demand SUMO was not run

The TfC records provide directional link totals. They do not provide intersection turning movements, lane-level counts, routes, signal IDs, phase states, timing, offsets, pedestrian operation, queues, speeds, or travel times. One link count cannot identify a complete turning-demand matrix. The predeclared observed link is also a motorway bridge rather than a verified controlled intersection.

The current OSM bundle establishes that genuine road geometry can be retrieved and conservatively aligned for the selected AOI; it does not close these demand and control gaps. In particular, the undefined TfC direction codes cannot be converted into legal route directions without unsupported assumptions.

Accordingly, this study does not construct an OSM-network signal scenario, infer a controller, force the existing two-phase synthetic controller onto Cairo geometry, run GA/PSO/random search against the measured counts, or claim calibration. The repository's existing 198-request SUMO benchmark remains a separate **abstract synthetic simulation** and is not relabeled as Cairo evidence.

## Limitations and technical risks

- The portal provides no explicit record-level open license; public access alone does not authorize redistribution of source or derived material.
- The published “60 locations” cannot be reconstructed from an official key; the 60 analytical groups are proximity-derived.
- All directional segments were observed on only one date, confounding date and site in the final split.
- Survey schedules are heterogeneous, and the source does not provide sample-validity flags or timezone.
- Five OSM identifiers disagree between count and geometry records.
- The released OSM-derived geometry is from 2026, two years after the counts.
- The validated current OSM graph covers only the Tahrir AOI. Its two in-AOI survey lines match spatially, but routing direction remains unverified and no signal inventory, turning-flow validation, or 2024 network reconstruction is available.
- Retrospective 2024 Sentinel context covers only the selected Tahrir-area link pair; WorldCover is from 2021, and neither can support a 60-site EO ablation.
- Bootstrap intervals are based on only ten inferred holdout sites and are correspondingly broad.
- The observed holdout is not a representative probability sample of all Greater Cairo roads or dates.

## Ethical and scientific considerations

The release contains road-link aggregates rather than individual trajectories, but licensing, attribution, and faithful source interpretation still matter. Results should not be used to rank neighborhoods, infer behavior of individuals, automate enforcement, or deploy a traffic controller. Model errors may be largest at high-flow or unusual road types, and a citywide claim would exceed the sample design. No model output is presented as causal, calibrated control evidence, or observed emissions. Simulator-modeled CO2 in the separate abstract SUMO study remains simulation-only.

## Reproducibility

The local normalized and benchmark bundles bind every artifact to SHA-256 hashes, source record IDs, the configuration hash, split hashes, code revision and source labels. The authoritative configuration is `configs/observed_traffic.yaml`; normalized files are kept under ignored `data/observed/`, and results under ignored `results/observed_traffic/`. Any redistribution requires a documented legal basis under the applicable publisher terms.

The executed local bundle records its configuration, split, EO stack, current-OSM manifest, source
archives, code revision and every result artifact by SHA-256. Experiment-tracking metadata is
accepted only when it matches the same source, configuration, split, artifact and code provenance.
Historical records remain immutable.

From Windows PowerShell 5.1 with Python 3.12 and legitimate local downloads:

```powershell
# Run from the UrbanTwin-AI repository root.
.\.venv\Scripts\Activate.ps1

urbantwin observed-data-audit `
  --counts "C:\path\to\2024_traffic_counts.zip" `
  --locations "C:\path\to\2024_traffic_survey_locations.zip"

$retrievedAt = (Get-Date).ToUniversalTime().ToString("o")
urbantwin observed-data-prepare `
  --counts "C:\path\to\2024_traffic_counts.zip" `
  --locations "C:\path\to\2024_traffic_survey_locations.zip" `
  --source-retrieved-at-utc $retrievedAt

urbantwin observed-eo-fetch
urbantwin observed-roads `
  --output-dir "data\observed\osm\current_tahrir_reproduction"
urbantwin observed-study --resume --mlflow
urbantwin observed-report
```

Use new output-directory names for repeated runs; the preparation and benchmark stages refuse to overwrite an inspected bundle. The manual download pages and verified schemas are documented in [real_data_feasibility.md](real_data_feasibility.md).

## Attribution

- Transport for Cairo (TfC): 2024 traffic counts and traffic survey locations. The audited portal metadata stated no explicit record-level open license, and no unrecorded redistribution right is claimed.
- Contains modified Copernicus Sentinel data 2024.
- © ESA WorldCover project 2021 / Contains modified Copernicus Sentinel data (2021) processed by ESA WorldCover consortium. WorldCover 2021 is provided under CC BY 4.0.
- Road data © [OpenStreetMap contributors](https://www.openstreetmap.org/copyright), available under ODbL 1.0.
