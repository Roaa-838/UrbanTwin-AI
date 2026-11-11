# Data sources, licensing, and attribution

This document records the origin, role, time basis, and reuse boundary of each external source used by UrbanTwin-AI. Source categories remain distinct in code, artifacts, figures, and reports.

## Source register

| Source | Research role | Time and geometry basis | License or reuse boundary |
|---|---|---|---|
| [Transport for Cairo traffic counts, record 190](https://data.transportforcairo.com/catalogue/#/document/190) | Directional 20-minute measured vehicle counts by class | Observations collected in 2024 | The audited portal metadata did not state an explicit record-level license. Public download must not be interpreted as an open license. Raw files and derived measured-traffic material are not redistributed by this repository. |
| [Transport for Cairo survey locations, record 193](https://data.transportforcairo.com/catalogue/#/dataset/193) | Directional survey-segment geometry and road descriptors | Released geometry is derived from a July 2026 map extract and is retrospective for the 2024 observations | Same record-level licensing boundary as the count data; attribute Transport for Cairo and confirm applicable terms before any redistribution. |
| Copernicus Sentinel-2 Level-2A | Surface reflectance, cloud screening, spectral indices, and retrospective spatial context | 2021 scenes for the land-cover study; 2024 scenes for a separate retrospective context bundle | Contains modified Copernicus Sentinel data 2021 and 2024. Follow the [Copernicus data terms](https://dataspace.copernicus.eu/terms-and-conditions). |
| [ESA WorldCover 2021](https://esa-worldcover.org/en/data-access) | Historic categorical reference and GeoAI pseudo-labels | 2021 reference product; never treated as 2024 field truth | CC BY 4.0. Credit the ESA WorldCover project and the consortium attribution below. |
| [OpenStreetMap](https://www.openstreetmap.org/copyright) | Current road-network context and conservative spatial matching | Current third-party mapping; not a contemporaneous 2024 traffic observation | © OpenStreetMap contributors, ODbL 1.0. Database-derived redistribution must follow ODbL requirements. |
| [Geofabrik Egypt extract](https://download.geofabrik.de/africa/egypt.html) | Optional offline geometry source when live road retrieval is unavailable | Current extract at retrieval time | OpenStreetMap data distributed under ODbL 1.0; retain OpenStreetMap attribution. |
| [Eclipse SUMO](https://eclipse.dev/sumo/) | Microscopic counterfactual traffic simulation | Project-generated synthetic network, routes, demand, and modeled emissions | Simulator output is not measured traffic or environmental data. Tool licensing is documented by the upstream project. |

## Evidence categories

| Category | Examples | Allowed interpretation |
|---|---|---|
| Measured observations | Directional vehicle counts and source survey fields | Observed link-flow evidence for the recorded intervals; not complete turning demand or signal operation. |
| Remotely sensed observations | Sentinel-2 surface reflectance after documented masking and compositing | Spatial context for the acquisition dates; later scenes are retrospective only. |
| Historic pseudo-labels | WorldCover 2021 | A reproducible categorical reference for agreement studies, not independent field ground truth. |
| Third-party map data | Road geometry and attributes | Mapped context at the source revision; not traffic measurement or a historic road reconstruction. |
| Inferred parameters | Analytical site clusters, conservative spatial matches, and model features | Reproducible analytical constructs with explicit uncertainty; not publisher-supplied facts. |
| Simulated variables | Routes, turning proportions, controller plans, queues, delays, and modeled emissions | Counterfactual simulator quantities only; never relabeled as observed Cairo outcomes. |

## Redistribution and repository policy

- The repository does not track raw measured-traffic files, normalized row-level observations, local source archives, or figures derived from those observations.
- The traffic portal records had no explicit license identifier at audit time. Written clarification from the publisher is required before redistributing source data or derived material beyond the applicable terms.
- Compact Earth-observation, map-context, and abstract-simulation images are included only with source attribution and a checksum manifest.
- Large rasters, signed asset URLs, local experiment databases, caches, checkpoints, and credentials remain excluded from version control.
- The MIT license applies to project source code, not to third-party data products.

## Required acknowledgements

- **Measured traffic:** Transport for Cairo, 2024 traffic counts and traffic survey locations; use and redistribution remain subject to the publisher's applicable terms.
- **Copernicus:** Contains modified Copernicus Sentinel data 2021 and 2024.
- **WorldCover:** © ESA WorldCover project 2021 / Contains modified Copernicus Sentinel data (2021) processed by the ESA WorldCover consortium. WorldCover 2021 is provided under CC BY 4.0.
- **Road data:** © OpenStreetMap contributors, available under the Open Database License (ODbL) 1.0.

For acquisition, validation, and reproduction commands, see [REPRODUCIBILITY.md](REPRODUCIBILITY.md). For the source-access audit and schema findings, see [real_data_feasibility.md](real_data_feasibility.md).
