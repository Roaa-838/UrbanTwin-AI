# UrbanTwin-AI

**Geospatial Intelligence & Traffic Optimization for Sustainable Cities**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB)](https://www.python.org/) [![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE) [![Evidence status](https://img.shields.io/badge/Research-evaluation%20required-orange)](#research-status-and-honesty)

> **Research question:** Can satellite-derived land-cover information and controlled AI-based traffic-signal optimization provide a reproducible framework for analyzing urban context and simulated mobility?

**Author:** Roaa Raafat · B.Sc. Data Science & Artificial Intelligence · Zewail City of Science and Technology

A modular research prototype integrating:

- **Real geospatial context** around Tahrir Square, Cairo: Sentinel-2 L2A, ESA WorldCover 2021, OpenStreetMap road network, georeferenced maps, NDVI, NDBI.
- **ML evaluation:** Random Forest vs Histogram Gradient Boosting, spectral feature ablation, geographically blocked held-out test patches, macro F1 and confusion matrix **against WorldCover pseudo-labels**.
- **Traffic experiments:** SUMO fixed-time control versus GA and PSO for two main traffic-signal green phases. Same traffic routes and held-out random seeds across controllers.
- **Visualization and research outputs:** Streamlit dashboard, plots, machine-readable provenance, research-report PDF generator.

**IMPORTANT:** The default SUMO intersection is **abstract and synthetic**, **not a calibrated simulation of Tahrir Square**, and its traffic demand is generated independently of satellite imagery. The geospatial and mobility findings are adjacent research components, not a validated causal relationship. A SUMOFlow external network adapter is included only for compatible two-green-phase signal programs and matching prebuilt vehicle-route files.

## Research status and honesty

| Component | Status in this initial repository | How to validate |
|---|---|---|
| Geospatial data ingestion | Implemented; requires internet to run | `urbantwin eo-fetch` |
| QGIS-compatible projected raster metrics | Implemented; real-data execution pending | `urbantwin geo-train` |
| OpenStreetMap road data | Implemented; network access required | `urbantwin roads` |
| Spatial-blocked WorldCover reference classification | Implemented and offline-tested on **synthetic fixtures** | Run on real Sentinel/WorldCover rasters |
| SUMO network and demand generator | Implemented; installed SUMO executable required | `urbantwin traffic-run --quick` |
| GA and PSO signal timing optimization | Implemented; tested on deterministic mathematical objective | Test actual simulation and then run full budget |
| Dashboard and report generator | Implemented; display outputs only if present | See commands below |
| Validated Cairo real-world traffic improvement | **NOT CLAIMED** | Needs calibrated roads, demand and traffic counts |

**There are no pre-filled experimental performance results** in the repository. Synthetic demo metrics are deliberately excluded from scholarship results. Do not cite the deterministic synthetic-demo scores as scientific findings. Real-time online data acquisition and SUMO execution could not be verified in the authoring environment; verify these steps locally before presenting them as completed.

## Architecture

```mermaid
flowchart TB
    subgraph Earth[REAL GEO / Earth Observation]
      S[Sentinel-2 L2A · STAC] --> A[SCL-screened aligned bands]
      W[ESA WorldCover 2021] --> C[Reference class raster]
      O[OpenStreetMap roads] --> R[Projected road-network metrics]
      A --> F[NDVI / NDBI / spectral features]
      C --> ML[Spatial blocked holdout: RF / HistGB]
      F --> ML
    end
    subgraph Sim[SIMULATED TRAFFIC · abstract cross-junction]
      N[Generated SUMO intersection + stochastic route demand] --> B[Fixed-time baseline]
      N --> GA[Genetic algorithm timing]
      N --> PSO[Particle swarm timing]
      B --> T[Time loss / queue / throughput / unfinished / optional CO2]
      GA --> T
      PSO --> T
    end
    R --> V[Separate panels: Urban Analytics Dashboard]
    ML --> V
    T --> V
    V --> P[Reproducibility metadata / Figures / PDF report]
```

## Quickstart (Windows PowerShell 5.1 / Python 3.12)

**1. Download this repository ZIP, extract to a local folder, and open PowerShell inside the extracted `UrbanTwin-AI` folder.** You do not need to clone or modify a private repository first.

```powershell
cd "D:\UrbanTwin-AI"   # CHANGE to wherever you extracted the ZIP
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[all]"
python -m pytest -q
```

If PowerShell blocks activation, use `.\.venv\Scripts\python.exe` in place of `python`, or run `Set-ExecutionPolicy -Scope Process Bypass` for your current session only. The package will install geospatial wheels, Streamlit, PySTAC, TraCI and testing tools; internet access is necessary for first installation.

**2. Run the OFFLINE unit tests, followed by a synthetic-data pipeline sanity check.**

```powershell
python -m pytest -q
urbantwin demo
urbantwin status
```

The `demo` command saves **only synthetic** data in `data/synthetic/` and synthetic scores in `results/demo/`. Its classification scores can be very high because the data were deliberately generated from simple patterns. Never use those scores in your academic CV.

**3. Download REAL satellite and reference-label data.** This may take time and requires an internet connection.

```powershell
urbantwin eo-fetch
urbantwin geo-train
```

Read `data/real/provenance.json` for the actual STAC item IDs, dates, coordinate system and acquisition metadata. The project takes a small AOI and caps pixels at 400,000 to avoid large accidental downloads.

**4. Retrieve actual OpenStreetMap road geometry.**

```powershell
urbantwin roads
```

This downloads the driving-road network for the bounding box (not a historical snapshot), exports `data/osm/roads.geojson`, and reports road density in projected meters per km². Be respectful of Overpass request rate limits. OSM road metadata and WorldCover are not necessarily from the same year.

**5. Install SUMO separately** (Python `traci` does **not** install the SUMO simulation executable).

Get Eclipse SUMO from https://eclipse.dev/sumo/ . Install Windows binaries, then make `sumo.exe` and `netconvert.exe` accessible on `PATH`, or set `SUMO_HOME` to the installation directory. Confirm:

```powershell
sumo --version
netconvert --version
```

**6. Run a very small real SUMO smoke test, then the full configured simulation.**

```powershell
urbantwin traffic-run --quick
# After the quick run executes correctly, run the configured multi-scenario benchmark:
urbantwin traffic-run
```

The quick mode runs small optimizer populations and records its results **separately** in `results/traffic_quick/`; it is not a meaningful optimization study. A full run can require dozens of SUMO simulations per scenario and substantial CPU time. You can adjust population, generations and evaluation seeds in `configs/experiments.yaml`. Keep all tested methods, routes and split seeds consistent.

**7. Display the dashboard and generate the report once real results exist.**

```powershell
streamlit run src/urbantwin/visualization/dashboard.py
urbantwin report
```

The PDF is saved as `reports/generated_research_report.pdf` and includes only the result CSVs currently available. If you want an **explicitly labeled methodology-only PDF** before running experiments:

```powershell
urbantwin report --draft
```

The repository includes an initial `reports/research_report.pdf` that clearly identifies itself as methodology-only. Replace it with a measured and verified report before citing any numeric research findings.

## Exact repository file layout

```text
UrbanTwin-AI/
├── .env.example
├── .gitignore
├── LICENSE
├── README.md
├── pyproject.toml
├── requirements.txt
├── Makefile
├── configs/
│   ├── study_area.yaml
│   └── experiments.yaml
├── src/urbantwin/
│   ├── cli.py
│   ├── common.py
│   ├── data/
│   │   ├── earth_observation.py
│   │   ├── spatial_features.py
│   │   └── demo.py
│   ├── geoai/
│   │   ├── train.py
│   │   └── evaluate.py
│   ├── traffic/
│   │   ├── network.py
│   │   ├── baselines.py
│   │   ├── optimize.py
│   │   └── benchmark.py
│   └── visualization/
│       ├── plots.py
│       ├── dashboard.py
│       └── report_writer.py
├── notebooks/
│   ├── 01_geospatial_analysis.ipynb
│   └── 02_experiment_results.ipynb
├── scripts/
│   └── verify_release.py
├── results/
│   ├── README.md
│   └── figures/.gitkeep
├── reports/
│   ├── research_report.md
│   └── research_report.pdf  # methodology only, no claimed results
└── tests/
```

## Data methods

### Coordinate reference system and spatial validity

The study area is defined as `[west, south, east, north]` in WGS84. Raster bands and WorldCover tiles are reprojected onto an identical 10 m raster grid in **EPSG:32636 (UTM 36N)** for the Cairo AOI. Cloud screening uses Sentinel-2 Scene Classification Layer (SCL) classes 4, 5, 6 and 7; selected scene-level cloud cover is an additional filter, not a guarantee of cloud-free study pixels. Pixel-wise median across selected scenes is computed only on nonmasked observations. Data source timestamps and item IDs are in `data/real/provenance.json`.

Indices use `(NIR − Red)/(NIR + Red)` for NDVI and `(SWIR − NIR)/(SWIR + NIR)` for NDBI; divide-by-zero outputs are NaN. Missing bands/cloudy pixels are excluded. Percentages are computed on the valid Sentinel-WorldCover overlap and the represented area is explicitly reported.

### WorldCover reference-label study, not ground-truth survey

The 2021 ESA WorldCover raster is the label source and therefore the evaluation target. Classifier macro-F1 measures **agreement with WorldCover**, not independent physical land-cover accuracy. Models are evaluated with nonoverlapping spatial blocks rather than independent random pixels. Because adjacent blocks are autocorrelated, the numbers should not be interpreted as cross-region generalization. Only classes with adequate training and held-out support are evaluated; support and confusion matrices are saved.

### Traffic baseline and fairness

The generated network is a one-junction, four-approach, one-lane-per-direction **synthetic** SUMO network. The default four route directions go straight across the intersection. GA and PSO each tune the durations (in seconds) of **two detected main green phases**, keeping SUMO yellow and other phases intact. A controller with a more complex traffic light requires a dedicated, verified phase adapter; unsupported programs raise an error.

Demand is independently generated for low/medium/high scenarios with Poisson sampled counts and randomized departure times. A **fixed random seed** fixes vehicle demand so the baseline, GA and PSO see the same scenarios. Optimizers train on separate seeds from the final comparison. The objective used for *tuning*, not scientific ranking, is:

```
mean completed-trip time loss (s) + 2 × mean approach queue (vehicles) + 2 × unfinished vehicles
```

This is a deliberately simple, scale-dependent heuristic; the report compares individual KPIs rather than claiming the scalar objective is universally meaningful. Report `timeLoss`, waiting time, mean controlled-approach halting count, throughput, unfinished vehicles and optimization runtime. Means from SUMO `tripinfo` include **completed trips only**, while unfinished vehicles are reported separately to expose censoring. If `track_emissions: true` is set, CO₂ is the integrated SUMO model estimate (not measured air pollution). The default disables emissions to reduce runtime.

### Integrating your existing SUMOFlow network (advanced)

Do not substitute your SUMOFlow network and immediately claim this is a Tahrir digital twin. First confirm network geometry, phases, demand, vehicle characteristics, and traffic counts. The existing `traffic-run` supports an external `.net.xml` path and matching route files generated for each scenario and seed:

```powershell
urbantwin traffic-run --network "D:\SUMOFlow\network.net.xml" `
  --routes-template "D:\SUMOFlow\routes\{scenario}_{seed}.rou.xml" `
  --tls-id "YOUR_TLS_ID"
```

It will intentionally reject signal programs with anything other than exactly two main green phases. A custom adapter is needed for more complex intersections. Calibrate against observed traffic counts before making any real-world interpretation.

## Application-specific portfolio framing

| Program | Emphasis |
|---|---|
| Erasmus Mundus SMACCs | Sustainable mobility, signal-optimization baselines, reproducibility |
| Copernicus Digital Earth | Sentinel-2 processing, WorldCover labels, NDVI/NDBI, cloud masks |
| GeoTech | GIS/CRS handling, georeferenced rasters, OSM network statistics |
| AISS | Responsible use of simulated outcomes for urban planning |
| Toulouse | Optimization experiment design and reliability |
| Lille Data Science | Spatial validation, ML ablation, class imbalance, error analysis |

**Safe CV bullet after running real-data experiments:**

> Developed a reproducible geospatial pipeline over a Cairo AOI using Sentinel-2 L2A, ESA WorldCover and OpenStreetMap; evaluated spatially blocked Random Forest and gradient-boosted classification and compared GA/PSO traffic-signal timing with a fixed-time SUMO baseline under controlled synthetic-demand scenarios.

Don't add numeric percentage improvements until the real simulations and statistical summaries have been generated and reviewed. Do not claim a *digital twin of Cairo* without traffic network geometry and demand calibration.

## Limitations, ethics, and data attribution

- WorldCover data are 2021 ESA labels; no 2020–2021 change detection is performed because the products use different algorithms.
- The Sentinel scenes are selected for dates within 2021 but do not necessarily match the WorldCover processing composite exactly.
- OSM data are contemporary when retrieved, not a historical 2021 street map.
- The road network and traffic simulation are not automatically linked or empirically synchronized.
- The current default signals model is an idealized straight-through intersection, not real Cairo road geometry.
- No individually identifiable GPS traces or patient data are used.
- **Data attribution:** Contains modified Copernicus Sentinel data; ESA WorldCover © ESA (CC BY 4.0); road data © OpenStreetMap contributors (ODbL). Source links: [ESA WorldCover](https://esa-worldcover.org/en/data-access), [Planetary Computer](https://planetarycomputer.microsoft.com/docs/), [OSM](https://www.openstreetmap.org/copyright), [SUMO](https://eclipse.dev/sumo/).

## License

This project's original source code is released under MIT (`LICENSE`). Source data are **not** relicensed by this code repository. Follow each dataset's own terms. Do not commit large downloaded data or unverified performance claims.
