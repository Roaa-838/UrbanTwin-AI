# UrbanTwin-AI reproducibility guide

This guide contains the operational detail intentionally removed from the project homepage. It covers environment setup, offline validation, real-data acquisition, observed-study execution, the separate abstract SUMO benchmark, checkpoint recovery, MLflow, notebooks, dashboards, reports, and troubleshooting.

## 1. Evidence boundaries

UrbanTwin-AI has four source labels that must not be mixed:

| Label | Meaning |
|---|---|
| `real` | Real Sentinel-2/WorldCover inputs and GeoAI outputs |
| `real_observed_traffic` / `real_observed_traffic_analysis` | TfC measured-link inputs and validated observed-study outputs |
| `simulation_abstract` | Synthetic SUMO network, demand, and simulator outputs |
| `synthetic` | Deterministic software fixtures only |

The measured-demand signal-control phase is not run. TfC link counts do not provide turning movements, complete routes, observed phase timing, or a calibrated intersection demand matrix. Never relabel the abstract SUMO benchmark as a Cairo traffic result.

The audited TfC portal records have no published record-level license identifier. Public download is not treated as an open license: obtain the files from the publisher, follow the applicable terms, and do not redistribute source or derived material without an explicit legal basis. Source roles and required acknowledgements are centralized in [DATA_SOURCES.md](DATA_SOURCES.md).

## 2. Prerequisites

- 64-bit Python 3.12.
- Git.
- Internet access for the initial dependency installation and real-data downloads.
- Eclipse SUMO only for traffic simulation; its Python packages do not install the simulator binary.
- Legitimate access to the TfC records for the observed workflow.

Binary wheels are strongly recommended for Rasterio, GeoPandas, PyProj, and Shapely. Building them from source requires a native geospatial toolchain that is outside this project.

### Windows PowerShell 5.1

```powershell
git clone https://github.com/Roaa-838/UrbanTwin-AI.git
Set-Location .\UrbanTwin-AI

python --version
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[all,mlflow]"
```

Activation is optional. Using `.\.venv\Scripts\python.exe` avoids PowerShell execution-policy and interpreter-resolution surprises.

### Linux or macOS

```bash
git clone https://github.com/Roaa-838/UrbanTwin-AI.git
cd UrbanTwin-AI

python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[all,mlflow]'
```

### Dependency extras

| Extra | Purpose |
|---|---|
| `eo` | STAC and Planetary Computer access |
| `roads` | OSMnx and GeoPandas road workflows |
| `observed` | Observed-traffic geospatial processing |
| `traffic` | TraCI and SUMO Python utilities |
| `app` | Streamlit and PyDeck dashboard |
| `mlflow` | Optional experiment tracking |
| `notebooks` | JupyterLab, nbclient, and nbformat |
| `dev` | Pytest and Ruff |
| `all` | Runtime, geospatial, traffic, app, and developer extras; MLflow and notebooks remain explicit |

Use `.[all,mlflow]` for the complete offline test suite. Add `notebooks` only when needed:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[all,mlflow,notebooks]"
```

## 3. Complete offline validation

Offline tests use deterministic fixtures and mocks. They do not retrieve TfC, STAC, Overpass, or SUMO network data.

```powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts\verify_release.py
.\.venv\Scripts\urbantwin.exe status
```

The release verifier covers repository structure, packaging, tracked-file hygiene, and static provenance boundaries. Its `not_verified` list must not be reported as passed work.

## 4. Deterministic synthetic smoke fixture

The demo is a software check, not research evidence:

```powershell
.\.venv\Scripts\urbantwin.exe demo
.\.venv\Scripts\urbantwin.exe status
```

It writes only to `data/synthetic/` and `results/demo/`. Its deliberately simple patterns can produce high scores; do not cite them as empirical results.

## 5. Real 2021 EO and GeoAI workflow

The configured AOI is the Tahrir vicinity in `configs/study_area.yaml`. Real mode retrieves Sentinel-2 L2A plus ESA WorldCover 2021, creates an aligned 10 m EPSG:32636 stack, and records STAC item IDs without retaining signed asset URLs.

```powershell
.\.venv\Scripts\urbantwin.exe eo-fetch
.\.venv\Scripts\urbantwin.exe geo-train
.\.venv\Scripts\urbantwin.exe status
```

Important processing rules:

- continuous reflectance bands use bilinear resampling;
- categorical WorldCover uses nearest-neighbour resampling;
- Sentinel SCL classes 4, 5, 6, and 7 are retained as clear surface;
- NDVI is `(NIR - Red) / (NIR + Red)`;
- NDBI is `(SWIR - NIR) / (SWIR + NIR)`;
- repeated group-disjoint splits are used instead of random pixel splitting; and
- class eligibility is determined from training support only.

The scores quantify agreement with WorldCover pseudo-labels, not independent land-cover accuracy.

## 6. Current OpenStreetMap road retrieval

The general road command downloads the current driving graph for the configured AOI and reports projected network statistics:

```powershell
.\.venv\Scripts\urbantwin.exe roads
```

This requires Overpass access and must respect provider rate limits. Current OSM geometry is not a historic 2021 or 2024 road reconstruction.

## 7. TfC source acquisition and validation

The authoritative portal records are:

- [Traffic Counts, record 190](https://data.transportforcairo.com/catalogue/#/document/190), asset 252.
- [Traffic Survey Locations, record 193](https://data.transportforcairo.com/catalogue/#/dataset/193), asset 255.

If direct download is unavailable, open those pages in a browser and save the official files as:

```text
data/observed/source/tfc_2024_traffic_counts.zip
data/observed/source/tfc_2024_traffic_survey_locations.zip
```

### Verified direct-download path in PowerShell

```powershell
$repoRoot = (Resolve-Path ".").Path
$sourceDir = Join-Path $repoRoot "data\observed\source"
New-Item -ItemType Directory -Force $sourceDir | Out-Null

$counts = Join-Path $sourceDir "tfc_2024_traffic_counts.zip"
$locations = Join-Path $sourceDir "tfc_2024_traffic_survey_locations.zip"

Invoke-WebRequest -UseBasicParsing `
  -Uri "https://data.transportforcairo.com/api/v2/assets/252/download" `
  -OutFile $counts
Invoke-WebRequest -UseBasicParsing `
  -Uri "https://data.transportforcairo.com/api/v2/assets/255/download" `
  -OutFile $locations

$retrievedAtUtc = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
(Get-FileHash -Algorithm SHA256 $counts).Hash.ToLower()
(Get-FileHash -Algorithm SHA256 $locations).Hash.ToLower()
```

Expected audited archive hashes:

```text
counts:    d90703a022b2cad08e0533a4d9538236e45034c2fe260673a08c6995a56bbd7b
locations: d8b8e329be14b2c008460e7caed55c3dc820dc151119f4d0d37fa4a6070b42b1
```

Do not continue if a newly retrieved archive does not match the expected source record and schema. Investigate a legitimate publisher update rather than bypassing validation.

### Audit and prepare

```powershell
.\.venv\Scripts\urbantwin.exe observed-data-audit `
  --counts $counts `
  --locations $locations

.\.venv\Scripts\urbantwin.exe observed-data-prepare `
  --counts $counts `
  --locations $locations `
  --source-retrieved-at-utc $retrievedAtUtc
```

Preparation writes `data/observed/tfc/normalized_v1/` atomically and refuses to overwrite an inspected bundle. It preserves missing values and undefined source semantics rather than inventing timezone, sample-validity, or travel-direction labels.

## 8. Observed-site road context

The bounded Overpass workflow creates a checksum-validated current OSM road bundle for the selected AOI:

```powershell
.\.venv\Scripts\urbantwin.exe observed-roads `
  --output-dir "data\observed\osm\current_tahrir_reproduction"
```

Use a new output directory for every reproduction. Existing evidence is never overwritten.

If Overpass is unavailable, an independently downloaded Geofabrik Egypt PBF can provide a geometry-only fallback:

```powershell
.\.venv\Scripts\urbantwin.exe observed-roads `
  --output-dir "data\observed\osm\geofabrik_geometry_reproduction" `
  --pbf "C:\path\to\egypt-latest.osm.pbf"
```

The fallback is explicitly not treated as a legal routable graph. The TfC direction codes still lack a semantic travel-direction legend.

## 9. Retrospective 2024 EO context and observed study

The 2024 EO bundle is separate from the 2021 GeoAI stack:

```powershell
.\.venv\Scripts\urbantwin.exe observed-eo-fetch
.\.venv\Scripts\urbantwin.exe observed-study --resume
```

The verified Sentinel acquisitions include 24 October, 29 October, and 3 November 2024. The 3 November scene postdates the selected 31 October survey and is retrospective context, not a prospectively available model feature. WorldCover 2021 remains a historic reference.

EO descriptors are not features in the multi-site traffic benchmark. The benchmark uses clock-time and static released road descriptors, with a training-only grouped development protocol and one frozen joint future-date/unseen-inferred-site holdout.

### Optional MLflow association

```powershell
.\.venv\Scripts\urbantwin.exe observed-study --resume --mlflow
```

Resume reuses a completed local study only after strict source, configuration, split, artifact, and code-compatibility checks. It does not rewrite historical MLflow runs.

### Reports

Complete local report, only when the retained TfC terms permit the intended use:

```powershell
.\.venv\Scripts\urbantwin.exe observed-report `
  --output "output\pdf\urbantwin_observed_traffic_study.pdf"
```

Methods-only report:

```powershell
.\.venv\Scripts\urbantwin.exe observed-report `
  --publication-safe `
  --output "output\pdf\urbantwin_observed_traffic_publication_safe.pdf"
```

Both reports receive provenance sidecars. Neither should be committed unless the applicable publication rights are confirmed.

## 10. Notebooks

Install the notebook extra and open only after the required local bundles exist:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[notebooks]"
.\.venv\Scripts\jupyter.exe lab
```

The tracked notebooks are output-free:

- `notebooks/01_geospatial_analysis.ipynb`
- `notebooks/02_experiment_results.ipynb`
- `notebooks/03_observed_traffic_measurements.ipynb`
- `notebooks/04_observed_traffic_benchmark.ipynb`

They must fail clearly when required real artifacts are missing or incompatible; they must not substitute synthetic observations.

## 11. Eclipse SUMO installation

Install the Eclipse SUMO binaries from <https://eclipse.dev/sumo/>. `traci` and `sumolib` alone are insufficient.

Verify both executables:

```powershell
sumo --version
netconvert --version
```

If they are not on `PATH`, set `SUMO_HOME` to the SUMO installation root and add its `bin` directory for the current shell.

## 12. Abstract SUMO benchmark

Run the small smoke configuration first:

```powershell
.\.venv\Scripts\urbantwin.exe traffic-run --quick --resume
```

Quick mode writes to `results/traffic_quick/` and is not a scientific optimization result.

Run the configured full abstract comparison only when the required compute time is acceptable:

```powershell
.\.venv\Scripts\urbantwin.exe traffic-run --resume
```

The checked-in design compares fixed time, GA, PSO, and equal-budget random search under separate training and evaluation seeds. It uses generated routes on a synthetic one-junction network. Do not run it merely to validate documentation changes.

## 13. Checkpoint identity, resume, and cleanup

Each successful unique SUMO execution is written atomically below `.urbantwin/checkpoints/traffic/`. Cache identity binds:

- configuration hash;
- network and route hashes;
- SUMO version;
- scenario and seed;
- signal plan;
- emission settings; and
- simulation settings.

`--resume` rejects stale, incomplete, corrupt, and incompatible entries. Ctrl+C preserves checkpoints that were already completed.

Override the location when needed:

```powershell
.\.venv\Scripts\urbantwin.exe traffic-run --resume `
  --checkpoint-dir "D:\UrbanTwin-cache\traffic"
```

To clear only the default traffic checkpoints, first stop every benchmark and use this guarded PowerShell command:

```powershell
$repoRoot = (Resolve-Path ".").Path
$checkpointRoot = [System.IO.Path]::GetFullPath(
  (Join-Path $repoRoot ".urbantwin\checkpoints")
)
$trafficCheckpoints = [System.IO.Path]::GetFullPath(
  (Join-Path $checkpointRoot "traffic")
)

if (-not $trafficCheckpoints.StartsWith(
    $checkpointRoot + [System.IO.Path]::DirectorySeparatorChar,
    [System.StringComparison]::OrdinalIgnoreCase
)) {
  throw "Refusing to remove a path outside the checkpoint root."
}

if (Test-Path -LiteralPath $trafficCheckpoints) {
  Remove-Item -LiteralPath $trafficCheckpoints -Recurse -Force
}
```

This does not delete final result CSVs or MLflow data. The next run recomputes cleared simulations.

## 14. MLflow tracking and UI

Checkpointing does not require MLflow. Enable tracking explicitly:

```powershell
.\.venv\Scripts\urbantwin.exe traffic-run --mlflow --resume
```

The default store is `.urbantwin/mlflow/mlflow.db`, with artifacts under `.urbantwin/mlflow/artifacts/`. Both are Git-ignored. The authoritative recovery mechanism remains the local checkpoint store.

Use a configurable tracking URI:

```powershell
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
.\.venv\Scripts\urbantwin.exe traffic-run --mlflow --resume
```

Open the default local UI:

```powershell
$repoRoot = (Resolve-Path ".").Path
$dbPath = (Join-Path $repoRoot ".urbantwin\mlflow\mlflow.db").Replace('\', '/')
$artifactPath = (Join-Path $repoRoot ".urbantwin\mlflow\artifacts").Replace('\', '/')

.\.venv\Scripts\mlflow.exe ui `
  --backend-store-uri "sqlite:///$dbPath" `
  --default-artifact-root "file:///$artifactPath" `
  --host 127.0.0.1 `
  --port 5000
```

Visit <http://127.0.0.1:5000>. Tracking errors must be reported, but they do not invalidate already verified local computation.

## 15. Dashboard and reports

```powershell
.\.venv\Scripts\streamlit.exe run src\urbantwin\visualization\dashboard.py
.\.venv\Scripts\urbantwin.exe report
```

The dashboard separates observed EO, observed traffic, the blocked measured-demand simulation phase, and abstract simulation. It validates local artifacts before display.

Generate a methodology-only report without consuming experiment results:

```powershell
.\.venv\Scripts\urbantwin.exe report --draft
```

## 16. Output locations

| Path | Contents | Git status |
|---|---|---|
| `data/real/` | Real 2021 EO stack and provenance | Ignored |
| `data/observed/` | TfC sources/normalized data, 2024 EO, current OSM | Ignored |
| `data/synthetic/` | Deterministic software fixtures | Ignored |
| `results/geoai/` | Real WorldCover-agreement results | Ignored |
| `results/observed_traffic/` | Validated measured-study artifacts | Ignored |
| `results/traffic/` | Full abstract SUMO results | Ignored |
| `results/traffic_quick/` | Quick simulator smoke outputs | Ignored |
| `.urbantwin/checkpoints/` | Authoritative simulation checkpoints | Ignored |
| `.urbantwin/mlflow/` | Local SQLite tracking and artifacts | Ignored |
| `output/pdf/` | Generated observed-study PDFs and sidecars | Ignored |

Do not commit raw data, non-public records, local databases, signed STAC URLs, caches, or large rasters.

## 17. External SUMO networks

The external adapter requires an existing `.net.xml`, an explicit route template containing both `{scenario}` and `{seed}`, and a verified traffic-light ID:

```powershell
.\.venv\Scripts\urbantwin.exe traffic-run `
  --network "D:\path\to\network.net.xml" `
  --routes-template "D:\path\to\routes\{scenario}_{seed}.rou.xml" `
  --tls-id "TLS_ID" `
  --resume
```

The current adapter intentionally supports exactly two detected main green phases while preserving every non-green phase from the network. It rejects incompatible programs rather than forcing the abstract controller onto real geometry. Supplying an external network does not establish calibration.

## 18. Troubleshooting

| Symptom | Action |
|---|---|
| `ModuleNotFoundError: geopandas` | Install `.[all,mlflow]`, not only `.[dev]`. |
| Rasterio/GeoPandas tries to compile | Use 64-bit Python 3.12 with current binary wheels. |
| `sumo` or `netconvert` not found | Install Eclipse SUMO binaries and check `PATH`/`SUMO_HOME`. |
| STAC or Overpass unavailable | Retry respectfully; real mode stops and never substitutes synthetic data. |
| Observed data missing | Acquire TfC records 190 and 193, then run `observed-data-audit` and `observed-data-prepare`. |
| Existing output directory rejected | Use a new directory; evidence bundles are intentionally not overwritten. |
| Observed results rejected as stale | Reproduce on the current code/configuration or inspect the provenance mismatch; never edit hashes by hand. |
| Checkpoint rejected | Let `--resume` recompute it; do not copy checkpoints between experiments. |
| MLflow logging fails | Preserve local validated results, fix tracking separately, and do not claim the run was logged. |

For source licensing and acknowledgements, see [DATA_SOURCES.md](DATA_SOURCES.md). For the scientific design and limitations, see [observed_traffic_case_study.md](observed_traffic_case_study.md), [research_case_study.md](research_case_study.md), and [real_data_feasibility.md](real_data_feasibility.md).
