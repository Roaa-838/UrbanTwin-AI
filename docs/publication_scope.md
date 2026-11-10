# Publication scope for the observed Greater Cairo study

## Current permission status

The project author reports that Transport for Cairo (TfC) directly authorized continued academic
UrbanTwin-AI research and data reuse. The original message, sender details and exact terms are kept
in a private permission record outside Git. The portal metadata itself still supplies no explicit
license identifier.

This repository therefore does **not** claim that the TfC records are open-source or openly licensed,
and it does not infer public redistribution rights beyond the exact private permission wording.
Transport for Cairo must be attributed for the 2024 traffic counts and traffic survey locations.

## Public-safe repository inventory

The tracked repository contains ingestion, validation, feature, benchmark, reporting and dashboard
code; deterministic synthetic test fixtures; schema descriptions; source URLs; source hashes; and
bounded scientific descriptions. Output-free notebooks contain no embedded source rows or rendered
results.

The following stay outside Git by default:

- original TfC downloads and correspondence;
- normalized row-level traffic tables and mapped survey geometries;
- local observed-study figures, PDFs and complete research bundles;
- Sentinel rasters, OSM extracts and caches;
- MLflow SQLite databases and artifact stores; and
- SUMO checkpoints, secrets and signed asset URLs.

Tracked text may describe aggregate measurements and evaluated model metrics. Publishing those
descriptions is appropriate only if the exact TfC permission covers public derived results.

## Audit of the already-public feature branch

The tracked tree and the branch diff from `main` were inspected. No TfC source archive, normalized
row table, measured GeoJSON/GeoPackage, Sentinel raster, OSM extract, observed-results directory,
measured research PDF, local MLflow database, checkpoint, credential or signed STAC URL is tracked.
The only tracked PDF predates the observed workflow and is not an observed-TfC result artifact.

The public branch does, however, contain derived aggregate counts, model-error summaries,
uncertainty intervals, source checksums and bounded scientific interpretations in Markdown and
output-free notebooks. These are derived publications even though they do not expose source rows.
Because the exact private permission wording was not available for this repository audit, their
continued public distribution cannot be certified from the portal terms alone. Written
confirmation should expressly cover aggregate metrics, figures, PDFs, portfolio summaries and
publication in a public Git repository. A later redaction commit would not remove material from
existing Git history; if TfC does not authorize a category that is already public, the repository
owner should agree an appropriate remediation with TfC before merging or publishing further
artifacts.

## Decision checklist before publication

Complete the private permission record, then mark each intended release as `allowed`, `not allowed`
or `not stated` from TfC's exact wording:

- original source archives or samples;
- normalized rows or a derived database;
- survey geometry, GeoJSON or GeoPackage outputs;
- aggregate counts and vehicle-class tables;
- model metrics, uncertainty intervals and error breakdowns;
- maps, charts and other derived figures;
- the research PDF and dashboard screenshots;
- output-bearing notebooks;
- portfolio/application summaries; and
- publication in a public Git repository.

If any intended use is not stated, keep it private and request written clarification. Do not rewrite
permission wording as a broader grant, attach private correspondence, or treat anonymous download as
a license.

## Third-party attribution retained independently

- Contains modified Copernicus Sentinel data 2024.
- © ESA WorldCover project 2021 / Contains modified Copernicus Sentinel data (2021) processed by ESA
  WorldCover consortium. WorldCover 2021 is provided under CC BY 4.0.
- Road data © OpenStreetMap contributors, available under ODbL 1.0.
