# UrbanTwin-AI — Research report (methodology-only draft)

**Status:** Initial implementation. No experimental findings have been verified for the real Sentinel-2/WorldCover data or in SUMO. Do not describe this document as a finished empirical study.

## Research question

Can satellite-derived land-cover context and simulated optimization of urban traffic signals be studied together in a reproducible framework without confusing environmental description with traffic-demand prediction?

## Research objectives

1. Acquire and spatially align a small Cairo area-of-interest from Sentinel-2 2021 and ESA WorldCover 2021.
2. Calculate NDVI/NDBI, raster class proportions, and optionally contemporary OSM road density.
3. Compare Random Forest with histogram gradient boosting on WorldCover pseudo-label agreement under repeated grouped spatial evaluation and band-index ablation, without using test labels to select eligible classes.
4. Compare a fixed-time controller with GA, PSO, and equal-budget random-search signal timing on an *abstract* synthetic intersection under low/medium/high simulated demand and paired held-out seeds.
5. Document accuracy, reliability, uncertainty, failure modes, and resource constraints separately for both subsystems.

## Method summary

See `configs/`, `src/urbantwin/`, the README, and the generated STAC provenance for exact parameters. Generated empirical reports require matching SHA-256 provenance for source data, configurations, and result tables. The map and simulation are juxtaposed for a multidisciplinary urban-analytics study, not a calibrated traffic digital twin.

## Results

**Not yet generated or reviewed.** This section should be updated from actual `results/geoai` and `results/traffic` files after running locally.

## Limitations

- The ML study measures agreement with WorldCover reference labels, not independent ground truth.
- The single AOI and spatially adjacent blocks limit external validity.
- The abstract SUMO network has no validated relationship to actual Tahrir intersections, real traffic counts, or observed emissions.
- Performance differences require quantitative evaluation across scenarios and paired seeds, and interpretation of generated/inserted/completed/unfinished/pending counts alongside completed-trip averages.
- Any SUMO CO₂ value is simulator-estimated under an explicit emissions-class assumption; it is not observed Cairo air pollution.

## Sources

- ESA WorldCover: https://esa-worldcover.org/en/data-access
- Microsoft Planetary Computer STAC: https://planetarycomputer.microsoft.com/docs/
- OpenStreetMap attribution: https://www.openstreetmap.org/copyright
- SUMO documentation: https://eclipse.dev/sumo/docs/
