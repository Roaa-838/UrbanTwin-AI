# README research-image manifest

These compact images make every homepage reference reproducible from a clean checkout. They contain no measured-traffic rows, flow plots, survey geometry, selected-segment overlay, or observed-traffic model result. Licensing and attribution are recorded in [DATA_SOURCES.md](../../DATA_SOURCES.md).

| Asset | Evidence and rendering | Source validation | Asset SHA-256 | Publication basis |
|---|---|---|---|---|
| `sentinel2_2021_rgb.png` | Natural-colour display of the real 2021 Sentinel-2 L2A median composite; per-channel 2nd–98th percentile stretch, gamma 0.9, invalid pixels white | `data/real/eo_stack.npz`, SHA-256 `c8ff16e6b5dee42af669ee3dde6efdf91437505b36fb33fd304781e0db606fb4`; provenance schema `urbantwin.eo-provenance/v1` | `973a919d737d88e83f3bbc3a91f8fa7a1f7f1fb8b1cd66c877166bc1be7ee38d` | Contains modified Copernicus Sentinel data 2021 |
| `road_network_context.png` | Current road-network lines in EPSG:32636; no survey segment or match overlay | `edges_projected.geojson`, SHA-256 `b9d85cad6453dc6541c98ca507ed42917f0d0fd1b06a71874539d6c572edc0ee`; bundle label `third_party_osm_current_road_context` | `deddb2871bb898776d2ced87d5f136e90d189a12ec4307db68daabd57b7a01cd` | Road data © OpenStreetMap contributors, ODbL 1.0 |
| `osm_tahrir_road_context.png` | Source-attributed rendering of the same current road-network bundle; retained for technical documentation | Same validated `edges_projected.geojson` and bundle label as above | `36c628066fa2223c7bcce4b87f0f5b9244137b6c0916fc71a115a28ae2a3073b` | Road data © OpenStreetMap contributors, ODbL 1.0 |
| `geoai_worldcover_agreement.png` | Exact copy of the validated repeated grouped GeoAI model-comparison output | `results/geoai/figures/geoai_model_comparison.png`; GeoAI inputs bind the same EO-stack hash above and `data_kind: real` | `c26967c78d11e6e57aef1026501d61ad53bca1e9281d327b7d1c049c48673aa8` | Copernicus Sentinel data plus ESA WorldCover 2021 under CC BY 4.0; scores are pseudo-label agreement |
| `controlled_simulation_time_loss.png` | Neutral-label rendering of completed-trip time-loss means from the validated full benchmark | `results/traffic/traffic_summary.csv`, SHA-256 `0eacd814e8844db94a3aa93227a3832bf6793f570b1d064f4653349d381dd31d`; provenance label `simulation_abstract` | `744f3fbd67945d3b2f8ba5652e30964b4c3773c7e56afc4bfffdb36b37aa41fb` | Project-generated abstract simulation output |
| `sumo_abstract_time_loss.png` | Exact source-attributed rendering from the validated abstract benchmark | `results/traffic/figures/traffic_time_loss.png`; provenance label `simulation_abstract` | `f37b2db3b78315c716be4734f88fc66c43ca96286218ed42798e3e187fbbeb74` | Project-generated abstract simulation output |

## Deliberately excluded

Files under `results/observed_traffic/figures/` remain local because the audited portal metadata does not state an explicit record-level license for redistribution of source or derived measured-traffic material. Public download is not interpreted as an open license.

The Earth-observation and road maps above were rendered only from independently reusable source layers. They contain no measured survey lines. The GeoAI image is copied byte-for-byte from a validated local result, while the controlled-simulation image is re-rendered from its validated aggregate CSV with neutral public labels.
