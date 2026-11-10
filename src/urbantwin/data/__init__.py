"""Data ingestion and deterministic fixtures for UrbanTwin-AI.

The :mod:`urbantwin.data.earth_observation` module is the only code path that
creates a ``data_kind == "real"`` raster stack.  The demo generator is kept in
a separate module so callers cannot accidentally fall back to synthetic data
when a network request fails.
"""

from .demo import write_demo
from .earth_observation import WORLDCOVER_CLASSES, compute_indices, fetch_real, grid_for_aoi
from .observed_study import load_normalized_counts, prepare_observed_data
from .observed_traffic import audit_observed_traffic
from .spatial_features import landcover_percentages, road_network_stats, summarize

__all__ = [
    "WORLDCOVER_CLASSES",
    "audit_observed_traffic",
    "compute_indices",
    "fetch_real",
    "grid_for_aoi",
    "landcover_percentages",
    "load_normalized_counts",
    "prepare_observed_data",
    "road_network_stats",
    "summarize",
    "write_demo",
]
