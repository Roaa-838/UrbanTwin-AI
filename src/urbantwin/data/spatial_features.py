"""Spatial summaries for aligned rasters and OpenStreetMap roads."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

import numpy as np

from urbantwin.data.earth_observation import WORLDCOVER_CLASSES, compute_indices


def landcover_percentages(
    labels: np.ndarray, valid_mask: np.ndarray | None = None
) -> dict[str, float]:
    """Return class shares over supported, non-zero WorldCover pixels.

    Unknown codes and WorldCover no-data code 0 are excluded from the
    denominator.  ``valid_mask`` can further restrict the denominator to the
    valid Sentinel/WorldCover overlap.
    """

    values = np.asarray(labels)
    if values.ndim != 2:
        raise ValueError("labels must be a two-dimensional raster")
    selected = np.isin(values, sorted(WORLDCOVER_CLASSES))
    if valid_mask is not None:
        mask = np.asarray(valid_mask, dtype=bool)
        if mask.shape != values.shape:
            raise ValueError("valid_mask must have the same shape as labels")
        selected &= mask
    count = int(np.count_nonzero(selected))
    if count == 0:
        return {}
    result: dict[str, float] = {}
    for class_id, class_name in WORLDCOVER_CLASSES.items():
        class_count = int(np.count_nonzero(selected & (values == class_id)))
        if class_count:
            result[class_name] = 100.0 * class_count / count
    return result


def _finite_summary(values: np.ndarray) -> dict[str, float | None]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return {"mean": None, "median": None, "std": None}
    return {
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "std": float(np.std(finite)),
    }


def summarize(bands: np.ndarray, labels: np.ndarray, resolution_m: float) -> dict[str, Any]:
    """Summarize only the finite Sentinel/WorldCover overlap."""

    spectral = np.asarray(bands, dtype=np.float32)
    classes = np.asarray(labels)
    if spectral.ndim != 3 or spectral.shape[0] != 5:
        raise ValueError("bands must have shape (5, height, width)")
    if classes.shape != spectral.shape[1:]:
        raise ValueError("labels must align exactly with the band height and width")
    try:
        pixel_size = float(resolution_m)
    except (TypeError, ValueError) as exc:
        raise ValueError("resolution_m must be a positive number") from exc
    if not math.isfinite(pixel_size) or pixel_size <= 0:
        raise ValueError("resolution_m must be a positive number")

    valid = np.all(np.isfinite(spectral), axis=0) & np.isin(
        classes, sorted(WORLDCOVER_CLASSES)
    )
    valid_count = int(np.count_nonzero(valid))
    if valid_count == 0:
        raise ValueError("No finite Sentinel pixels overlap supported WorldCover labels")
    ndvi, ndbi = compute_indices(spectral)
    total = int(classes.size)
    area_km2 = valid_count * pixel_size * pixel_size / 1_000_000
    class_counts = {
        WORLDCOVER_CLASSES[class_id]: int(np.count_nonzero(valid & (classes == class_id)))
        for class_id in WORLDCOVER_CLASSES
        if np.any(valid & (classes == class_id))
    }
    return {
        "summary_scope": "finite Sentinel-2 / supported WorldCover overlap",
        "grid_pixel_count": total,
        "valid_overlap_pixels": valid_count,
        "valid_overlap_percent": 100.0 * valid_count / total,
        "represented_area_km2": float(area_km2),
        "resolution_m": pixel_size,
        "landcover_percentages": landcover_percentages(classes, valid),
        "landcover_pixel_counts": class_counts,
        "ndvi": _finite_summary(ndvi[valid]),
        "ndbi": _finite_summary(ndbi[valid]),
    }


def _validated_bbox(config: dict[str, Any]) -> tuple[float, float, float, float]:
    bbox = config.get("bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise ValueError("bbox must be [west, south, east, north] in EPSG:4326")
    try:
        west, south, east, north = (float(value) for value in bbox)
    except (TypeError, ValueError) as exc:
        raise ValueError("bbox coordinates must be finite numbers") from exc
    if not all(math.isfinite(value) for value in (west, south, east, north)):
        raise ValueError("bbox coordinates must be finite numbers")
    if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
        raise ValueError("bbox must be ordered WGS84 [west, south, east, north]")
    return west, south, east, north


def road_network_stats(config: dict[str, Any]):
    """Download real OSM drive roads and return projected length statistics.

    The graph is converted to an undirected street graph before length is
    summed, preventing ordinary two-way streets from being counted once per
    travel direction.  Geometries are clipped to the configured AOI in its
    projected metre CRS, then returned as WGS84 for GeoJSON/dashboard use.
    """

    if not isinstance(config, dict):
        raise TypeError("Configuration must be a mapping")
    if config.get("mode", "real") != "real":
        raise ValueError("road_network_stats downloads real OSM data and requires mode: real")
    west, south, east, north = _validated_bbox(config)
    try:
        import geopandas as gpd
        import osmnx as ox
        from pyproj import CRS
        from shapely.geometry import box
    except ImportError as exc:  # pragma: no cover - installation profile dependent
        raise RuntimeError(
            "OSM road retrieval requires the roads dependencies. Install with "
            "`python -m pip install -e \".[roads]\"`."
        ) from exc

    try:
        target_crs = CRS.from_user_input(config["crs"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("A valid projected metre CRS is required for road metrics") from exc
    if not target_crs.is_projected:
        raise ValueError("Road length and density require a projected metre CRS")
    if not target_crs.axis_info or not math.isclose(
        float(target_crs.axis_info[0].unit_conversion_factor), 1.0, rel_tol=0, abs_tol=1e-9
    ):
        raise ValueError("Road length and density require a projected metre CRS")

    bbox = (west, south, east, north)
    try:
        graph = ox.graph.graph_from_bbox(
            bbox, network_type="drive", simplify=True, retain_all=True, truncate_by_edge=False
        )
    except Exception as exc:
        raise RuntimeError(f"OpenStreetMap/Overpass road download failed for bbox {bbox}: {exc}") from exc
    if graph.number_of_nodes() == 0 or graph.number_of_edges() == 0:
        raise RuntimeError("OpenStreetMap returned an empty drive network for the configured AOI")

    projected = ox.projection.project_graph(graph, to_crs=target_crs)
    undirected = ox.convert.to_undirected(projected)
    edges = ox.convert.graph_to_gdfs(
        undirected, nodes=False, edges=True, fill_edge_geometry=True
    )
    aoi = gpd.GeoSeries([box(west, south, east, north)], crs="EPSG:4326").to_crs(
        target_crs
    ).iloc[0]
    clipped = edges.geometry.intersection(aoi)
    keep = ~clipped.is_empty & clipped.notna()
    edges = edges.loc[keep].copy()
    edges.geometry = clipped.loc[keep]
    total_length_m = float(edges.geometry.length.sum())
    area_km2 = float(aoi.area / 1_000_000)
    if area_km2 <= 0:
        raise RuntimeError("Projected AOI area is zero")
    output_edges = edges.to_crs("EPSG:4326")
    stats = {
        "schema_version": "urbantwin.osm-road-summary/v1",
        "data_kind": "real",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source": "OpenStreetMap current Overpass snapshot",
        "source_label": "real OpenStreetMap road geometry",
        "attribution": "© OpenStreetMap contributors, ODbL",
        "network_type": "drive",
        "bbox_wgs84": list(bbox),
        "analysis_crs": target_crs.to_string(),
        "returned_geometry_crs": "EPSG:4326",
        "graph_nodes": int(graph.number_of_nodes()),
        "directed_graph_edges": int(graph.number_of_edges()),
        "undirected_clipped_street_segments": len(output_edges),
        "aoi_area_km2": area_km2,
        "street_length_km": total_length_m / 1000,
        "street_length_density_km_per_km2": (total_length_m / 1000) / area_km2,
        "metric_note": "Undirected street geometries clipped to the configured projected AOI",
    }
    return stats, output_edges


__all__ = ["landcover_percentages", "road_network_stats", "summarize"]
