"""Spatial and historic Earth-observation context for measured TfC link counts.

The public release does not contain an official physical-site identifier.  The
``inferred_site_id`` produced here is therefore an analytical grouping only:
survey lines whose projected geometries are within a declared distance are
connected, and connected components become groups.  It must never be presented
as a publisher location ID.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from urbantwin.common import require_real_provenance, sha256_file
from urbantwin.data.earth_observation import compute_indices
from urbantwin.data.observed_study import load_normalized_counts
from urbantwin.data.observed_traffic import ObservedTrafficValidationError


def _manifest(study_dir: Path) -> dict[str, Any]:
    path = study_dir / "source_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Observed-study manifest is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ObservedTrafficValidationError(f"Invalid observed-study manifest: {path}") from exc
    if not isinstance(value, dict) or value.get("data_kind") != "real_observed_traffic":
        raise ObservedTrafficValidationError("Observed-study manifest is not real traffic data")
    return value


def _verified_artifact(study_dir: Path, name: str) -> Path:
    manifest = _manifest(study_dir)
    record = (manifest.get("artifacts") or {}).get(name)
    if not isinstance(record, dict):
        raise ObservedTrafficValidationError(f"Manifest lacks observed artifact {name!r}")
    filename = record.get("filename")
    expected = record.get("sha256")
    if not isinstance(filename, str) or not filename:
        raise ObservedTrafficValidationError(f"Observed artifact {name!r} has no filename")
    path = (study_dir / filename).resolve()
    if path.parent != study_dir.resolve():
        raise ObservedTrafficValidationError(f"Observed artifact {name!r} escapes its bundle")
    if not path.is_file():
        raise FileNotFoundError(f"Observed artifact is missing: {path}")
    if not isinstance(expected, str) or sha256_file(path) != expected:
        raise ObservedTrafficValidationError(f"Observed artifact hash mismatch: {path.name}")
    return path


def _component_labels(geometries: list[Any], segment_ids: list[str], distance_m: float) -> list[str]:
    if not math.isfinite(distance_m) or distance_m <= 0:
        raise ValueError("inferred site-link distance must be positive and finite")
    count = len(geometries)
    parents = list(range(count))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parents[max(left_root, right_root)] = min(left_root, right_root)

    for left in range(count):
        for right in range(left + 1, count):
            if geometries[left].distance(geometries[right]) <= distance_m:
                union(left, right)

    components: dict[int, list[int]] = {}
    for index in range(count):
        components.setdefault(find(index), []).append(index)
    ordered = sorted(
        components.values(),
        key=lambda members: min(segment_ids[index] for index in members),
    )
    labels: dict[int, str] = {}
    for number, members in enumerate(ordered, start=1):
        for index in members:
            labels[index] = f"inferred_site_{number:03d}"
    return [labels[index] for index in range(count)]


def spatial_segment_features(
    study_dir: str | Path,
    *,
    projected_crs: str = "EPSG:32636",
    inferred_site_link_distance_m: float = 250.0,
):
    """Return verified segment geometry with projected, non-causal descriptors."""
    try:
        import geopandas as gpd
    except ImportError as exc:  # pragma: no cover - optional-dependency message
        raise RuntimeError(
            "Observed spatial features require GeoPandas; install `.[roads]`."
        ) from exc

    path = _verified_artifact(Path(study_dir), "survey_segments")
    segments = gpd.read_file(path)
    if segments.empty or segments.crs is None:
        raise ObservedTrafficValidationError("Survey geometry is empty or has no CRS")
    if not segments.crs.is_geographic:
        raise ObservedTrafficValidationError(
            f"Survey source geometry must be geographic WGS84, found {segments.crs}"
        )
    key = "segment_id_normalized"
    if key not in segments.columns or segments[key].astype(str).duplicated().any():
        raise ObservedTrafficValidationError("Survey geometry lacks unique normalized segments")

    projected = segments.to_crs(projected_crs).reset_index(drop=True)
    projected["segment_id"] = projected[key].astype(str)
    projected["geometry_osm_id"] = projected["osm_id"].astype(str)
    projected["geometry_road_name"] = projected["road_name"]
    projected["geometry_direction_code"] = projected[
        "direction_code_normalized"
    ].astype(str)
    projected["geometry_length_m"] = projected.geometry.length.astype(float)
    centroids = projected.geometry.centroid
    projected["centroid_easting_m"] = centroids.x.astype(float)
    projected["centroid_northing_m"] = centroids.y.astype(float)
    lon_lat = gpd.GeoSeries(centroids, crs=projected_crs).to_crs("EPSG:4326")
    projected["centroid_longitude"] = lon_lat.x.astype(float)
    projected["centroid_latitude"] = lon_lat.y.astype(float)
    projected["inferred_site_id"] = _component_labels(
        projected.geometry.tolist(),
        projected["segment_id"].tolist(),
        inferred_site_link_distance_m,
    )
    projected["spatial_match_status"] = np.where(
        projected["count_join_status"].astype(str).eq("matched"),
        "eligible_segment_join",
        "excluded_ambiguous_or_unobserved",
    )
    projected["spatial_source_label"] = "TfC survey line; OSM-derived geometry (2026 extract)"
    keep = [
        "segment_id",
        "geometry_osm_id",
        "geometry_direction_code",
        "geometry_road_name",
        "highway",
        "oneway",
        "survey_line_name",
        "osm_extract_date",
        "count_join_status",
        "spatial_match_status",
        "inferred_site_id",
        "geometry_length_m",
        "centroid_easting_m",
        "centroid_northing_m",
        "centroid_longitude",
        "centroid_latitude",
        "spatial_source_label",
        "geometry",
    ]
    return projected[keep]


def build_analysis_table(
    study_dir: str | Path,
    *,
    projected_crs: str = "EPSG:32636",
    inferred_site_link_distance_m: float = 250.0,
) -> pd.DataFrame:
    """Join normalized observations to unambiguous released survey geometry."""
    counts = load_normalized_counts(study_dir)
    segments = spatial_segment_features(
        study_dir,
        projected_crs=projected_crs,
        inferred_site_link_distance_m=inferred_site_link_distance_m,
    )
    attributes = pd.DataFrame(segments.drop(columns="geometry"))
    frame = counts.merge(attributes, on="segment_id", how="left", validate="many_to_one")
    if frame["geometry_osm_id"].isna().any():
        raise ObservedTrafficValidationError("A measured segment has no survey geometry")
    direction_disagreement = frame["direction_code"].astype(str).ne(
        frame["geometry_direction_code"].astype(str)
    )
    if direction_disagreement.any():
        raise ObservedTrafficValidationError("Count and geometry direction codes disagree")

    frame["observed_date"] = pd.to_datetime(frame["observed_date"], format="%Y-%m-%d")
    start = frame["interval_start_local"].str.split(":", expand=True).astype(int)
    frame["minute_of_day"] = start[0] * 60 + start[1]
    angle = 2 * np.pi * frame["minute_of_day"] / (24 * 60)
    frame["time_sin"] = np.sin(angle)
    frame["time_cos"] = np.cos(angle)
    frame["day_of_week"] = frame["observed_date"].dt.dayofweek.astype(int)
    frame["is_weekend"] = frame["day_of_week"].isin([4, 5]).astype(int)
    frame["eligible_for_spatial_benchmark"] = frame["spatial_match_status"].eq(
        "eligible_segment_join"
    )
    return frame


def historic_eo_context(
    study_dir: str | Path,
    eo_dir: str | Path,
    segment_ids: list[str],
    *,
    buffer_distances_m: tuple[int, ...] = (50, 100, 250),
) -> list[dict[str, Any]]:
    """Summarize real EO within fixed buffers around selected observed links.

    Acquisition dates are reported from provenance. WorldCover remains the
    historic 2021 reference product even when Sentinel-2 imagery is from 2024.
    """
    from rasterio.features import geometry_mask
    from rasterio.transform import Affine

    eo_dir = Path(eo_dir)
    stack_path = eo_dir / "eo_stack.npz"
    provenance_path = eo_dir / "provenance.json"
    if not stack_path.is_file() or not provenance_path.is_file():
        raise FileNotFoundError(f"Complete EO bundle is missing from {eo_dir}")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    require_real_provenance(provenance)
    if provenance.get("eo_stack_sha256") != sha256_file(stack_path):
        raise ValueError("EO stack hash does not match provenance")
    with np.load(stack_path, allow_pickle=False) as archive:
        bands = archive["bands"].copy()
        labels = archive["labels"].copy()
        valid = archive["valid_mask"].astype(bool)
    ndvi, ndbi = compute_indices(bands)
    segments = spatial_segment_features(study_dir, projected_crs=provenance["crs"])
    selected = segments.loc[segments["segment_id"].isin(segment_ids)]
    if len(selected) != len(set(segment_ids)):
        raise ValueError("One or more selected observed segment IDs are missing")
    selected_geometry = selected.geometry.union_all()
    transform = Affine(*provenance["transform"])
    results: list[dict[str, Any]] = []
    for distance in buffer_distances_m:
        if distance <= 0:
            raise ValueError("EO buffer distances must be positive")
        region = ~geometry_mask(
            [selected_geometry.buffer(float(distance)).__geo_interface__],
            out_shape=labels.shape,
            transform=transform,
            invert=False,
        )
        region_pixels = int(region.sum())
        usable = region & valid & np.isfinite(ndvi) & np.isfinite(ndbi)
        usable_pixels = int(usable.sum())
        if region_pixels == 0:
            raise ValueError(f"The {distance} m selected-link buffer misses the EO raster")
        represented_labels = labels[usable]
        results.append(
            {
                "buffer_m": int(distance),
                "raster_pixels": region_pixels,
                "valid_pixels": usable_pixels,
                "valid_pixel_coverage": usable_pixels / region_pixels,
                "mean_ndvi": float(np.nanmean(ndvi[usable])) if usable_pixels else None,
                "mean_ndbi": float(np.nanmean(ndbi[usable])) if usable_pixels else None,
                "worldcover_built_up_fraction": (
                    float(np.mean(represented_labels == 50)) if usable_pixels else None
                ),
                "worldcover_vegetated_fraction": (
                    float(np.mean(np.isin(represented_labels, [10, 20, 30, 40])))
                    if usable_pixels
                    else None
                ),
                "eo_data_kind": "real_remote_sensing_with_historic_reference",
                "sentinel_item_ids": provenance["sentinel_stac_item_ids"],
                "sentinel_period": provenance["configuration"].get("sentinel_period"),
                "worldcover_year": provenance["configuration"].get("worldcover_year"),
                "interpretation": (
                    "Descriptive spatial context only. Sentinel acquisition dates are "
                    "recorded; WorldCover 2021 is a historic reference product, not 2024 "
                    "land-cover ground truth or a causal traffic feature."
                ),
            }
        )
    return results


__all__ = ["build_analysis_table", "historic_eo_context", "spatial_segment_features"]
