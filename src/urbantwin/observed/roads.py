"""Current OSM road context for the measured-traffic study.

An Overpass/OSMnx result is a directed, routable *current-map* graph.  The
optional Geofabrik PBF fallback deliberately exports geometry only because a
GDAL line layer is not a topologically noded routing graph.  Neither source is
treated as observed 2024 signal timing or a calibrated traffic network.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
import time
from collections.abc import Callable, Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from urbantwin.common import sha256_file
from urbantwin.observed.features import spatial_segment_features

SCHEMA_VERSION = "urbantwin.observed-road-bundle/v1"
OSM_ATTRIBUTION = "© OpenStreetMap contributors, ODbL 1.0"


class ObservedRoadError(RuntimeError):
    """Raised when current-road retrieval or validation cannot be defended."""


def _dependencies():
    try:
        import geopandas as gpd
        import networkx as nx
        import osmnx as ox
        from pyproj import CRS
        from shapely.geometry import Point, box
        from shapely.ops import unary_union
    except ImportError as exc:  # pragma: no cover - installation profile dependent
        raise ObservedRoadError(
            "Observed road retrieval requires the roads extra: "
            "`python -m pip install -e \".[roads]\"`."
        ) from exc
    return gpd, nx, ox, CRS, Point, box, unary_union


def _bbox_and_crs(config: Mapping[str, Any]) -> tuple[tuple[float, float, float, float], str]:
    selection = config.get("site_selection")
    spatial = config.get("spatial")
    if not isinstance(selection, Mapping) or not isinstance(spatial, Mapping):
        raise TypeError("Observed config requires site_selection and spatial mappings")
    bbox = selection.get("aoi_wgs84")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise ValueError("site_selection.aoi_wgs84 must be [west, south, east, north]")
    try:
        west, south, east, north = (float(value) for value in bbox)
    except (TypeError, ValueError) as exc:
        raise ValueError("Observed-road AOI coordinates must be numeric") from exc
    if not all(math.isfinite(value) for value in (west, south, east, north)):
        raise ValueError("Observed-road AOI coordinates must be finite")
    if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
        raise ValueError("Observed-road AOI must be ordered WGS84 bounds")
    projected_crs = str(spatial.get("projected_crs", ""))
    _, _, _, crs_type, _, _, _ = _dependencies()
    try:
        crs = crs_type.from_user_input(projected_crs)
    except (TypeError, ValueError) as exc:
        raise ValueError("spatial.projected_crs is invalid") from exc
    if not crs.is_projected or not crs.axis_info:
        raise ValueError("Observed roads require a projected metric CRS")
    if not math.isclose(
        float(crs.axis_info[0].unit_conversion_factor), 1.0, rel_tol=0, abs_tol=1e-9
    ):
        raise ValueError("Observed roads require projected metre units")
    return (west, south, east, north), crs.to_string()


def _json_text(value: Any) -> str:
    if isinstance(value, (list, tuple, set)):
        return json.dumps(list(value), ensure_ascii=False, separators=(",", ":"), default=str)
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value)


def _osm_ids(value: Any) -> set[str]:
    if isinstance(value, str) and value.strip().startswith("["):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            decoded = value
        if isinstance(decoded, list):
            value = decoded
    if isinstance(value, (list, tuple, set)):
        values: Iterable[Any] = value
    else:
        values = (value,)
    result = set()
    for item in values:
        text = str(item).strip()
        if text and text.casefold() not in {"none", "nan"}:
            result.add(text)
    return result


def _write_json(value: Any, path: Path) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _fetch_overpass_graph(
    bbox: tuple[float, float, float, float],
    *,
    cache_folder: Path,
    retry_attempts: int,
    retry_delays_s: tuple[float, ...],
    sleep: Callable[[float], None] = time.sleep,
):
    """Retrieve a current drive graph with bounded backoff and OSMnx caching."""
    if retry_attempts < 1:
        raise ValueError("retry_attempts must be at least one")
    _, _, ox, _, _, _, _ = _dependencies()
    cache_folder.mkdir(parents=True, exist_ok=True)
    ox.settings.use_cache = True
    ox.settings.cache_folder = str(cache_folder)
    ox.settings.requests_timeout = max(int(ox.settings.requests_timeout), 60)
    failures: list[str] = []
    for attempt in range(retry_attempts):
        try:
            graph = ox.graph.graph_from_bbox(
                bbox,
                network_type="drive",
                simplify=True,
                retain_all=True,
                truncate_by_edge=False,
            )
            if graph.number_of_nodes() == 0 or graph.number_of_edges() == 0:
                raise ObservedRoadError("Overpass returned an empty drive graph")
            return graph, attempt + 1
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:  # noqa: BLE001 - provider exceptions vary by OSMnx version
            failures.append(f"attempt {attempt + 1}: {type(exc).__name__}: {exc}")
            if attempt + 1 < retry_attempts:
                delay = retry_delays_s[min(attempt, len(retry_delays_s) - 1)]
                sleep(max(0.0, float(delay)))
    raise ObservedRoadError(
        "OpenStreetMap/Overpass retrieval failed after bounded retries: " + " | ".join(failures)
    )


def _component_stats(graph: Any, nx: Any) -> dict[str, Any]:
    if not graph.is_directed():
        raise ObservedRoadError("OSM drive graph is not directed")
    components = sorted(
        (len(component) for component in nx.weakly_connected_components(graph)), reverse=True
    )
    if not components:
        raise ObservedRoadError("OSM drive graph has no weakly connected component")
    return {
        "weakly_connected_components": len(components),
        "largest_component_nodes": int(components[0]),
        "largest_component_node_fraction": float(components[0] / graph.number_of_nodes()),
        "component_node_counts": [int(value) for value in components],
    }


def _safe_edges(edges: Any) -> Any:
    frame = edges.reset_index()
    if "geometry" not in frame:
        raise ObservedRoadError("OSM graph edges have no geometry")
    frame["length_m_projected"] = frame.geometry.length.astype(float)
    if (
        frame["length_m_projected"].isna().any()
        or (~frame["length_m_projected"].map(math.isfinite)).any()
        or (frame["length_m_projected"] <= 0).any()
    ):
        raise ObservedRoadError("OSM graph contains non-positive projected edge lengths")
    for column in ("osmid", "name", "highway", "oneway", "maxspeed", "ref", "lanes"):
        if column not in frame:
            frame[column] = ""
        frame[column] = frame[column].map(_json_text)
    return frame[
        [
            "u",
            "v",
            "key",
            "osmid",
            "name",
            "highway",
            "oneway",
            "maxspeed",
            "ref",
            "lanes",
            "length_m_projected",
            "geometry",
        ]
    ]


def _safe_nodes(nodes: Any) -> Any:
    frame = nodes.reset_index()
    identifier = "osmid" if "osmid" in frame else frame.columns[0]
    frame = frame.rename(columns={identifier: "node_osm_id"})
    frame["node_osm_id"] = frame["node_osm_id"].map(_json_text)
    if "street_count" not in frame:
        frame["street_count"] = pd.NA
    return frame[["node_osm_id", "street_count", "geometry"]]


def _map_match_segments(
    survey_segments: Any,
    graph_edges: Any,
    *,
    maximum_distance_m: float,
    minimum_coverage: float,
    analysis_aoi: Any | None = None,
) -> pd.DataFrame:
    """Associate released survey lines to current OSM way IDs without inferring turns."""
    _, _, _, _, _, _, unary_union = _dependencies()
    if maximum_distance_m <= 0 or not 0 < minimum_coverage <= 1:
        raise ValueError("Map-match distance and coverage thresholds are invalid")
    edges = graph_edges.copy()
    edges["_osm_ids"] = edges["osmid"].map(_osm_ids)
    rows: list[dict[str, Any]] = []
    for survey in survey_segments.itertuples(index=False):
        requested_id = str(survey.geometry_osm_id)
        if analysis_aoi is not None and not survey.geometry.intersects(analysis_aoi):
            rows.append(
                {
                    "segment_id": str(survey.segment_id),
                    "geometry_osm_id": requested_id,
                    "inferred_site_id": str(survey.inferred_site_id),
                    "count_join_status": str(survey.count_join_status),
                    "candidate_directed_edges": 0,
                    "minimum_distance_m": math.nan,
                    "survey_coverage_fraction_15m": 0.0,
                    "spatial_match_status": "outside_selected_study_aoi",
                    "spatial_match_reliable": False,
                    "routing_direction_verified": False,
                    "edge_references": "",
                }
            )
            continue
        candidates = edges.loc[
            edges["_osm_ids"].map(lambda values, target=requested_id: target in values)
        ]
        if candidates.empty:
            nearest_distance = float(edges.geometry.distance(survey.geometry).min())
            rows.append(
                {
                    "segment_id": str(survey.segment_id),
                    "geometry_osm_id": requested_id,
                    "inferred_site_id": str(survey.inferred_site_id),
                    "count_join_status": str(survey.count_join_status),
                    "candidate_directed_edges": 0,
                    "minimum_distance_m": nearest_distance,
                    "survey_coverage_fraction_15m": 0.0,
                    "spatial_match_status": "unmatched_current_osm_id",
                    "spatial_match_reliable": False,
                    "routing_direction_verified": False,
                    "edge_references": "",
                }
            )
            continue
        union = unary_union(candidates.geometry.tolist())
        minimum_distance = float(survey.geometry.distance(union))
        coverage = float(
            survey.geometry.intersection(union.buffer(15.0)).length
            / max(float(survey.geometry.length), 1e-9)
        )
        geometry_supported = minimum_distance <= maximum_distance_m and coverage >= minimum_coverage
        source_join_ok = str(survey.count_join_status) == "matched"
        if source_join_ok and geometry_supported:
            status = "matched_osm_id_geometry_supported"
        elif not source_join_ok:
            status = "excluded_source_osm_id_conflict_or_unobserved"
        else:
            status = "ambiguous_or_weak_geometry_support"
        references = [
            f"{row.u}:{row.v}:{row.key}"
            for row in candidates[["u", "v", "key"]].itertuples(index=False)
        ]
        rows.append(
            {
                "segment_id": str(survey.segment_id),
                "geometry_osm_id": requested_id,
                "inferred_site_id": str(survey.inferred_site_id),
                "count_join_status": str(survey.count_join_status),
                "candidate_directed_edges": len(candidates),
                "minimum_distance_m": minimum_distance,
                "survey_coverage_fraction_15m": min(coverage, 1.0),
                "spatial_match_status": status,
                "spatial_match_reliable": bool(source_join_ok and geometry_supported),
                # Publisher codes 0/1 have no semantic legend, so edge travel direction
                # cannot be claimed even when current geometry is a strong spatial match.
                "routing_direction_verified": False,
                "edge_references": "|".join(sorted(references)),
            }
        )
    return pd.DataFrame(rows).sort_values("segment_id").reset_index(drop=True)


def _graph_bundle(
    graph: Any,
    staging: Path,
    *,
    study_dir: Path,
    projected_crs: str,
    bbox: tuple[float, float, float, float],
    retrieval_attempts: int,
    maximum_match_distance_m: float,
    minimum_match_coverage: float,
) -> dict[str, Any]:
    gpd, nx, ox, _, _, box, _ = _dependencies()
    if not graph.is_directed():
        raise ObservedRoadError("Downloaded OSM drive graph is unexpectedly undirected")
    projected = ox.projection.project_graph(graph, to_crs=projected_crs)
    component_stats = _component_stats(projected, nx)
    nodes, edges = ox.convert.graph_to_gdfs(
        projected, nodes=True, edges=True, fill_edge_geometry=True
    )
    safe_edges = gpd.GeoDataFrame(_safe_edges(edges), geometry="geometry", crs=projected_crs)
    safe_nodes = gpd.GeoDataFrame(_safe_nodes(nodes), geometry="geometry", crs=projected_crs)
    survey = spatial_segment_features(study_dir, projected_crs=projected_crs)
    analysis_aoi = gpd.GeoSeries([box(*bbox)], crs="EPSG:4326").to_crs(projected_crs).iloc[0]
    matches = _map_match_segments(
        survey,
        safe_edges,
        maximum_distance_m=maximum_match_distance_m,
        minimum_coverage=minimum_match_coverage,
        analysis_aoi=analysis_aoi,
    )

    ox.io.save_graphml(projected, staging / "roads.graphml")
    safe_edges.to_file(staging / "edges_projected.geojson", driver="GeoJSON")
    safe_nodes.to_file(staging / "nodes_projected.geojson", driver="GeoJSON")
    safe_edges.to_crs("EPSG:4326").to_file(
        staging / "edges_wgs84.geojson", driver="GeoJSON"
    )
    matches.to_csv(staging / "survey_edge_matches.csv", index=False, lineterminator="\n")
    return {
        "source_kind": "overpass_current_osm",
        "source_label": "current OpenStreetMap drive graph retrieved through Overpass",
        "routable_graph": True,
        "network_type": "drive",
        "bbox_wgs84": list(bbox),
        "graph_crs": projected_crs,
        "geometry_export_crs": {
            "edges_projected.geojson": projected_crs,
            "nodes_projected.geojson": projected_crs,
            "edges_wgs84.geojson": "EPSG:4326",
        },
        "directed": True,
        "nodes": int(projected.number_of_nodes()),
        "directed_edges": int(projected.number_of_edges()),
        "retrieval_attempts": retrieval_attempts,
        "oneway_missing_edges": int(safe_edges["oneway"].eq("").sum()),
        "edge_length_m": {
            "minimum": float(safe_edges["length_m_projected"].min()),
            "maximum": float(safe_edges["length_m_projected"].max()),
            "total_directed": float(safe_edges["length_m_projected"].sum()),
        },
        "connectivity": component_stats,
        "map_matching": {
            "method": "released geometry OSM ID followed by projected geometry coverage",
            "maximum_distance_m": maximum_match_distance_m,
            "coverage_buffer_m": 15.0,
            "minimum_coverage_fraction": minimum_match_coverage,
            "survey_segments": len(matches),
            "within_selected_study_aoi": int(
                matches["spatial_match_status"].ne("outside_selected_study_aoi").sum()
            ),
            "outside_selected_study_aoi": int(
                matches["spatial_match_status"].eq("outside_selected_study_aoi").sum()
            ),
            "spatially_reliable": int(matches["spatial_match_reliable"].sum()),
            "unreliable_or_unmatched": int((~matches["spatial_match_reliable"]).sum()),
            "routing_direction_verified": 0,
            "direction_note": (
                "TfC direction codes 0/1 lack a semantic legend; no surveyed link is "
                "assigned to a legal OSM travel direction for demand routing."
            ),
        },
        "simulation_readiness": "blocked_missing_turns_routes_and_signal_observations",
    }


def _read_pbf_lines(
    pbf_path: Path, bbox: tuple[float, float, float, float], projected_crs: str
):
    gpd, _, _, _, _, box, _ = _dependencies()
    try:
        lines = gpd.read_file(
            pbf_path,
            layer="lines",
            bbox=bbox,
            engine="pyogrio",
            use_arrow=True,
        )
    except Exception as exc:
        raise ObservedRoadError(f"Could not read the Geofabrik PBF line layer: {exc}") from exc
    if lines.empty or lines.crs is None:
        raise ObservedRoadError("Geofabrik PBF contains no readable lines in the selected AOI")
    if "highway" not in lines:
        raise ObservedRoadError("Geofabrik PBF line layer exposes no highway field")
    roads = lines.loc[lines["highway"].notna() & lines.geometry.notna()].copy()
    if roads.empty:
        raise ObservedRoadError("Geofabrik PBF contains no highway geometry in the AOI")
    aoi = gpd.GeoSeries([box(*bbox)], crs="EPSG:4326").to_crs(projected_crs).iloc[0]
    roads = roads.to_crs(projected_crs)
    roads.geometry = roads.geometry.intersection(aoi)
    roads = roads.loc[~roads.geometry.is_empty & roads.geometry.notna()].copy()
    roads["length_m_projected"] = roads.geometry.length.astype(float)
    roads = roads.loc[roads["length_m_projected"] > 0].copy()
    if roads.empty:
        raise ObservedRoadError("PBF highway geometry is empty after projected AOI clipping")
    return roads


def _pbf_geometry_bundle(
    pbf_path: Path,
    staging: Path,
    *,
    bbox: tuple[float, float, float, float],
    projected_crs: str,
) -> dict[str, Any]:
    gpd, _, _, _, Point, _, _ = _dependencies()
    roads = _read_pbf_lines(pbf_path, bbox, projected_crs)
    keep = [column for column in ("osm_id", "name", "highway", "other_tags") if column in roads]
    geometry = roads[[*keep, "length_m_projected", "geometry"]].copy()
    for column in keep:
        geometry[column] = geometry[column].map(_json_text)
    geometry.to_file(staging / "edges_projected.geojson", driver="GeoJSON")
    geometry.to_crs("EPSG:4326").to_file(
        staging / "edges_wgs84.geojson", driver="GeoJSON"
    )
    endpoints = []
    for value in geometry.geometry:
        if value.geom_type == "LineString":
            endpoints.extend((Point(value.coords[0]), Point(value.coords[-1])))
        elif value.geom_type == "MultiLineString":
            for line in value.geoms:
                endpoints.extend((Point(line.coords[0]), Point(line.coords[-1])))
    nodes = gpd.GeoDataFrame(
        {
            "node_role": ["geometry_endpoint_not_osm_node"] * len(endpoints),
            "geometry": endpoints,
        },
        crs=projected_crs,
    ).drop_duplicates(subset="geometry")
    nodes.to_file(staging / "nodes_projected.geojson", driver="GeoJSON")
    return {
        "source_kind": "geofabrik_pbf_geometry_only",
        "source_label": "offline Geofabrik Egypt OSM PBF line geometry",
        "source_filename": pbf_path.name,
        "source_sha256": sha256_file(pbf_path),
        "routable_graph": False,
        "network_type": "highway geometry (not a legal routing graph)",
        "bbox_wgs84": list(bbox),
        "graph_crs": None,
        "geometry_export_crs": {
            "edges_projected.geojson": projected_crs,
            "nodes_projected.geojson": projected_crs,
            "edges_wgs84.geojson": "EPSG:4326",
        },
        "directed": None,
        "geometry_features": len(geometry),
        "endpoint_features": len(nodes),
        "edge_length_m": {
            "minimum": float(geometry["length_m_projected"].min()),
            "maximum": float(geometry["length_m_projected"].max()),
            "total_undirected_geometry": float(geometry["length_m_projected"].sum()),
        },
        "connectivity": None,
        "map_matching": None,
        "simulation_readiness": "not_routable_geometry_only",
        "limitation": (
            "GDAL PBF line extraction does not preserve a topologically noded, directed legal "
            "road graph. It is suitable only as offline map context."
        ),
    }


def prepare_observed_roads(
    study_dir: str | Path,
    output_dir: str | Path,
    config: Mapping[str, Any],
    *,
    pbf_path: str | Path | None = None,
    retry_attempts: int = 3,
    retry_delays_s: tuple[float, ...] = (2.0, 8.0, 20.0),
    maximum_match_distance_m: float = 25.0,
    minimum_match_coverage: float = 0.8,
    graph_fetcher: Callable[..., tuple[Any, int]] | None = None,
) -> dict[str, Any]:
    """Atomically create current-road context without overwriting prior evidence."""
    study_dir = Path(study_dir).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite observed-road bundle: {output_dir}")
    bbox, projected_crs = _bbox_and_crs(config)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        fetch = graph_fetcher or _fetch_overpass_graph
        overpass_error = None
        try:
            graph, attempts = fetch(
                bbox,
                cache_folder=output_dir.parent / "overpass_cache",
                retry_attempts=retry_attempts,
                retry_delays_s=retry_delays_s,
            )
            details = _graph_bundle(
                graph,
                staging,
                study_dir=study_dir,
                projected_crs=projected_crs,
                bbox=bbox,
                retrieval_attempts=attempts,
                maximum_match_distance_m=maximum_match_distance_m,
                minimum_match_coverage=minimum_match_coverage,
            )
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:
            overpass_error = f"{type(exc).__name__}: {exc}"
            if pbf_path is None:
                raise ObservedRoadError(
                    "Overpass road retrieval failed and no Geofabrik PBF fallback was supplied: "
                    f"{overpass_error}"
                ) from exc
            pbf = Path(pbf_path).resolve()
            if not pbf.is_file():
                raise FileNotFoundError(f"Geofabrik PBF fallback is not a file: {pbf}") from exc
            details = _pbf_geometry_bundle(
                pbf, staging, bbox=bbox, projected_crs=projected_crs
            )
        artifacts = {
            path.relative_to(staging).as_posix(): {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in sorted(staging.rglob("*"))
            if path.is_file()
        }
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "data_kind": "third_party_osm_current_road_context",
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "attribution": OSM_ATTRIBUTION,
            "license": "Open Database License (ODbL) 1.0",
            "source_url": (
                "https://www.openstreetmap.org/copyright"
                if details["source_kind"] == "overpass_current_osm"
                else "https://download.geofabrik.de/africa/egypt.html"
            ),
            "overpass_error_before_fallback": overpass_error,
            **details,
            "artifacts": artifacts,
            "interpretation_limits": [
                "Current OSM geometry is third-party map data, not a 2024 traffic measurement.",
                "No signal phases, timings, turning movements, or route demand are observed here.",
                "This bundle is not evidence of a calibrated Cairo traffic simulation.",
            ],
        }
        _write_json(manifest, staging / "road_manifest.json")
        os.replace(staging, output_dir)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def validate_observed_road_bundle(output_dir: str | Path) -> dict[str, Any] | None:
    """Read and checksum-validate an existing road context bundle."""
    root = Path(output_dir).resolve()
    if not root.exists():
        return None
    manifest_path = root / "road_manifest.json"
    if not root.is_dir() or not manifest_path.is_file():
        raise ObservedRoadError(f"Observed-road bundle is incomplete: {root}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ObservedRoadError(f"Invalid observed-road manifest: {manifest_path}") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != SCHEMA_VERSION:
        raise ObservedRoadError("Observed-road bundle schema is incompatible")
    if manifest.get("data_kind") != "third_party_osm_current_road_context":
        raise ObservedRoadError("Observed-road bundle has the wrong data_kind")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ObservedRoadError("Observed-road manifest has no artifact hashes")
    for relative, record in artifacts.items():
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ObservedRoadError(f"Observed-road artifact escapes bundle: {relative}") from exc
        if not isinstance(record, dict) or not path.is_file():
            raise ObservedRoadError(f"Observed-road artifact is missing: {relative}")
        if record.get("sha256") != sha256_file(path) or record.get("bytes") != path.stat().st_size:
            raise ObservedRoadError(f"Observed-road artifact hash mismatch: {relative}")
    required = {"edges_projected.geojson", "edges_wgs84.geojson", "nodes_projected.geojson"}
    if manifest.get("routable_graph") is True:
        required |= {"roads.graphml", "survey_edge_matches.csv"}
        if manifest.get("directed") is not True or not isinstance(manifest.get("connectivity"), dict):
            raise ObservedRoadError("Routable observed-road graph lacks direction/connectivity")
    elif manifest.get("source_kind") != "geofabrik_pbf_geometry_only":
        raise ObservedRoadError("Non-routable road bundle is not the declared PBF fallback")
    missing = sorted(required - set(artifacts))
    if missing:
        raise ObservedRoadError("Observed-road bundle omits: " + ", ".join(missing))
    return manifest


__all__ = [
    "OSM_ATTRIBUTION",
    "ObservedRoadError",
    "prepare_observed_roads",
    "validate_observed_road_bundle",
]
