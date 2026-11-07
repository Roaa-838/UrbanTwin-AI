"""Earth-observation ingestion on one projected, aligned raster grid.

Real mode intentionally has no synthetic fallback.  A failed STAC query,
missing dependency, unreadable asset, or empty cloud-free overlap raises an
error and leaves no output that could be mistaken for a successful download.
"""

from __future__ import annotations

import math
import re
import tempfile
import warnings
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from urbantwin.common import save_json, sha256_file

WORLDCOVER_CLASSES: dict[int, str] = {
    10: "Tree cover",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    60: "Bare / sparse vegetation",
    70: "Snow and ice",
    80: "Permanent water bodies",
    90: "Herbaceous wetland",
    95: "Mangroves",
    100: "Moss and lichen",
}

BAND_ORDER = ("B02", "B03", "B04", "B08", "B11")
BAND_NAMES = ("blue", "green", "red", "nir", "swir1")
# Sentinel-2 L2A Scene Classification Layer classes retained as clear surface.
# 4 vegetation, 5 not vegetated, 6 water, 7 unclassified.
CLEAR_SCL_CLASSES = (4, 5, 6, 7)
STAC_ENDPOINT = "https://planetarycomputer.microsoft.com/api/stac/v1"
SENTINEL_COLLECTION = "sentinel-2-l2a"
WORLDCOVER_COLLECTION = "esa-worldcover"
PROVENANCE_SCHEMA = "urbantwin.eo-provenance/v1"


def compute_indices(bands: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute NDVI and NDBI from ``[blue, green, red, nir, swir1]``.

    Division by zero and non-finite inputs are represented by ``NaN`` instead
    of silently becoming zero or infinity.
    """

    values = np.asarray(bands, dtype=np.float32)
    if values.ndim != 3 or values.shape[0] != len(BAND_ORDER):
        raise ValueError("Expected bands with shape (5, height, width) in B02/B03/B04/B08/B11 order")

    red = values[2]
    nir = values[3]
    swir = values[4]

    def normalized_difference(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        numerator = left - right
        denominator = left + right
        result = np.full(left.shape, np.nan, dtype=np.float32)
        valid = np.isfinite(numerator) & np.isfinite(denominator) & (denominator != 0)
        np.divide(numerator, denominator, out=result, where=valid)
        return result

    return normalized_difference(nir, red), normalized_difference(swir, nir)


def _geo_dependencies():
    try:
        from affine import Affine
        from pyproj import CRS, Transformer
    except ImportError as exc:  # pragma: no cover - depends on installation profile
        raise RuntimeError(
            "Projected raster processing requires rasterio and pyproj. "
            "Install the project with `python -m pip install -e .`."
        ) from exc
    return Affine, CRS, Transformer


def grid_for_aoi(config: dict[str, Any]):
    """Return ``(transform, height, width)`` for a projected AOI grid.

    ``bbox`` is always interpreted as WGS84 ``[west, south, east, north]``.
    Grid edges are snapped outwards to the requested metre resolution so the
    entire AOI is covered deterministically.
    """

    if not isinstance(config, dict):
        raise TypeError("AOI configuration must be a mapping")
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

    try:
        resolution = float(config["resolution_m"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("resolution_m must be a positive number of metres") from exc
    if not math.isfinite(resolution) or resolution <= 0:
        raise ValueError("resolution_m must be a positive number of metres")

    Affine, CRS, Transformer = _geo_dependencies()
    try:
        target_crs = CRS.from_user_input(config["crs"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("A valid projected target `crs` is required") from exc
    if not target_crs.is_projected:
        raise ValueError("Target CRS must be projected; geographic degrees are not valid metre pixels")
    axes = target_crs.axis_info
    if not axes or not math.isclose(float(axes[0].unit_conversion_factor), 1.0, rel_tol=0, abs_tol=1e-9):
        raise ValueError("Target CRS axes must use metres because resolution_m is specified in metres")

    transformer = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
    left_raw, bottom_raw, right_raw, top_raw = transformer.transform_bounds(
        west, south, east, north, densify_pts=21
    )
    if not all(math.isfinite(v) for v in (left_raw, bottom_raw, right_raw, top_raw)):
        raise ValueError("AOI could not be transformed into the configured CRS")
    left = math.floor(left_raw / resolution) * resolution
    bottom = math.floor(bottom_raw / resolution) * resolution
    right = math.ceil(right_raw / resolution) * resolution
    top = math.ceil(top_raw / resolution) * resolution
    width = round((right - left) / resolution)
    height = round((top - bottom) / resolution)
    if width <= 0 or height <= 0:
        raise ValueError("Projected AOI produced an empty raster grid")
    max_pixels = int(config.get("max_pixels", 400_000))
    if max_pixels <= 0:
        raise ValueError("max_pixels must be positive")
    if width * height > max_pixels:
        raise ValueError(
            f"Requested AOI is {width * height:,} pixels, above max_pixels={max_pixels:,}; "
            "shrink the bbox, use a coarser resolution, or raise the cap explicitly"
        )
    return Affine(resolution, 0.0, left, 0.0, -resolution, top), height, width


def _composite_scenes(
    scene_bands: np.ndarray, clear_masks: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mask and median-composite aligned scenes.

    Returns ``(bands, valid_mask, observation_count)``.  A scene contributes to
    a pixel only when the SCL mask is clear *and all five bands are finite*.
    Keeping this operation separate makes cloud-mask behavior offline-testable.
    """

    stack = np.asarray(scene_bands, dtype=np.float32)
    masks = np.asarray(clear_masks, dtype=bool)
    if stack.ndim != 4 or stack.shape[1] != len(BAND_ORDER):
        raise ValueError("scene_bands must have shape (scene, 5, height, width)")
    if masks.shape != (stack.shape[0], stack.shape[2], stack.shape[3]):
        raise ValueError("clear_masks must have shape (scene, height, width)")
    usable = masks & np.all(np.isfinite(stack), axis=1)
    masked = np.where(usable[:, None, :, :], stack, np.nan)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
        composite = np.nanmedian(masked, axis=0).astype(np.float32, copy=False)
    observation_count = usable.sum(axis=0).astype(np.uint16)
    valid = (observation_count > 0) & np.all(np.isfinite(composite), axis=0)
    composite[:, ~valid] = np.nan
    return composite, valid, observation_count


def _reproject_asset(
    href: str,
    transform: Any,
    height: int,
    width: int,
    target_crs: Any,
    *,
    categorical: bool,
    scale: float = 1.0,
    offset: float = 0.0,
):
    """Read one remote/local raster and align it to the destination grid."""

    try:
        import rasterio
        from rasterio.warp import Resampling, reproject
    except ImportError as exc:  # pragma: no cover - depends on installation profile
        raise RuntimeError("Raster ingestion requires rasterio") from exc

    if categorical:
        destination = np.zeros((height, width), dtype=np.uint16)
        dst_nodata: float | int = 0
        resampling = Resampling.nearest
    else:
        destination = np.full((height, width), np.nan, dtype=np.float32)
        dst_nodata = np.nan
        resampling = Resampling.bilinear

    with rasterio.open(href) as source:
        if source.crs is None:
            raise RuntimeError("Raster asset has no CRS and cannot be aligned")
        src_nodata = source.nodata
        # Sentinel and WorldCover COGs use zero as their declared/semantic
        # no-data code even when a driver omits the metadata field.
        if src_nodata is None:
            src_nodata = 0
        reproject(
            source=rasterio.band(source, 1),
            destination=destination,
            src_transform=source.transform,
            src_crs=source.crs,
            src_nodata=src_nodata,
            dst_transform=transform,
            dst_crs=target_crs,
            dst_nodata=dst_nodata,
            resampling=resampling,
            init_dest_nodata=True,
            num_threads=1,
        )
        dataset_scale = float(source.scales[0]) if source.scales else 1.0
        dataset_offset = float(source.offsets[0]) if source.offsets else 0.0

    if categorical:
        return destination, dataset_scale, dataset_offset
    destination = destination * np.float32(scale) + np.float32(offset)
    return destination.astype(np.float32, copy=False), dataset_scale, dataset_offset


def _raster_band_metadata(asset: Any) -> dict[str, Any]:
    fields = getattr(asset, "extra_fields", {}) or {}
    bands = fields.get("raster:bands")
    if isinstance(bands, list) and bands and isinstance(bands[0], dict):
        return bands[0]
    return {}


def _sentinel_scale_offset(item: Any, asset: Any, dataset_scale: float, dataset_offset: float):
    """Resolve L2A DN-to-reflectance scaling and record how it was chosen."""

    metadata = _raster_band_metadata(asset)
    if metadata.get("scale") is not None or metadata.get("offset") is not None:
        return (
            float(metadata.get("scale", 1.0)),
            float(metadata.get("offset", 0.0)),
            "STAC raster:bands",
        )
    if not math.isclose(dataset_scale, 1.0) or not math.isclose(dataset_offset, 0.0):
        return dataset_scale, dataset_offset, "GeoTIFF scale/offset"

    # Sentinel-2 L2A stores BOA reflectance with QUANTIFICATION_VALUE=10000.
    # Processing baseline 04.00 introduced BOA_ADD_OFFSET=-1000.  The 2021
    # study configuration predates that change, but the rule also keeps this
    # function correct for explicitly configured later acquisitions.
    raw_baseline = (getattr(item, "properties", {}) or {}).get("s2:processing_baseline", "0")
    try:
        baseline = float(raw_baseline)
    except (TypeError, ValueError):
        baseline = 0.0
    return 0.0001, (-0.1 if baseline >= 4.0 else 0.0), "Sentinel-2 L2A quantification metadata rule"


def _item_datetime(item: Any) -> str | None:
    dt = getattr(item, "datetime", None)
    if dt is not None:
        return dt.isoformat()
    properties = getattr(item, "properties", {}) or {}
    value = properties.get("datetime") or properties.get("start_datetime")
    return str(value) if value is not None else None


def _worldcover_year(item: Any) -> int | None:
    properties = getattr(item, "properties", {}) or {}
    for key in ("year", "worldcover:year", "esa_worldcover:year"):
        if key in properties:
            try:
                return int(properties[key])
            except (TypeError, ValueError):
                pass
    for value in (getattr(item, "id", ""), _item_datetime(item) or ""):
        match = re.search(r"(?<!\d)(20\d{2})(?!\d)", str(value))
        if match:
            return int(match.group(1))
    return None


def _search_items(client: Any, **kwargs: Any) -> list[Any]:
    try:
        return list(client.search(**kwargs).items())
    except Exception as exc:
        raise RuntimeError(f"STAC search failed for {kwargs.get('collections')}: {exc}") from exc


def _asset(item: Any, name: str):
    assets = getattr(item, "assets", {}) or {}
    if name not in assets:
        raise RuntimeError(f"STAC item {getattr(item, 'id', '<unknown>')} has no required asset {name}")
    return assets[name]


def _sentinel_scene(
    item: Any, transform: Any, height: int, width: int, target_crs: Any
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    scl_asset = _asset(item, "SCL")
    scl, _, _ = _reproject_asset(
        scl_asset.href, transform, height, width, target_crs, categorical=True
    )
    clear = np.isin(scl, CLEAR_SCL_CLASSES)
    arrays: list[np.ndarray] = []
    scaling: dict[str, dict[str, Any]] = {}
    for name in BAND_ORDER:
        asset = _asset(item, name)
        # Read once with identity scaling so dataset metadata is available, then
        # apply the explicitly resolved L2A conversion without resampling twice.
        raw, dataset_scale, dataset_offset = _reproject_asset(
            asset.href, transform, height, width, target_crs, categorical=False
        )
        scale, offset, source = _sentinel_scale_offset(item, asset, dataset_scale, dataset_offset)
        array = raw * np.float32(scale) + np.float32(offset)
        arrays.append(array.astype(np.float32, copy=False))
        scaling[name] = {"scale": scale, "offset": offset, "source": source}
    return np.stack(arrays), clear, scaling


def _worldcover_labels(
    items: Iterable[Any], transform: Any, height: int, width: int, target_crs: Any
) -> tuple[np.ndarray, int]:
    labels = np.zeros((height, width), dtype=np.uint8)
    conflicts = 0
    supported = np.array(sorted(WORLDCOVER_CLASSES), dtype=np.uint16)
    for item in sorted(items, key=lambda candidate: str(candidate.id)):
        asset = _asset(item, "map")
        tile, _, _ = _reproject_asset(
            asset.href, transform, height, width, target_crs, categorical=True
        )
        tile_valid = np.isin(tile, supported)
        overlap = tile_valid & (labels != 0)
        conflicts += int(np.count_nonzero(overlap & (labels != tile)))
        labels[tile_valid] = tile[tile_valid].astype(np.uint8)
    if conflicts:
        raise RuntimeError(f"Overlapping WorldCover tiles disagree at {conflicts} destination pixels")
    return labels, conflicts


def _transform_coefficients(transform: Any) -> list[float]:
    return [
        float(transform.a),
        float(transform.b),
        float(transform.c),
        float(transform.d),
        float(transform.e),
        float(transform.f),
    ]


def _write_stack_atomic(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.stem}-", suffix=".npz", delete=False) as handle:
            temporary = Path(handle.name)
        np.savez_compressed(temporary, **arrays)
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def fetch_real(config: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    """Fetch real Sentinel-2 L2A and ESA WorldCover 2021 for an AOI.

    All assets are reprojected onto the exact grid returned by
    :func:`grid_for_aoi`. Continuous reflectance bands use bilinear resampling;
    SCL and WorldCover use nearest-neighbour resampling.  Signed asset URLs are
    used only in memory and are never written to provenance.
    """

    if not isinstance(config, dict):
        raise TypeError("Configuration must be a mapping")
    if config.get("mode") != "real":
        raise ValueError("fetch_real requires configuration mode: real; synthetic substitution is forbidden")
    try:
        import planetary_computer
        import rasterio  # noqa: F401 - fail before a network request if the dependency is absent
        from pyproj import CRS
        from pystac_client import Client
    except ImportError as exc:  # pragma: no cover - depends on installation profile
        raise RuntimeError(
            "Real EO ingestion requires the `eo` dependencies. Install with "
            "`python -m pip install -e \".[eo]\"`."
        ) from exc

    transform, height, width = grid_for_aoi(config)
    target_crs = CRS.from_user_input(config["crs"])
    target_crs_for_rasterio = target_crs.to_string()
    bbox = [float(v) for v in config["bbox"]]
    period = config.get("sentinel_period")
    if not isinstance(period, (list, tuple)) or len(period) != 2:
        raise ValueError("sentinel_period must contain [start_date, end_date]")
    date_range = f"{period[0]}/{period[1]}"
    max_cloud = float(config.get("max_scene_cloud_pct", 35))
    if not 0 <= max_cloud <= 100:
        raise ValueError("max_scene_cloud_pct must be in [0, 100]")
    scene_count = int(config.get("composite_scenes", 3))
    if scene_count <= 0:
        raise ValueError("composite_scenes must be positive")
    worldcover_year = int(config.get("worldcover_year", 2021))
    if worldcover_year != 2021:
        raise ValueError("This pipeline supports the ESA WorldCover 2021 product only")

    try:
        client = Client.open(STAC_ENDPOINT, modifier=planetary_computer.sign_inplace)
    except Exception as exc:
        raise RuntimeError(f"Could not connect to the Planetary Computer STAC API: {exc}") from exc

    sentinel = _search_items(
        client,
        collections=[SENTINEL_COLLECTION],
        bbox=bbox,
        datetime=date_range,
        query={"eo:cloud_cover": {"lte": max_cloud}},
    )
    sentinel = [
        item
        for item in sentinel
        if float((getattr(item, "properties", {}) or {}).get("eo:cloud_cover", 101)) <= max_cloud
    ]
    sentinel.sort(
        key=lambda item: (
            float((getattr(item, "properties", {}) or {}).get("eo:cloud_cover", 101)),
            _item_datetime(item) or "",
            str(item.id),
        )
    )
    selected_sentinel = sentinel[:scene_count]
    if not selected_sentinel:
        raise RuntimeError(
            f"No Sentinel-2 L2A scenes matched {date_range}, AOI, and cloud <= {max_cloud}%"
        )

    scenes: list[np.ndarray] = []
    clear_masks: list[np.ndarray] = []
    scaling_by_item: dict[str, Any] = {}
    for item in selected_sentinel:
        try:
            bands, clear, scaling = _sentinel_scene(
                item, transform, height, width, target_crs_for_rasterio
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to read Sentinel-2 item {item.id}: {exc}") from exc
        scenes.append(bands)
        clear_masks.append(clear)
        scaling_by_item[str(item.id)] = scaling
    composite, clear_valid, observation_count = _composite_scenes(
        np.stack(scenes), np.stack(clear_masks)
    )

    worldcover_candidates = _search_items(
        client, collections=[WORLDCOVER_COLLECTION], bbox=bbox
    )
    worldcover = [item for item in worldcover_candidates if _worldcover_year(item) == worldcover_year]
    if not worldcover:
        raise RuntimeError(f"No ESA WorldCover {worldcover_year} tile matched the configured AOI")
    labels, _ = _worldcover_labels(
        worldcover, transform, height, width, target_crs_for_rasterio
    )
    valid_mask = clear_valid & np.isin(labels, sorted(WORLDCOVER_CLASSES))
    if not np.any(valid_mask):
        raise RuntimeError("The selected assets have no cloud-free Sentinel/WorldCover overlap")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    stack_path = destination / "eo_stack.npz"
    _write_stack_atomic(
        stack_path,
        bands=composite.astype(np.float32, copy=False),
        labels=labels.astype(np.uint8, copy=False),
        valid_mask=valid_mask,
        observation_count=observation_count,
    )

    sentinel_details = []
    for item in selected_sentinel:
        properties = getattr(item, "properties", {}) or {}
        sentinel_details.append(
            {
                "id": str(item.id),
                "datetime": _item_datetime(item),
                "eo_cloud_cover_pct": properties.get("eo:cloud_cover"),
                "processing_baseline": properties.get("s2:processing_baseline"),
            }
        )
    worldcover_details = [
        {"id": str(item.id), "product_year": _worldcover_year(item)}
        for item in sorted(worldcover, key=lambda candidate: str(candidate.id))
    ]
    provenance: dict[str, Any] = {
        "schema_version": PROVENANCE_SCHEMA,
        "data_kind": "real",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": dict(config),
        "crs": target_crs.to_string(),
        "transform": _transform_coefficients(transform),
        "width": width,
        "height": height,
        "band_order": list(BAND_ORDER),
        "band_names": list(BAND_NAMES),
        "array_keys": ["bands", "labels", "valid_mask", "observation_count"],
        "valid_pixel_count": int(np.count_nonzero(valid_mask)),
        "grid_pixel_count": int(width * height),
        "sentinel_stac_collection": SENTINEL_COLLECTION,
        "sentinel_stac_item_ids": [str(item.id) for item in selected_sentinel],
        "sentinel_items": sentinel_details,
        "worldcover_stac_collection": WORLDCOVER_COLLECTION,
        "worldcover_stac_item_ids": [entry["id"] for entry in worldcover_details],
        "worldcover_items": worldcover_details,
        "stac_endpoint": STAC_ENDPOINT,
        "processing": {
            "continuous_resampling": "bilinear",
            "categorical_resampling": "nearest",
            "clear_scl_classes": list(CLEAR_SCL_CLASSES),
            "cloud_mask_rule": "SCL in {4,5,6,7}; all five reflectance bands finite",
            "composite": "per-band median over scenes passing the shared per-scene clear mask",
            "reflectance_scaling_by_item": scaling_by_item,
            "worldcover_nodata": 0,
        },
        "source_labels": {
            "imagery": "real Copernicus Sentinel-2 Level-2A surface reflectance",
            "reference_labels": "real ESA WorldCover 2021 categorical product",
        },
        "attribution": {
            "sentinel_2": "Contains modified Copernicus Sentinel data",
            "worldcover": "ESA WorldCover 2021 © ESA, licensed under CC BY 4.0",
            "distribution": "Assets accessed through Microsoft Planetary Computer STAC",
        },
        "signed_asset_urls_recorded": False,
        "eo_stack_sha256": sha256_file(stack_path),
    }
    save_json(provenance, destination / "provenance.json")
    return provenance


__all__ = [
    "BAND_NAMES",
    "BAND_ORDER",
    "CLEAR_SCL_CLASSES",
    "WORLDCOVER_CLASSES",
    "compute_indices",
    "fetch_real",
    "grid_for_aoi",
]
