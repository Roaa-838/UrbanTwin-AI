"""Shared configuration and provenance utilities."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_yaml(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as f:
        result = yaml.safe_load(f)
    if not isinstance(result, dict):
        raise TypeError(f"Expected YAML mapping: {path}")
    return result


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def metadata(kind: str, config: dict, **extra: Any) -> dict:
    return {
        "data_kind": kind,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": config,
        **extra,
    }


def require_real_provenance(meta: dict) -> None:
    """Require the complete provenance contract emitted by real EO ingestion.

    A caller-provided ``data_kind: real`` flag is not sufficient evidence.  The
    required fields bind a stack to named Sentinel-2 and WorldCover STAC items,
    the projected grid, and the documented mask/resampling protocol.
    """
    if not isinstance(meta, dict):
        raise TypeError("Research analysis requires provenance as a JSON object.")
    if meta.get("data_kind") != "real":
        raise ValueError("Research analysis requires real provenance. Synthetic test data rejected.")
    if meta.get("schema_version") != "urbantwin.eo-provenance/v1":
        raise ValueError("Real EO provenance has an unsupported or missing schema_version.")
    if meta.get("sentinel_stac_collection") != "sentinel-2-l2a":
        raise ValueError("Real EO provenance must identify the Sentinel-2 L2A collection.")
    if meta.get("worldcover_stac_collection") != "esa-worldcover":
        raise ValueError("Real EO provenance must identify the ESA WorldCover collection.")
    for key in ("sentinel_stac_item_ids", "worldcover_stac_item_ids"):
        values = meta.get(key)
        if not isinstance(values, list) or not values or not all(
            isinstance(value, str) and value for value in values
        ):
            raise ValueError(f"Real EO provenance must contain non-empty {key}.")
    if meta.get("band_order") != ["B02", "B03", "B04", "B08", "B11"]:
        raise ValueError("Real EO provenance has an unsupported band order.")
    if not isinstance(meta.get("crs"), str) or not meta["crs"]:
        raise ValueError("Real EO provenance must identify its projected CRS.")
    transform = meta.get("transform")
    if not isinstance(transform, list) or len(transform) != 6:
        raise ValueError("Real EO provenance must contain a six-coefficient affine transform.")
    if not isinstance(meta.get("width"), int) or meta["width"] <= 0:
        raise ValueError("Real EO provenance must contain a positive raster width.")
    if not isinstance(meta.get("height"), int) or meta["height"] <= 0:
        raise ValueError("Real EO provenance must contain a positive raster height.")
    processing = meta.get("processing")
    if not isinstance(processing, dict):
        raise TypeError("Real EO provenance must document raster processing.")
    if processing.get("continuous_resampling") != "bilinear":
        raise ValueError("Real EO provenance must document bilinear continuous resampling.")
    if processing.get("categorical_resampling") != "nearest":
        raise ValueError("Real EO provenance must document nearest categorical resampling.")
    if processing.get("clear_scl_classes") != [4, 5, 6, 7]:
        raise ValueError("Real EO provenance must document the supported SCL clear classes.")
    stack_hash = meta.get("eo_stack_sha256")
    if not isinstance(stack_hash, str) or len(stack_hash) != 64:
        raise ValueError("Real EO provenance must contain eo_stack_sha256.")
    if meta.get("signed_asset_urls_recorded") is not False:
        raise ValueError("Real EO provenance must not retain signed STAC asset URLs.")
