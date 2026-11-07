"""Deterministic synthetic fixture for offline software checks only."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from urbantwin.common import save_json, sha256_file
from urbantwin.data.earth_observation import (
    BAND_NAMES,
    BAND_ORDER,
    PROVENANCE_SCHEMA,
    _write_stack_atomic,
)

_FIXTURE_CLASSES = np.array([10, 50, 60, 80], dtype=np.uint8)
_CLASS_REFLECTANCE = {
    # blue, green, red, nir, swir1 -- deliberately separable test patterns.
    10: (0.05, 0.10, 0.08, 0.55, 0.20),
    50: (0.18, 0.22, 0.28, 0.30, 0.45),
    60: (0.20, 0.26, 0.32, 0.36, 0.42),
    80: (0.08, 0.10, 0.06, 0.03, 0.02),
}


def write_demo(output_dir: str | Path, n: int = 120, seed: int = 20240229) -> dict[str, Any]:
    """Write a reproducible, explicitly synthetic raster fixture.

    No wall-clock value or output path is embedded, so identical parameters
    produce identical arrays, provenance content, and stack hashes in separate
    directories.  This function is never called by :func:`fetch_real`.
    """

    if isinstance(n, bool) or int(n) != n or int(n) < 40:
        raise ValueError("n must be an integer of at least 40 pixels")
    n = int(n)
    if isinstance(seed, bool) or int(seed) != seed:
        raise ValueError("seed must be an integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)

    rows, cols = np.indices((n, n))
    # Eight-pixel patches place every fixture class in many independent
    # 20-pixel evaluation blocks, avoiding an accidental single-class split.
    class_index = ((rows // 8) + 3 * (cols // 8)) % len(_FIXTURE_CLASSES)
    labels = _FIXTURE_CLASSES[class_index]
    bands = np.empty((len(BAND_ORDER), n, n), dtype=np.float32)
    for class_id, spectrum in _CLASS_REFLECTANCE.items():
        selected = labels == class_id
        for band_index, value in enumerate(spectrum):
            bands[band_index, selected] = value
    bands += rng.normal(0.0, 0.008, size=bands.shape).astype(np.float32)
    np.clip(bands, 0.0, 1.0, out=bands)

    # A stable missing-data pattern exercises mask propagation without making
    # the spatial classifier dependent on wall-clock or process randomness.
    missing = rng.random((n, n)) < 0.0125
    bands[:, missing] = np.nan
    valid_mask = ~missing
    observation_count = valid_mask.astype(np.uint16)

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    stack_path = destination / "eo_stack.npz"
    _write_stack_atomic(
        stack_path,
        bands=bands,
        labels=labels,
        valid_mask=valid_mask,
        observation_count=observation_count,
    )
    provenance: dict[str, Any] = {
        "schema_version": PROVENANCE_SCHEMA,
        "data_kind": "synthetic",
        "generated_utc": None,
        "fixture_id": "urbantwin-deterministic-eo-v1",
        "configuration": {
            "mode": "synthetic",
            "n": n,
            "seed": seed,
            "resolution_m": 10,
        },
        "crs": "EPSG:32636",
        "transform": [10.0, 0.0, 300000.0, 0.0, -10.0, 3330000.0],
        "width": n,
        "height": n,
        "band_order": list(BAND_ORDER),
        "band_names": list(BAND_NAMES),
        "array_keys": ["bands", "labels", "valid_mask", "observation_count"],
        "valid_pixel_count": int(np.count_nonzero(valid_mask)),
        "grid_pixel_count": int(n * n),
        "source_labels": {
            "imagery": "synthetic deterministic reflectance fixture",
            "reference_labels": "synthetic deterministic class fixture",
        },
        "warning": "SYNTHETIC TEST DATA — not an observation and not research evidence",
        "signed_asset_urls_recorded": False,
        "eo_stack_sha256": sha256_file(stack_path),
    }
    save_json(provenance, destination / "provenance.json")
    return provenance


__all__ = ["write_demo"]
