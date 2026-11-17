"""Shared configuration and provenance utilities."""
from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]

CANONICAL_CONFIG_HASH_SCHEME = "urbantwin.config-canonical-json/v1"
_CONFIG_HASH_SCHEME_FIELD = "config_hash_scheme"
_CONFIG_CANONICAL_HASH_FIELD = "config_canonical_sha256"
_CONFIG_RAW_HASH_FIELD = "config_sha256"


class ConfigurationHashError(ValueError):
    """Raised when a configuration cannot satisfy its declared hash contract."""


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """Safe YAML loader that refuses ambiguous duplicate mapping keys."""


def _construct_unique_mapping(
    loader: _UniqueKeySafeLoader,
    node: yaml.nodes.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in result
        except TypeError as exc:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found an unhashable mapping key",
                key_node.start_mark,
            ) from exc
        if duplicate:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _parse_yaml_mapping(text: str, source: str | Path) -> dict:
    result = yaml.load(text, Loader=_UniqueKeySafeLoader)
    if not isinstance(result, dict):
        raise TypeError(f"Expected YAML mapping: {source}")
    return result


def load_yaml(path: str | Path) -> dict:
    path = Path(path)
    return _parse_yaml_mapping(path.read_text(encoding="utf-8"), path)


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


def _require_canonical_config_value(value: Any, location: str = "configuration") -> None:
    """Require a type-stable, JSON-native configuration tree."""
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, int) and not isinstance(value, bool):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ConfigurationHashError(f"{location} contains a non-finite number")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _require_canonical_config_value(item, f"{location}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ConfigurationHashError(f"{location} contains a non-string mapping key")
            _require_canonical_config_value(item, f"{location}.{key}")
        return
    raise ConfigurationHashError(
        f"{location} contains unsupported value type {type(value).__name__}"
    )


def canonical_config_bytes(configuration: dict[str, Any]) -> bytes:
    """Serialize one parsed configuration under the versioned canonical contract."""
    if not isinstance(configuration, dict):
        raise ConfigurationHashError("Canonical configuration must be a mapping")
    _require_canonical_config_value(configuration)
    try:
        payload = json.dumps(
            configuration,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise ConfigurationHashError("Configuration cannot be serialized canonically") from exc
    return payload.encode("utf-8")


def canonical_config_sha256(configuration: dict[str, Any]) -> str:
    """Hash parsed configuration semantics, independent of YAML line endings."""
    payload = CANONICAL_CONFIG_HASH_SCHEME.encode("ascii") + b"\n"
    return hashlib.sha256(payload + canonical_config_bytes(configuration)).hexdigest()


def configuration_hash_record(
    path: str | Path,
    configuration: dict[str, Any],
) -> dict[str, str]:
    """Build the explicit hash fields used by future configuration-bound artifacts."""
    path = Path(path)
    try:
        raw_bytes = path.read_bytes()
        file_configuration = _parse_yaml_mapping(raw_bytes.decode("utf-8"), path)
    except (OSError, TypeError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ConfigurationHashError(f"Configuration cannot be parsed: {path}") from exc
    if canonical_config_sha256(file_configuration) != canonical_config_sha256(configuration):
        raise ConfigurationHashError(
            "Configuration snapshot does not match the configuration file"
        )
    return {
        _CONFIG_RAW_HASH_FIELD: hashlib.sha256(raw_bytes).hexdigest(),
        _CONFIG_HASH_SCHEME_FIELD: CANONICAL_CONFIG_HASH_SCHEME,
        _CONFIG_CANONICAL_HASH_FIELD: canonical_config_sha256(configuration),
    }


def validate_configuration_hash(
    path: str | Path,
    record: dict[str, Any],
    *,
    mode: str,
    embedded_configuration: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate either a legacy raw-byte or future canonical hash contract.

    Modes are deliberately selected by the enclosing artifact schema. A failed
    legacy raw-byte comparison never falls back to canonical equivalence.
    """
    if mode not in {"legacy_raw", "canonical_v1"}:
        raise ConfigurationHashError(f"Unsupported configuration hash mode: {mode}")
    if not isinstance(record, dict):
        raise ConfigurationHashError("Configuration hash record must be a mapping")
    path = Path(path)
    if not path.is_file():
        raise ConfigurationHashError(f"Configuration file is missing: {path}")
    expected_raw = record.get(_CONFIG_RAW_HASH_FIELD)
    if not _is_sha256(expected_raw):
        raise ConfigurationHashError("Configuration raw SHA-256 is missing or malformed")
    try:
        raw_bytes = path.read_bytes()
    except OSError as exc:
        raise ConfigurationHashError(f"Configuration cannot be read: {path}") from exc
    actual_raw = hashlib.sha256(raw_bytes).hexdigest()
    canonical_fields_present = any(
        field in record for field in (_CONFIG_HASH_SCHEME_FIELD, _CONFIG_CANONICAL_HASH_FIELD)
    )
    if mode == "legacy_raw":
        if canonical_fields_present:
            raise ConfigurationHashError(
                "Legacy configuration provenance must not contain canonical hash fields"
            )
        if actual_raw != expected_raw:
            raise ConfigurationHashError("Legacy configuration raw SHA-256 mismatch")
        return {"mode": mode, "raw_checkout_matches": True}

    if record.get(_CONFIG_HASH_SCHEME_FIELD) != CANONICAL_CONFIG_HASH_SCHEME:
        raise ConfigurationHashError("Unknown or missing canonical configuration hash scheme")
    expected_canonical = record.get(_CONFIG_CANONICAL_HASH_FIELD)
    if not _is_sha256(expected_canonical):
        raise ConfigurationHashError("Canonical configuration SHA-256 is missing or malformed")
    try:
        current_configuration = _parse_yaml_mapping(raw_bytes.decode("utf-8"), path)
    except (TypeError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ConfigurationHashError(f"Configuration cannot be parsed: {path}") from exc
    if canonical_config_sha256(current_configuration) != expected_canonical:
        raise ConfigurationHashError("Canonical configuration SHA-256 mismatch")
    if embedded_configuration is None:
        raise ConfigurationHashError("Canonical provenance has no embedded configuration")
    if canonical_config_sha256(embedded_configuration) != expected_canonical:
        raise ConfigurationHashError("Embedded configuration snapshot hash mismatch")
    return {"mode": mode, "raw_checkout_matches": actual_raw == expected_raw}


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
