"""Deterministic, local-only normalization of TfC observed traffic records.

This module never downloads source data and never fills a missing observation.
Its outputs are intended for the ignored ``data/observed`` workspace until the
publisher supplies an explicit redistribution license.
"""
from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
import os
import shutil
import tempfile
import zipfile
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, TextIO

import pandas as pd

from urbantwin.common import sha256_file
from urbantwin.data.observed_traffic import (
    COUNTS_COLUMNS,
    VEHICLE_COUNT_COLUMNS,
    ObservedTrafficValidationError,
    audit_observed_traffic,
)

SCHEMA_VERSION = "urbantwin.observed-traffic-study/v1"
TIMEZONE_STATUS = "unspecified_by_publisher"
SAMPLE_VALIDITY_STATUS = "not_provided_by_source"

TFC_SOURCE_RECORDS = {
    "counts": {
        "record_id": 190,
        "asset_id": 252,
        "title": "[Cairo] Traffic Counts",
        "record_url": "https://data.transportforcairo.com/catalogue/#/document/190",
        "download_url": "https://data.transportforcairo.com/documents/190/download",
    },
    "locations": {
        "record_id": 193,
        "asset_id": 255,
        "title": "[Cairo] Traffic Survey Locations",
        "record_url": "https://data.transportforcairo.com/catalogue/#/dataset/193",
        "download_url": (
            "https://data.transportforcairo.com/api/v2/assets/255/download"
        ),
    },
}

_WIDE_COLUMNS = (
    "observation_id",
    "source_row_number",
    "source_osm_id",
    "osm_id",
    "source_segment_id",
    "segment_id",
    "source_direction",
    "direction_code",
    "source_day",
    "day_normalized",
    "source_date",
    "observed_date",
    "source_hour_start",
    "source_hour_end",
    "interval_start_local",
    "interval_end_local",
    "interval_minutes",
    "timezone_status",
    "road_name",
    *VEHICLE_COUNT_COLUMNS,
    "observed_vehicle_total",
    "sample_validity_status",
)

_LONG_COLUMNS = (
    "observation_id",
    "source_row_number",
    "segment_id",
    "osm_id",
    "direction_code",
    "observed_date",
    "interval_start_local",
    "interval_end_local",
    "timezone_status",
    "road_name",
    "vehicle_class_source",
    "vehicle_count",
    "sample_validity_status",
)

_DIAGNOSTIC_COLUMNS = (
    "segment_id",
    "location_osm_id",
    "count_osm_ids",
    "direction_code",
    "count_rows",
    "observed_vehicle_total",
    "status",
)


def _require_utc_timestamp(value: str) -> str:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, ValueError) as exc:
        raise ValueError("source_retrieved_at_utc must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ValueError("source_retrieved_at_utc must include a UTC offset")
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


@contextmanager
def _open_single_member(
    path: Path, suffixes: set[str], description: str
) -> Iterator[TextIO]:
    if path.suffix.casefold() == ".zip":
        try:
            archive = zipfile.ZipFile(path)
        except zipfile.BadZipFile as exc:
            raise ObservedTrafficValidationError(f"Invalid ZIP archive: {path}") from exc
        with archive:
            members = [
                member
                for member in archive.infolist()
                if not member.is_dir()
                and Path(member.filename).suffix.casefold() in suffixes
            ]
            if len(members) != 1:
                raise ObservedTrafficValidationError(
                    f"{description} ZIP must contain exactly one supported file; "
                    f"found {len(members)}"
                )
            with (
                archive.open(members[0], "r") as binary,
                io.TextIOWrapper(binary, encoding="utf-8-sig", newline="") as text,
            ):
                yield text
        return
    if path.suffix.casefold() not in suffixes:
        choices = ", ".join(sorted(suffixes))
        raise ObservedTrafficValidationError(
            f"{description} input must use one of {choices}, or a single-file ZIP"
        )
    with path.open(encoding="utf-8-sig", newline="") as text:
        yield text


def _read_count_rows(path: Path) -> list[dict[str, str]]:
    with _open_single_member(path, {".csv"}, "Counts") as text:
        reader = csv.DictReader(text)
        if tuple(reader.fieldnames or ()) != COUNTS_COLUMNS:
            raise ObservedTrafficValidationError(
                "Traffic-count columns changed after validation; refusing normalization"
            )
        return [dict(row) for row in reader]


def _read_locations(path: Path) -> dict[str, Any]:
    with _open_single_member(path, {".geojson", ".json"}, "Locations") as text:
        try:
            payload = json.load(text)
        except json.JSONDecodeError as exc:
            raise ObservedTrafficValidationError(
                f"Invalid locations GeoJSON: {path}"
            ) from exc
    if not isinstance(payload, dict):
        raise ObservedTrafficValidationError("Locations GeoJSON must be an object")
    return payload


def _write_json(value: Any, path: Path) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_csv(rows: list[dict[str, Any]], columns: tuple[str, ...], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _observation_id(row: dict[str, str]) -> str:
    identity = "|".join(
        (
            row["segmentId"].strip(),
            row["direction"].strip(),
            row["Date"].strip(),
            row["hour_start"].strip(),
            row["hour_end"].strip(),
        )
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]


def _normalize_counts(
    rows: list[dict[str, str]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    wide: list[dict[str, Any]] = []
    long: list[dict[str, Any]] = []
    segment_stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "osm_ids": set(),
            "directions": set(),
            "rows": 0,
            "vehicle_total": 0,
        }
    )
    for source_row_number, source in enumerate(rows, start=2):
        segment_id = source["segmentId"].strip()
        osm_id = source["osm_id"].strip()
        direction = source["direction"].strip().casefold()
        observed = datetime.strptime(source["Date"].strip(), "%d-%m-%y").date()
        day_normalized = observed.strftime("%A")
        count_values = {
            vehicle_class: int(source[vehicle_class].strip())
            for vehicle_class in VEHICLE_COUNT_COLUMNS
        }
        vehicle_total = sum(count_values.values())
        observation_id = _observation_id(source)
        normalized = {
            "observation_id": observation_id,
            "source_row_number": source_row_number,
            "source_osm_id": source["osm_id"],
            "osm_id": osm_id,
            "source_segment_id": source["segmentId"],
            "segment_id": segment_id,
            "source_direction": source["direction"],
            "direction_code": direction,
            "source_day": source["Day"],
            "day_normalized": day_normalized,
            "source_date": source["Date"],
            "observed_date": observed.isoformat(),
            "source_hour_start": source["hour_start"],
            "source_hour_end": source["hour_end"],
            "interval_start_local": source["hour_start"].strip(),
            "interval_end_local": source["hour_end"].strip(),
            "interval_minutes": 20,
            "timezone_status": TIMEZONE_STATUS,
            "road_name": source["road_name"],
            **count_values,
            "observed_vehicle_total": vehicle_total,
            "sample_validity_status": SAMPLE_VALIDITY_STATUS,
        }
        wide.append(normalized)
        for vehicle_class in VEHICLE_COUNT_COLUMNS:
            long.append(
                {
                    "observation_id": observation_id,
                    "source_row_number": source_row_number,
                    "segment_id": segment_id,
                    "osm_id": osm_id,
                    "direction_code": direction,
                    "observed_date": observed.isoformat(),
                    "interval_start_local": source["hour_start"].strip(),
                    "interval_end_local": source["hour_end"].strip(),
                    "timezone_status": TIMEZONE_STATUS,
                    "road_name": source["road_name"],
                    "vehicle_class_source": vehicle_class,
                    "vehicle_count": count_values[vehicle_class],
                    "sample_validity_status": SAMPLE_VALIDITY_STATUS,
                }
            )
        stats = segment_stats[segment_id]
        stats["osm_ids"].add(osm_id)
        stats["directions"].add(direction)
        stats["rows"] += 1
        stats["vehicle_total"] += vehicle_total
    return wide, long, segment_stats


def _normalize_locations(
    payload: dict[str, Any], segment_stats: dict[str, dict[str, Any]]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    output = copy.deepcopy(payload)
    diagnostics: list[dict[str, Any]] = []
    for feature in output["features"]:
        properties = feature["properties"]
        segment_key = "segmentId" if "segmentId" in properties else "segmentid"
        source_segment_id = properties[segment_key]
        segment_id = str(source_segment_id).strip()
        location_osm_id = str(properties["osm_id"]).strip()
        direction = str(properties["direction"]).strip().casefold()
        stats = segment_stats.get(segment_id)
        if stats is None:
            status = "location_without_count_rows"
            count_osm_ids: list[str] = []
            count_rows = 0
            vehicle_total = 0
        else:
            count_osm_ids = sorted(stats["osm_ids"])
            count_rows = int(stats["rows"])
            vehicle_total = int(stats["vehicle_total"])
            status = (
                "matched"
                if count_osm_ids == [location_osm_id]
                else "osm_id_mismatch"
            )
        properties["segment_id_normalized"] = segment_id
        properties["direction_code_normalized"] = direction
        properties["count_join_status"] = status
        properties["count_rows"] = count_rows
        properties["observed_vehicle_total"] = vehicle_total
        diagnostics.append(
            {
                "segment_id": segment_id,
                "location_osm_id": location_osm_id,
                "count_osm_ids": "|".join(count_osm_ids),
                "direction_code": direction,
                "count_rows": count_rows,
                "observed_vehicle_total": vehicle_total,
                "status": status,
            }
        )
    diagnostics.sort(key=lambda item: (item["segment_id"], item["location_osm_id"]))
    return output, diagnostics


def _sanitized_validation(
    validation: dict[str, Any], counts_path: Path, locations_path: Path
) -> dict[str, Any]:
    result = copy.deepcopy(validation)
    result["inputs"]["counts"]["path"] = counts_path.name
    result["inputs"]["locations"]["path"] = locations_path.name
    return result


def build_observed_study(
    counts_path: str | Path,
    locations_path: str | Path,
    output_dir: str | Path,
    *,
    source_retrieved_at_utc: str,
    license_status: str = "unspecified_by_publisher",
    license_identifier: str | None = None,
    redistribution_permitted: bool = False,
    expected_count_rows: int | None = None,
    expected_segments: int | None = None,
) -> dict[str, Any]:
    """Validate and atomically write a normalized local observed-data bundle.

    The destination must not exist.  This fail-closed rule prevents a later run
    from overwriting a previously inspected research input.  A timezone and
    sample-validity flag are deliberately *not* inferred because neither field
    is supplied by the published records.
    """
    counts_path = Path(counts_path)
    locations_path = Path(locations_path)
    output_dir = Path(output_dir)
    retrieved_at = _require_utc_timestamp(source_retrieved_at_utc)
    if not license_status.strip():
        raise ValueError("license_status must be explicit")
    if redistribution_permitted and not (license_identifier or "").strip():
        raise ValueError(
            "A license_identifier is required before marking redistribution permitted"
        )
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite observed-data output directory: {output_dir}"
        )

    validation = audit_observed_traffic(
        counts_path,
        locations_path,
        expected_count_rows=expected_count_rows,
        expected_segments=expected_segments,
    )
    count_rows = _read_count_rows(counts_path)
    location_payload = _read_locations(locations_path)
    wide, long, segment_stats = _normalize_counts(count_rows)
    normalized_locations, diagnostics = _normalize_locations(
        location_payload, segment_stats
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        paths = {
            "counts_wide": staging / "counts_wide.csv",
            "counts_long": staging / "counts_long.csv",
            "survey_segments": staging / "survey_segments.geojson",
            "join_diagnostics": staging / "join_diagnostics.csv",
            "validation": staging / "validation.json",
        }
        _write_csv(wide, _WIDE_COLUMNS, paths["counts_wide"])
        _write_csv(long, _LONG_COLUMNS, paths["counts_long"])
        _write_json(normalized_locations, paths["survey_segments"])
        _write_csv(diagnostics, _DIAGNOSTIC_COLUMNS, paths["join_diagnostics"])
        sanitized_validation = _sanitized_validation(
            validation, counts_path, locations_path
        )
        _write_json(sanitized_validation, paths["validation"])

        artifacts = {
            name: {
                "filename": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for name, path in paths.items()
        }
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "data_kind": "real_observed_traffic",
            "provider": "Transport for Cairo (TfC)",
            "attribution": "Transport for Cairo (TfC)",
            "source_retrieved_at_utc": retrieved_at,
            "sources": {
                role: {
                    **TFC_SOURCE_RECORDS[role],
                    "filename": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for role, path in (
                    ("counts", counts_path),
                    ("locations", locations_path),
                )
            },
            "license": {
                "status": license_status,
                "identifier": license_identifier,
                "redistribution_permitted": redistribution_permitted,
                "note": (
                    "The public portal did not state an explicit reuse license at audit "
                    "time. Keep source and row-level derived records outside Git until "
                    "written terms permit redistribution."
                ),
            },
            "normalization": {
                "primary_join_field": "trimmed segmentId/segmentid",
                "cross_checks": ["direction", "osm_id"],
                "source_values_preserved": True,
                "missing_value_policy": "preserve; never impute",
                "timezone_status": TIMEZONE_STATUS,
                "direction_legend_status": "codes_0_and_1_not_semantically_defined",
                "sample_validity_status": SAMPLE_VALIDITY_STATUS,
                "vehicle_classes": list(VEHICLE_COUNT_COLUMNS),
                "interval_minutes": 20,
            },
            "row_counts": {
                "counts_wide": len(wide),
                "counts_long": len(long),
                "survey_segments": len(normalized_locations["features"]),
                "join_diagnostics": len(diagnostics),
            },
            "validation_status": validation["status"],
            "validation_warnings": validation["warnings"],
            "artifacts": artifacts,
            "interpretation_limits": [
                "These are directional link counts, not intersection turning movements.",
                (
                    "Timezone, sample-validity flags, and the semantic mapping of direction "
                    "codes 0/1 are not supplied by the publisher."
                ),
                (
                    "The supplied survey geometry is OSM-derived context, not a measured "
                    "signal or controller inventory."
                ),
            ],
        }
        _write_json(manifest, staging / "source_manifest.json")
        os.replace(staging, output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def prepare_observed_data(
    counts_path: str | Path,
    locations_path: str | Path,
    output_dir: str | Path,
    *,
    source_retrieved_at_utc: str,
    license_status: str = "unspecified_by_publisher",
    license_identifier: str | None = None,
    redistribution_permitted: bool = False,
    expected_count_rows: int | None = None,
    expected_segments: int | None = None,
) -> dict[str, Any]:
    """Prepare a bundle and return stable paths for downstream local workflows."""
    root = Path(output_dir)
    manifest = build_observed_study(
        counts_path,
        locations_path,
        root,
        source_retrieved_at_utc=source_retrieved_at_utc,
        license_status=license_status,
        license_identifier=license_identifier,
        redistribution_permitted=redistribution_permitted,
        expected_count_rows=expected_count_rows,
        expected_segments=expected_segments,
    )
    return {
        "root": root,
        "manifest": manifest,
        "manifest_path": root / "source_manifest.json",
        "counts_wide_path": root / "counts_wide.csv",
        "counts_long_path": root / "counts_long.csv",
        "survey_segments_path": root / "survey_segments.geojson",
        "join_diagnostics_path": root / "join_diagnostics.csv",
        "validation_path": root / "validation.json",
    }


def _validated_artifact_path(
    root: Path, manifest: dict[str, Any], artifact_name: str
) -> Path:
    artifact = (manifest.get("artifacts") or {}).get(artifact_name)
    if not isinstance(artifact, dict):
        raise ObservedTrafficValidationError(
            f"Observed-study manifest has no {artifact_name!r} artifact record"
        )
    filename = artifact.get("filename")
    expected_hash = artifact.get("sha256")
    if not isinstance(filename, str) or not filename:
        raise ObservedTrafficValidationError(
            f"Observed-study {artifact_name!r} filename is invalid"
        )
    path = (root / filename).resolve()
    if path.parent != root.resolve():
        raise ObservedTrafficValidationError(
            f"Observed-study {artifact_name!r} path escapes its bundle"
        )
    if not path.is_file():
        raise FileNotFoundError(f"Observed-study artifact is missing: {path}")
    if not isinstance(expected_hash, str) or sha256_file(path) != expected_hash:
        raise ObservedTrafficValidationError(
            f"Observed-study artifact hash mismatch: {path.name}"
        )
    return path


def load_normalized_counts(study_dir: str | Path) -> pd.DataFrame:
    """Load hash-verified normalized counts with numeric invariants rechecked."""
    root = Path(study_dir)
    manifest_path = root / "source_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Observed-study manifest is missing: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ObservedTrafficValidationError(
            f"Observed-study manifest is not valid JSON: {manifest_path}"
        ) from exc
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ObservedTrafficValidationError("Observed-study schema version is incompatible")
    if manifest.get("data_kind") != "real_observed_traffic":
        raise ObservedTrafficValidationError("Observed-study data_kind is not real observed traffic")
    counts_path = _validated_artifact_path(root, manifest, "counts_wide")
    frame = pd.read_csv(counts_path, dtype=str, keep_default_na=False)
    if tuple(frame.columns) != _WIDE_COLUMNS:
        raise ObservedTrafficValidationError("Normalized count columns are incompatible")
    expected_rows = (manifest.get("row_counts") or {}).get("counts_wide")
    if not isinstance(expected_rows, int) or len(frame) != expected_rows:
        raise ObservedTrafficValidationError(
            "Normalized count row count disagrees with the source manifest"
        )
    if frame["observation_id"].duplicated().any():
        raise ObservedTrafficValidationError("Normalized counts contain duplicate observations")
    numeric_columns = [
        "source_row_number",
        "interval_minutes",
        *VEHICLE_COUNT_COLUMNS,
        "observed_vehicle_total",
    ]
    try:
        for column in numeric_columns:
            frame[column] = pd.to_numeric(frame[column], errors="raise").astype("int64")
    except (TypeError, ValueError) as exc:
        raise ObservedTrafficValidationError(
            "Normalized counts contain non-integer count metadata"
        ) from exc
    if (frame[list(VEHICLE_COUNT_COLUMNS)] < 0).any(axis=None):
        raise ObservedTrafficValidationError("Normalized counts contain negative vehicles")
    recomputed = frame[list(VEHICLE_COUNT_COLUMNS)].sum(axis=1)
    if not recomputed.equals(frame["observed_vehicle_total"]):
        raise ObservedTrafficValidationError(
            "Normalized observed_vehicle_total is inconsistent with vehicle classes"
        )
    if set(frame["timezone_status"]) != {TIMEZONE_STATUS}:
        raise ObservedTrafficValidationError("Normalized counts infer an unsupported timezone")
    if set(frame["sample_validity_status"]) != {SAMPLE_VALIDITY_STATUS}:
        raise ObservedTrafficValidationError(
            "Normalized counts contain an unsupported sample-validity claim"
        )
    return frame


__all__ = [
    "SAMPLE_VALIDITY_STATUS",
    "SCHEMA_VERSION",
    "TFC_SOURCE_RECORDS",
    "TIMEZONE_STATUS",
    "build_observed_study",
    "load_normalized_counts",
    "prepare_observed_data",
]
