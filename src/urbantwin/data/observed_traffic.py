"""Offline validation for the published 2024 Greater Cairo traffic counts.

The validator deliberately accepts local files only.  It does not download data,
write research artifacts, infer turning movements, or substitute generated data.
"""
from __future__ import annotations

import csv
import io
import json
import math
import re
import uuid
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

from urbantwin.common import sha256_file

COUNTS_COLUMNS = (
    "osm_id",
    "Day",
    "Date",
    "road_name",
    "direction",
    "hour_start",
    "hour_end",
    "motorcycle",
    "car",
    "taxi",
    "pickup",
    "2AxTruck",
    "3AxTruck",
    ">3AxTruck",
    "other",
    "mini_microbus",
    "publicbus",
    "privatebus",
    "tuktuk",
    "segmentId",
)

VEHICLE_COUNT_COLUMNS = COUNTS_COLUMNS[7:-1]

LOCATION_PROPERTIES = (
    "osm_id",
    "segmentId",
    "direction",
    "road_name",
    "highway",
    "oneway",
    "survey_line_name",
    "osm_extract_date",
)

_INTEGER = re.compile(r"[+-]?\d+")
_SUPPORTED_DIRECTIONS = {"0", "1", "inbound", "outbound"}


class ObservedTrafficValidationError(ValueError):
    """Raised when observed traffic inputs violate the published data contract."""


@contextmanager
def _open_csv(path: Path) -> Iterator[TextIO]:
    if path.suffix.casefold() == ".zip":
        try:
            archive = zipfile.ZipFile(path)
        except zipfile.BadZipFile as exc:
            raise ObservedTrafficValidationError(f"Invalid ZIP archive: {path}") from exc
        with archive:
            members = [
                member
                for member in archive.infolist()
                if not member.is_dir() and Path(member.filename).suffix.casefold() == ".csv"
            ]
            if len(members) != 1:
                raise ObservedTrafficValidationError(
                    f"Counts ZIP must contain exactly one CSV file; found {len(members)}"
                )
            with (
                archive.open(members[0], "r") as binary,
                io.TextIOWrapper(binary, encoding="utf-8-sig", newline="") as text,
            ):
                yield text
        return
    if path.suffix.casefold() != ".csv":
        raise ObservedTrafficValidationError("Counts input must be a .csv or single-CSV .zip")
    with path.open(encoding="utf-8-sig", newline="") as text:
        yield text


def _required_text(value: Any, label: str) -> str:
    text = "" if value is None else str(value).strip()
    if not text:
        raise ObservedTrafficValidationError(f"Missing required value: {label}")
    return text


def _integer(value: Any, label: str, *, positive: bool = False) -> int:
    text = _required_text(value, label)
    if not _INTEGER.fullmatch(text):
        raise ObservedTrafficValidationError(f"{label} must be an integer, found {text!r}")
    result = int(text)
    minimum = 1 if positive else 0
    if result < minimum:
        qualifier = "positive" if positive else "non-negative"
        raise ObservedTrafficValidationError(f"{label} must be {qualifier}, found {result}")
    return result


def _segment_id(value: Any, label: str) -> str:
    text = _required_text(value, label)
    try:
        return str(uuid.UUID(text))
    except ValueError as exc:
        raise ObservedTrafficValidationError(f"{label} is not a UUID: {text!r}") from exc


def _direction(value: Any, label: str) -> str:
    direction = _required_text(value, label).casefold()
    if direction not in _SUPPORTED_DIRECTIONS:
        raise ObservedTrafficValidationError(
            f"{label} has unsupported direction {direction!r}; expected 0/1 or Inbound/Outbound"
        )
    return direction


def _clock_minutes(value: Any, label: str) -> int:
    text = _required_text(value, label)
    try:
        hour_text, minute_text = text.split(":", maxsplit=1)
        hour, minute = int(hour_text), int(minute_text)
    except (TypeError, ValueError) as exc:
        raise ObservedTrafficValidationError(f"{label} is not an H:MM time: {text!r}") from exc
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ObservedTrafficValidationError(f"{label} is outside a 24-hour clock: {text!r}")
    return hour * 60 + minute


def _input_record(path: Path) -> dict[str, str | int]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _read_counts(path: Path) -> tuple[dict[str, Any], dict[tuple[str, str], set[str]]]:
    missing = {column: 0 for column in COUNTS_COLUMNS}
    directions: set[str] = set()
    segment_ids: set[str] = set()
    osm_ids: set[str] = set()
    dates = []
    interval_keys: set[tuple[str, str, str, str, str]] = set()
    duplicate_intervals = 0
    total_vehicles = 0
    pairs: dict[tuple[str, str], set[str]] = {}
    row_count = 0

    with _open_csv(path) as text:
        reader = csv.DictReader(text)
        actual_columns = tuple(reader.fieldnames or ())
        if actual_columns != COUNTS_COLUMNS:
            raise ObservedTrafficValidationError(
                "Traffic-count columns do not match the published schema. "
                f"Expected {list(COUNTS_COLUMNS)!r}; found {list(actual_columns)!r}"
            )
        for row_number, row in enumerate(reader, start=2):
            row_count += 1
            for column in COUNTS_COLUMNS:
                if row.get(column) is None or not str(row[column]).strip():
                    missing[column] += 1

            osm_id = str(_integer(row["osm_id"], f"row {row_number} osm_id", positive=True))
            segment_id = _segment_id(row["segmentId"], f"row {row_number} segmentId")
            direction = _direction(row["direction"], f"row {row_number} direction")

            date_text = _required_text(row["Date"], f"row {row_number} Date")
            try:
                observed_date = datetime.strptime(date_text, "%d-%m-%y").date()
            except ValueError as exc:
                raise ObservedTrafficValidationError(
                    f"row {row_number} Date must use DD-MM-YY: {date_text!r}"
                ) from exc
            day = _required_text(row["Day"], f"row {row_number} Day")
            if day.casefold() != observed_date.strftime("%A").casefold():
                raise ObservedTrafficValidationError(
                    f"row {row_number} Day {day!r} does not match {date_text}"
                )

            start_text = _required_text(row["hour_start"], f"row {row_number} hour_start")
            end_text = _required_text(row["hour_end"], f"row {row_number} hour_end")
            start = _clock_minutes(start_text, f"row {row_number} hour_start")
            end = _clock_minutes(end_text, f"row {row_number} hour_end")
            if (end - start) % (24 * 60) != 20:
                raise ObservedTrafficValidationError(
                    f"row {row_number} is not a 20-minute interval: {start_text}-{end_text}"
                )

            for column in VEHICLE_COUNT_COLUMNS:
                total_vehicles += _integer(row[column], f"row {row_number} {column}")

            interval_key = (segment_id, direction, date_text, start_text, end_text)
            if interval_key in interval_keys:
                duplicate_intervals += 1
            interval_keys.add(interval_key)
            pair = (segment_id, osm_id)
            pairs.setdefault(pair, set()).add(direction)
            directions.add(direction)
            segment_ids.add(segment_id)
            osm_ids.add(osm_id)
            dates.append(observed_date)

    if row_count == 0:
        raise ObservedTrafficValidationError("Traffic-count CSV contains no records")
    if duplicate_intervals:
        raise ObservedTrafficValidationError(
            f"Traffic counts contain {duplicate_intervals} duplicate directional interval records"
        )
    return (
        {
            "rows": row_count,
            "columns": list(COUNTS_COLUMNS),
            "missing_values": missing,
            "date_start": min(dates).isoformat(),
            "date_end": max(dates).isoformat(),
            "interval_minutes": 20,
            "direction_values": sorted(directions),
            "segment_ids": len(segment_ids),
            "osm_ids": len(osm_ids),
            "duplicate_intervals": duplicate_intervals,
            "vehicle_count_total": total_vehicles,
        },
        pairs,
    )


def _validate_crs(payload: dict[str, Any]) -> str:
    crs = payload.get("crs")
    if crs is None:
        return "RFC 7946 WGS84 (CRS member absent)"
    if not isinstance(crs, dict):
        raise ObservedTrafficValidationError("GeoJSON CRS must be an object when present")
    name = str((crs.get("properties") or {}).get("name", ""))
    normalized = name.casefold().replace(" ", "")
    if "4326" not in normalized and "crs84" not in normalized:
        raise ObservedTrafficValidationError(
            f"Survey locations must be WGS84 longitude/latitude, found CRS {name!r}"
        )
    return name


def _position(value: Any, label: str) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) < 2:
        raise ObservedTrafficValidationError(f"{label} is not a GeoJSON position")
    try:
        longitude, latitude = float(value[0]), float(value[1])
    except (TypeError, ValueError) as exc:
        raise ObservedTrafficValidationError(f"{label} has non-numeric coordinates") from exc
    if not math.isfinite(longitude) or not math.isfinite(latitude):
        raise ObservedTrafficValidationError(f"{label} has non-finite coordinates")
    if not (-180 <= longitude <= 180 and -90 <= latitude <= 90):
        raise ObservedTrafficValidationError(
            f"{label} is outside WGS84 longitude/latitude bounds: {longitude}, {latitude}"
        )
    return longitude, latitude


def _validate_geometry(geometry: Any, label: str) -> str:
    if not isinstance(geometry, dict):
        raise ObservedTrafficValidationError(f"{label} geometry is missing")
    geometry_type = geometry.get("type")
    coordinates = geometry.get("coordinates")
    if geometry_type == "LineString":
        lines = [coordinates]
    elif geometry_type == "MultiLineString":
        lines = coordinates
    else:
        raise ObservedTrafficValidationError(
            f"{label} must be LineString or MultiLineString, found {geometry_type!r}"
        )
    if not isinstance(lines, list) or not lines:
        raise ObservedTrafficValidationError(f"{label} has empty line geometry")
    for line_number, line in enumerate(lines, start=1):
        if not isinstance(line, list) or len(line) < 2:
            raise ObservedTrafficValidationError(
                f"{label} line {line_number} must contain at least two positions"
            )
        for position_number, value in enumerate(line, start=1):
            _position(value, f"{label} line {line_number} position {position_number}")
    return str(geometry_type)


def _read_locations(
    path: Path,
) -> tuple[dict[str, Any], dict[tuple[str, str], set[str]]]:
    try:
        if path.suffix.casefold() == ".zip":
            with zipfile.ZipFile(path) as archive:
                members = [
                    member
                    for member in archive.infolist()
                    if not member.is_dir()
                    and Path(member.filename).suffix.casefold() in {".geojson", ".json"}
                ]
                if len(members) != 1:
                    raise ObservedTrafficValidationError(
                        "Locations ZIP must contain exactly one GeoJSON file; "
                        f"found {len(members)}"
                    )
                payload = json.loads(archive.read(members[0]).decode("utf-8-sig"))
        elif path.suffix.casefold() in {".geojson", ".json"}:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
        else:
            raise ObservedTrafficValidationError(
                "Locations input must be a .geojson, .json, or single-GeoJSON .zip"
            )
    except zipfile.BadZipFile as exc:
        raise ObservedTrafficValidationError(f"Invalid ZIP archive: {path}") from exc
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ObservedTrafficValidationError(f"Invalid locations GeoJSON: {path}") from exc
    if not isinstance(payload, dict) or payload.get("type") != "FeatureCollection":
        raise ObservedTrafficValidationError("Locations input must be a GeoJSON FeatureCollection")
    crs = _validate_crs(payload)
    features = payload.get("features")
    if not isinstance(features, list) or not features:
        raise ObservedTrafficValidationError("Locations GeoJSON contains no features")

    missing = {column: 0 for column in LOCATION_PROPERTIES}
    geometry_types: set[str] = set()
    survey_locations: set[str] = set()
    segment_ids: set[str] = set()
    osm_ids: set[str] = set()
    extract_dates = []
    pairs: dict[tuple[str, str], set[str]] = {}

    for feature_number, feature in enumerate(features, start=1):
        if not isinstance(feature, dict) or feature.get("type") != "Feature":
            raise ObservedTrafficValidationError(f"locations feature {feature_number} is invalid")
        properties = feature.get("properties")
        if not isinstance(properties, dict):
            raise ObservedTrafficValidationError(
                f"locations feature {feature_number} has no properties object"
            )
        segment_key = "segmentId" if "segmentId" in properties else "segmentid"
        absent = [
            column
            for column in LOCATION_PROPERTIES
            if column != "segmentId" and column not in properties
        ]
        if segment_key not in properties:
            absent.append("segmentId/segmentid")
        if absent:
            raise ObservedTrafficValidationError(
                f"locations feature {feature_number} is missing properties: {absent}"
            )
        for column in LOCATION_PROPERTIES:
            value = properties[segment_key] if column == "segmentId" else properties[column]
            if value is None or not str(value).strip():
                missing[column] += 1

        if "fid" in properties and properties["fid"] is not None:
            _integer(properties["fid"], f"locations feature {feature_number} fid")
        osm_id = str(
            _integer(
                properties["osm_id"],
                f"locations feature {feature_number} osm_id",
                positive=True,
            )
        )
        segment_id = _segment_id(
            properties[segment_key], f"locations feature {feature_number} {segment_key}"
        )
        direction = _direction(
            properties["direction"], f"locations feature {feature_number} direction"
        )
        survey_line = str(properties["survey_line_name"] or "").strip()
        extract_date_text = _required_text(
            properties["osm_extract_date"],
            f"locations feature {feature_number} osm_extract_date",
        )
        try:
            extract_date = datetime.strptime(extract_date_text.removesuffix("Z"), "%Y-%m-%d").date()
        except ValueError as exc:
            raise ObservedTrafficValidationError(
                f"locations feature {feature_number} osm_extract_date must use YYYY-MM-DD"
            ) from exc
        geometry_types.add(
            _validate_geometry(feature.get("geometry"), f"locations feature {feature_number}")
        )

        pair = (segment_id, osm_id)
        if pair in pairs:
            raise ObservedTrafficValidationError(
                f"Locations contain duplicate segment/osm key: {segment_id}, {osm_id}"
            )
        pairs[pair] = {direction}
        if survey_line:
            survey_locations.add(survey_line)
        segment_ids.add(segment_id)
        osm_ids.add(osm_id)
        extract_dates.append(extract_date)

    return (
        {
            "features": len(features),
            "required_properties": list(LOCATION_PROPERTIES),
            "missing_values": missing,
            "crs": crs,
            "geometry_types": sorted(geometry_types),
            "survey_line_names": len(survey_locations),
            "survey_location_count": None,
            "survey_location_count_note": (
                "The release has no physical survey-location identifier; the portal's stated "
                "60-location count cannot be reconstructed from survey_line_name alone."
            ),
            "segment_ids": len(segment_ids),
            "osm_ids": len(osm_ids),
            "osm_extract_date_start": min(extract_dates).isoformat(),
            "osm_extract_date_end": max(extract_dates).isoformat(),
        },
        pairs,
    )


def audit_observed_traffic(
    counts_path: str | Path,
    locations_path: str | Path,
    *,
    expected_count_rows: int | None = None,
    expected_segments: int | None = None,
) -> dict[str, Any]:
    """Validate local observed-count and survey-location files without side effects."""
    counts_path = Path(counts_path)
    locations_path = Path(locations_path)
    for label, path in (("counts", counts_path), ("locations", locations_path)):
        if not path.is_file():
            raise FileNotFoundError(f"{label} input is not a file: {path}")

    counts, count_pairs = _read_counts(counts_path)
    locations, location_pairs = _read_locations(locations_path)

    if expected_count_rows is not None and counts["rows"] != expected_count_rows:
        raise ObservedTrafficValidationError(
            f"Expected {expected_count_rows} traffic-count rows, found {counts['rows']}"
        )
    if expected_segments is not None and locations["features"] != expected_segments:
        raise ObservedTrafficValidationError(
            f"Expected {expected_segments} directional segments, found {locations['features']}"
        )
    count_segments: dict[str, dict[str, set[str]]] = {}
    for (segment_id, osm_id), directions in count_pairs.items():
        values = count_segments.setdefault(segment_id, {"osm_ids": set(), "directions": set()})
        values["osm_ids"].add(osm_id)
        values["directions"].update(directions)
    location_segments: dict[str, dict[str, set[str]]] = {}
    for (segment_id, osm_id), directions in location_pairs.items():
        values = location_segments.setdefault(
            segment_id, {"osm_ids": set(), "directions": set()}
        )
        values["osm_ids"].add(osm_id)
        values["directions"].update(directions)

    unmatched_count_segments = sorted(count_segments.keys() - location_segments.keys())
    if unmatched_count_segments:
        raise ObservedTrafficValidationError(
            "Traffic counts contain segmentId values absent from survey locations; "
            f"first unmatched segment is {unmatched_count_segments[0]}"
        )
    direction_mismatches = sorted(
        segment_id
        for segment_id in count_segments.keys() & location_segments.keys()
        if count_segments[segment_id]["directions"]
        != location_segments[segment_id]["directions"]
    )
    if direction_mismatches:
        segment_id = direction_mismatches[0]
        raise ObservedTrafficValidationError(
            f"Traffic-count and location directions disagree for {segment_id}: "
            f"{sorted(count_segments[segment_id]['directions'])} vs "
            f"{sorted(location_segments[segment_id]['directions'])}"
        )
    osm_id_mismatches = [
        {
            "segment_id": segment_id,
            "counts_osm_ids": sorted(count_segments[segment_id]["osm_ids"]),
            "locations_osm_ids": sorted(location_segments[segment_id]["osm_ids"]),
        }
        for segment_id in sorted(count_segments.keys() & location_segments.keys())
        if count_segments[segment_id]["osm_ids"] != location_segments[segment_id]["osm_ids"]
    ]
    unused_location_segments = location_segments.keys() - count_segments.keys()
    warnings = []
    if unused_location_segments:
        warnings.append(
            f"{len(unused_location_segments)} survey-location segments have no count rows."
        )
    if osm_id_mismatches:
        warnings.append(
            f"{len(osm_id_mismatches)} matched segment IDs have conflicting OSM IDs."
        )
    count_missing = {
        column: count for column, count in counts["missing_values"].items() if count
    }
    if count_missing:
        warnings.append(f"Traffic-count missing values: {count_missing}.")
    location_missing = {
        column: count for column, count in locations["missing_values"].items() if count
    }
    if location_missing:
        warnings.append(f"Survey-location missing values: {location_missing}.")

    return {
        "schema_version": "urbantwin.observed-traffic-audit/v1",
        "data_kind": "real_observed_traffic",
        "status": "validated_with_warnings" if warnings else "validated",
        "warnings": warnings,
        "inputs": {
            "counts": _input_record(counts_path),
            "locations": _input_record(locations_path),
        },
        "counts": counts,
        "locations": locations,
        "association": {
            "primary_join_field": "segmentId/segmentid",
            "cross_check_fields": ["osm_id", "direction"],
            "matched_directional_segments": len(count_segments),
            "unmatched_count_segments": 0,
            "unused_location_segments": len(unused_location_segments),
            "direction_mismatches": 0,
            "osm_id_mismatch_count": len(osm_id_mismatches),
            "osm_id_mismatches": osm_id_mismatches,
            "status": (
                "validated_with_warnings"
                if osm_id_mismatches or unused_location_segments
                else "validated"
            ),
        },
        "interpretation_limits": [
            "Directional link counts are not complete intersection turning movements.",
            "No signal timing or phase observations are inferred by this audit.",
        ],
    }
