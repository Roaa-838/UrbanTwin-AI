from __future__ import annotations

import csv
import json
import uuid
import zipfile
from pathlib import Path

import pytest

from urbantwin.data.observed_study import (
    build_observed_study,
    load_normalized_counts,
    prepare_observed_data,
)
from urbantwin.data.observed_traffic import (
    COUNTS_COLUMNS,
    VEHICLE_COUNT_COLUMNS,
    ObservedTrafficValidationError,
)


def _synthetic_sources(
    root: Path, *, location_osm_id: int = 123, negative_count: bool = False
) -> tuple[Path, Path]:
    root.mkdir(parents=True)
    segment_id = str(uuid.UUID("12345678-1234-5678-1234-567812345678"))
    rows = []
    for index, (start, end) in enumerate((("08:00", "08:20"), ("08:20", "08:40"))):
        row = {column: "0" for column in COUNTS_COLUMNS}
        row.update(
            {
                "osm_id": "123",
                "Day": "Sunday " if index == 0 else "Sunday",
                "Date": "21-07-24",
                "road_name": "" if index == 0 else "SYNTHETIC TEST ROAD",
                "direction": "0",
                "hour_start": start,
                "hour_end": end,
                "car": "-1" if negative_count and index == 0 else str(10 + index),
                "taxi": "2",
                "segmentId": segment_id,
            }
        )
        rows.append(row)
    counts_csv = root / "synthetic_counts.csv"
    with counts_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COUNTS_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    counts_zip = root / "synthetic_counts.zip"
    with zipfile.ZipFile(counts_zip, "w") as archive:
        archive.write(counts_csv, arcname="synthetic_counts.csv")

    locations = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "OGC:CRS84"}},
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "osm_id": location_osm_id,
                    "segmentId": segment_id,
                    "direction": "0",
                    "road_name": "SYNTHETIC TEST ROAD",
                    "highway": "primary",
                    "oneway": "yes",
                    "survey_line_name": "SYNTHETIC TEST LINE",
                    "osm_extract_date": "2026-07-15",
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[31.23, 30.04], [31.231, 30.041]],
                },
            }
        ],
    }
    locations_path = root / "synthetic_locations.geojson"
    locations_path.write_text(json.dumps(locations), encoding="utf-8")
    return counts_zip, locations_path


def _build(sources: tuple[Path, Path], output: Path) -> dict:
    return build_observed_study(
        *sources,
        output,
        source_retrieved_at_utc="2026-11-08T10:00:00Z",
        expected_count_rows=2,
        expected_segments=1,
    )


def test_build_preserves_missing_values_and_writes_tidy_counts(tmp_path: Path) -> None:
    manifest = _build(_synthetic_sources(tmp_path / "source"), tmp_path / "study")

    with (tmp_path / "study" / "counts_wide.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        wide = list(csv.DictReader(stream))
    with (tmp_path / "study" / "counts_long.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        long = list(csv.DictReader(stream))

    assert len(wide) == 2
    assert wide[0]["road_name"] == ""
    assert wide[0]["source_day"] == "Sunday "
    assert wide[0]["day_normalized"] == "Sunday"
    assert wide[0]["observed_date"] == "2024-07-21"
    assert wide[0]["timezone_status"] == "unspecified_by_publisher"
    assert int(wide[0]["observed_vehicle_total"]) == 12
    assert len(long) == 2 * len(VEHICLE_COUNT_COLUMNS)
    assert {
        row["vehicle_class_source"] for row in long
    } == set(VEHICLE_COUNT_COLUMNS)
    assert manifest["normalization"]["missing_value_policy"] == "preserve; never impute"
    assert manifest["license"]["redistribution_permitted"] is False
    assert manifest["row_counts"]["counts_long"] == 24


def test_build_is_byte_deterministic_and_manifest_hashes_match(tmp_path: Path) -> None:
    sources = _synthetic_sources(tmp_path / "source")
    first = tmp_path / "first"
    second = tmp_path / "second"
    _build(sources, first)
    _build(sources, second)

    names = sorted(path.name for path in first.iterdir())
    assert names == sorted(path.name for path in second.iterdir())
    assert all((first / name).read_bytes() == (second / name).read_bytes() for name in names)

    manifest = json.loads((first / "source_manifest.json").read_text(encoding="utf-8"))
    for artifact in manifest["artifacts"].values():
        import hashlib

        content = (first / artifact["filename"]).read_bytes()
        assert hashlib.sha256(content).hexdigest() == artifact["sha256"]


def test_geometry_is_preserved_and_osm_conflict_remains_visible(tmp_path: Path) -> None:
    sources = _synthetic_sources(tmp_path / "source", location_osm_id=999)
    manifest = _build(sources, tmp_path / "study")

    locations = json.loads(
        (tmp_path / "study" / "survey_segments.geojson").read_text(encoding="utf-8")
    )
    feature = locations["features"][0]
    assert feature["geometry"]["coordinates"] == [
        [31.23, 30.04],
        [31.231, 30.041],
    ]
    assert feature["properties"]["osm_id"] == 999
    assert feature["properties"]["count_join_status"] == "osm_id_mismatch"
    assert any("conflicting OSM IDs" in item for item in manifest["validation_warnings"])
    with (tmp_path / "study" / "join_diagnostics.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        diagnostic = next(csv.DictReader(stream))
    assert diagnostic["location_osm_id"] == "999"
    assert diagnostic["count_osm_ids"] == "123"
    assert diagnostic["status"] == "osm_id_mismatch"


def test_build_fails_atomically_on_invalid_source(tmp_path: Path) -> None:
    sources = _synthetic_sources(tmp_path / "source", negative_count=True)
    output = tmp_path / "study"
    with pytest.raises(ObservedTrafficValidationError, match="non-negative"):
        _build(sources, output)
    assert not output.exists()
    assert not list(tmp_path.glob(".study.*"))


def test_build_refuses_overwrite_and_unlicensed_redistribution(tmp_path: Path) -> None:
    sources = _synthetic_sources(tmp_path / "source")
    output = tmp_path / "study"
    _build(sources, output)
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        _build(sources, output)

    with pytest.raises(ValueError, match="license_identifier"):
        build_observed_study(
            *sources,
            tmp_path / "other",
            source_retrieved_at_utc="2026-11-08T10:00:00Z",
            redistribution_permitted=True,
        )


def test_prepare_returns_paths_and_loader_rechecks_invariants(tmp_path: Path) -> None:
    sources = _synthetic_sources(tmp_path / "source")
    bundle = prepare_observed_data(
        *sources,
        tmp_path / "study",
        source_retrieved_at_utc="2026-11-08T10:00:00Z",
        expected_count_rows=2,
        expected_segments=1,
    )
    assert bundle["manifest_path"].is_file()
    assert bundle["survey_segments_path"].is_file()
    frame = load_normalized_counts(bundle["root"])
    assert len(frame) == 2
    assert frame.loc[0, "road_name"] == ""
    assert frame.loc[0, "observed_vehicle_total"] == 12
    assert frame["observed_vehicle_total"].dtype.name == "int64"


def test_loader_rejects_corrupted_normalized_artifact(tmp_path: Path) -> None:
    output = tmp_path / "study"
    _build(_synthetic_sources(tmp_path / "source"), output)
    counts_path = output / "counts_wide.csv"
    counts_path.write_text(
        counts_path.read_text(encoding="utf-8").replace(",10,2,", ",999,2,"),
        encoding="utf-8",
    )
    with pytest.raises(ObservedTrafficValidationError, match="hash mismatch"):
        load_normalized_counts(output)
