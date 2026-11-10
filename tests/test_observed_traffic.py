import csv
import json
import zipfile

import pytest

from urbantwin import cli
from urbantwin.data.observed_traffic import (
    COUNTS_COLUMNS,
    ObservedTrafficValidationError,
    audit_observed_traffic,
)

SEGMENT_0 = "11111111-1111-4111-8111-111111111111"
SEGMENT_1 = "22222222-2222-4222-8222-222222222222"


def _count_row(segment_id, osm_id, direction, start, end, *, car=1):
    row = {column: "0" for column in COUNTS_COLUMNS}
    row.update(
        {
            "osm_id": str(osm_id),
            "Day": "Sunday",
            "Date": "21-07-24",
            "road_name": "Fixture Road",
            "direction": str(direction),
            "hour_start": start,
            "hour_end": end,
            "car": str(car),
            "segmentId": segment_id,
        }
    )
    return row


def _write_counts(path, rows, *, zipped=False):
    csv_path = path if not zipped else path.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COUNTS_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    if not zipped:
        return path
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.write(csv_path, arcname="2024_traffic_counts.csv")
    csv_path.unlink()
    return path


def _feature(segment_id, osm_id, direction, coordinates, *, original_schema=False):
    properties = {
        "osm_id": osm_id,
        "segmentId" if original_schema else "segmentid": segment_id,
        "direction": str(direction),
        "road_name": "Fixture Road",
        "highway": "primary",
        "oneway": "false",
        "survey_line_name": "Fixture Site",
        "osm_extract_date": "2026-10-01",
    }
    if not original_schema:
        properties["fid"] = int(direction)
    return {
        "type": "Feature",
        "properties": properties,
        "geometry": (
            {"type": "LineString", "coordinates": coordinates}
            if original_schema
            else {"type": "MultiLineString", "coordinates": [coordinates]}
        ),
    }


def _write_locations(path, *, projected=False):
    first = [31.23, 30.04] if not projected else [331000, 3325000]
    payload = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "EPSG:4326"}},
        "features": [
            _feature(SEGMENT_0, 123, 0, [first, [31.231, 30.041]]),
            _feature(SEGMENT_1, 124, 1, [[31.232, 30.042], [31.233, 30.043]]),
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_original_locations_zip(path):
    payload = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
        "features": [
            _feature(
                SEGMENT_0,
                123,
                0,
                [[31.23, 30.04], [31.231, 30.041]],
                original_schema=True,
            ),
            _feature(
                SEGMENT_1,
                124,
                1,
                [[31.232, 30.042], [31.233, 30.043]],
                original_schema=True,
            ),
        ],
    }
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("2024_traffic_survey_locations.geojson", json.dumps(payload))
    return path


def _valid_inputs(tmp_path, *, zipped=False):
    counts = tmp_path / ("counts.zip" if zipped else "counts.csv")
    _write_counts(
        counts,
        [
            _count_row(SEGMENT_0, 123, 0, "6:00", "6:20", car=3),
            _count_row(SEGMENT_1, 124, 1, "6:00", "6:20", car=4),
        ],
        zipped=zipped,
    )
    return counts, _write_locations(tmp_path / "locations.geojson")


def test_audit_accepts_csv_and_reports_observed_provenance(tmp_path):
    counts, locations = _valid_inputs(tmp_path)

    result = audit_observed_traffic(
        counts,
        locations,
        expected_count_rows=2,
        expected_segments=2,
    )

    assert result["data_kind"] == "real_observed_traffic"
    assert result["status"] == "validated"
    assert result["warnings"] == []
    assert result["counts"]["rows"] == 2
    assert result["counts"]["vehicle_count_total"] == 7
    assert result["counts"]["interval_minutes"] == 20
    assert result["locations"]["crs"] == "EPSG:4326"
    assert result["locations"]["survey_location_count"] is None
    assert result["locations"]["survey_line_names"] == 1
    assert result["association"]["matched_directional_segments"] == 2
    assert result["association"]["unmatched_count_segments"] == 0
    assert len(result["inputs"]["counts"]["sha256"]) == 64
    assert "turning movements" in result["interpretation_limits"][0]


def test_audit_reads_single_csv_zip_deterministically(tmp_path):
    counts, locations = _valid_inputs(tmp_path, zipped=True)

    first = audit_observed_traffic(counts, locations)
    second = audit_observed_traffic(counts, locations)

    assert first == second
    assert first["inputs"]["counts"]["bytes"] == counts.stat().st_size


def test_audit_accepts_original_geojson_schema_in_zip(tmp_path):
    counts, _ = _valid_inputs(tmp_path)
    locations = _write_original_locations_zip(tmp_path / "locations.zip")

    result = audit_observed_traffic(counts, locations)

    assert result["locations"]["crs"].endswith("CRS84")
    assert result["locations"]["geometry_types"] == ["LineString"]
    assert result["locations"]["segment_ids"] == 2


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("car", "-1", "non-negative"),
        ("hour_end", "6:40", "20-minute"),
        ("Date", "2024-07-21", "DD-MM-YY"),
    ],
)
def test_audit_rejects_invalid_count_values(tmp_path, field, value, message):
    counts, locations = _valid_inputs(tmp_path)
    row = _count_row(SEGMENT_0, 123, 0, "6:00", "6:20")
    row[field] = value
    _write_counts(counts, [row])

    with pytest.raises(ObservedTrafficValidationError, match=message):
        audit_observed_traffic(counts, locations)


def test_audit_rejects_unmatched_segment_keys(tmp_path):
    counts, locations = _valid_inputs(tmp_path)
    _write_counts(
        counts,
        [_count_row("33333333-3333-4333-8333-333333333333", 999, 0, "6:00", "6:20")],
    )

    with pytest.raises(ObservedTrafficValidationError, match="absent from survey locations"):
        audit_observed_traffic(counts, locations)


def test_audit_reports_mutable_osm_id_mismatch_without_losing_segment_join(tmp_path):
    counts, locations = _valid_inputs(tmp_path)
    payload = json.loads(locations.read_text(encoding="utf-8"))
    payload["features"][0]["properties"]["osm_id"] = 999
    locations.write_text(json.dumps(payload), encoding="utf-8")

    result = audit_observed_traffic(counts, locations)

    assert result["association"]["status"] == "validated_with_warnings"
    assert result["association"]["osm_id_mismatch_count"] == 1
    assert result["association"]["osm_id_mismatches"][0]["counts_osm_ids"] == ["123"]


def test_audit_rejects_non_wgs84_coordinate_values(tmp_path):
    counts, _ = _valid_inputs(tmp_path)
    locations = _write_locations(tmp_path / "locations.geojson", projected=True)

    with pytest.raises(ObservedTrafficValidationError, match="outside WGS84"):
        audit_observed_traffic(counts, locations)


def test_cli_observed_data_audit_is_read_only(tmp_path, capsys):
    counts, locations = _valid_inputs(tmp_path, zipped=True)
    before = sorted(path.name for path in tmp_path.iterdir())

    result = cli.main(
        [
            "observed-data-audit",
            "--counts",
            str(counts),
            "--locations",
            str(locations),
            "--expected-count-rows",
            "2",
            "--expected-segments",
            "2",
        ]
    )

    assert result == 0
    assert sorted(path.name for path in tmp_path.iterdir()) == before
    output = json.loads(capsys.readouterr().out)
    assert output["data_kind"] == "real_observed_traffic"

