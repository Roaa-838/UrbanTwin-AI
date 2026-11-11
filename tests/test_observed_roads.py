from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import geopandas as gpd
import networkx as nx
import pytest
from shapely.geometry import LineString

from urbantwin.observed import roads
from urbantwin.observed.roads import (
    ObservedRoadError,
    prepare_observed_roads,
    validate_observed_road_bundle,
)


def _config() -> dict:
    return {
        "site_selection": {"aoi_wgs84": [31.225, 30.038, 31.247, 30.053]},
        "spatial": {"projected_crs": "EPSG:32636"},
    }


def _graph() -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph(crs="EPSG:4326")
    graph.add_node(1, x=31.2300, y=30.0450, street_count=1)
    graph.add_node(2, x=31.2350, y=30.0450, street_count=1)
    graph.add_edge(
        1,
        2,
        key=0,
        osmid=123,
        name="SYNTHETIC TEST ROAD",
        highway="motorway",
        oneway=True,
        maxspeed="80",
        length=500.0,
        geometry=LineString([(31.2300, 30.0450), (31.2350, 30.0450)]),
    )
    return graph


def _survey(projected_crs: str):
    return gpd.GeoDataFrame(
        {
            "segment_id": ["synthetic-segment"],
            "geometry_osm_id": ["123"],
            "inferred_site_id": ["synthetic-site"],
            "count_join_status": ["matched"],
            "geometry": [LineString([(31.2302, 30.0450), (31.2348, 30.0450)])],
        },
        crs="EPSG:4326",
    ).to_crs(projected_crs)


def test_overpass_bundle_is_directed_projected_and_checksum_validated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        roads,
        "spatial_segment_features",
        lambda _study, projected_crs: _survey(projected_crs),
    )
    output = tmp_path / "roads"

    manifest = prepare_observed_roads(
        tmp_path / "study",
        output,
        _config(),
        graph_fetcher=lambda *_args, **_kwargs: (_graph(), 1),
    )

    assert manifest["data_kind"] == "third_party_osm_current_road_context"
    assert manifest["source_kind"] == "overpass_current_osm"
    assert manifest["routable_graph"] is True
    assert manifest["directed"] is True
    assert manifest["connectivity"]["weakly_connected_components"] == 1
    assert manifest["edge_length_m"]["minimum"] > 0
    assert manifest["map_matching"]["spatially_reliable"] == 1
    assert manifest["map_matching"]["routing_direction_verified"] == 0
    matches = gpd.pd.read_csv(output / "survey_edge_matches.csv")
    assert matches.loc[0, "spatial_match_status"] == "matched_osm_id_geometry_supported"
    assert bool(matches.loc[0, "spatial_match_reliable"])
    assert not bool(matches.loc[0, "routing_direction_verified"])
    assert (output / "roads.graphml").is_file()
    assert validate_observed_road_bundle(output) == manifest


def test_road_validator_detects_corruption(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        roads,
        "spatial_segment_features",
        lambda _study, projected_crs: _survey(projected_crs),
    )
    output = tmp_path / "roads"
    prepare_observed_roads(
        tmp_path / "study",
        output,
        _config(),
        graph_fetcher=lambda *_args, **_kwargs: (_graph(), 1),
    )
    path = output / "edges_projected.geojson"
    path.write_text(path.read_text(encoding="utf-8") + "corrupt", encoding="utf-8")

    with pytest.raises(ObservedRoadError, match="hash mismatch"):
        validate_observed_road_bundle(output)


def test_overpass_retries_with_cache_and_bounded_backoff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempts = []

    def fetch(*_args, **_kwargs):
        attempts.append(1)
        if len(attempts) < 3:
            raise RuntimeError("SYNTHETIC TRANSIENT FAILURE")
        return _graph()

    settings = SimpleNamespace(use_cache=False, cache_folder="", requests_timeout=1)
    fake_ox = SimpleNamespace(
        settings=settings,
        graph=SimpleNamespace(graph_from_bbox=fetch),
    )
    real_dependencies = roads._dependencies()
    monkeypatch.setattr(
        roads,
        "_dependencies",
        lambda: (
            real_dependencies[0],
            real_dependencies[1],
            fake_ox,
            real_dependencies[3],
            real_dependencies[4],
            real_dependencies[5],
            real_dependencies[6],
        ),
    )
    waits = []

    graph, count = roads._fetch_overpass_graph(
        (31.225, 30.038, 31.247, 30.053),
        cache_folder=tmp_path / "cache",
        retry_attempts=3,
        retry_delays_s=(0.5, 1.0),
        sleep=waits.append,
    )

    assert graph.number_of_edges() == 1
    assert count == 3
    assert len(attempts) == 3
    assert waits == [0.5, 1.0]
    assert settings.use_cache is True
    assert settings.cache_folder == str(tmp_path / "cache")
    assert settings.requests_timeout == 60


def test_pbf_fallback_is_explicitly_geometry_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pbf = tmp_path / "egypt-latest.osm.pbf"
    pbf.write_bytes(b"SYNTHETIC TEST PBF PLACEHOLDER")
    geometry = gpd.GeoDataFrame(
        {
            "osm_id": [123],
            "name": ["SYNTHETIC TEST ROAD"],
            "highway": ["primary"],
            "length_m_projected": [100.0],
            "geometry": [LineString([(328000, 3325000), (328100, 3325000)])],
        },
        crs="EPSG:32636",
    )
    monkeypatch.setattr(roads, "_read_pbf_lines", lambda *_args, **_kwargs: geometry.copy())
    output = tmp_path / "roads"

    manifest = prepare_observed_roads(
        tmp_path / "study",
        output,
        _config(),
        pbf_path=pbf,
        graph_fetcher=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ObservedRoadError("SYNTHETIC OVERPASS FAILURE")
        ),
    )

    assert manifest["source_kind"] == "geofabrik_pbf_geometry_only"
    assert manifest["routable_graph"] is False
    assert manifest["connectivity"] is None
    assert manifest["simulation_readiness"] == "not_routable_geometry_only"
    assert "SYNTHETIC OVERPASS FAILURE" in manifest["overpass_error_before_fallback"]
    assert not (output / "roads.graphml").exists()
    assert validate_observed_road_bundle(output) == manifest


def test_failed_retrieval_is_atomic_and_existing_bundle_is_not_overwritten(
    tmp_path: Path,
) -> None:
    output = tmp_path / "roads"
    with pytest.raises(ObservedRoadError, match="no Geofabrik PBF fallback"):
        prepare_observed_roads(
            tmp_path / "study",
            output,
            _config(),
            graph_fetcher=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                ObservedRoadError("SYNTHETIC FAILURE")
            ),
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".roads.*"))

    output.mkdir()
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        prepare_observed_roads(tmp_path / "study", output, _config())
