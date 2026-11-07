from xml.etree import ElementTree as ET

import pytest

from urbantwin.traffic.network import (
    DEFAULT_EMISSION_CLASS,
    count_route_vehicles,
    create_routes,
    validate_external_inputs,
    validate_routes_for_network,
)


def _external_fixture(tmp_path, *, bad_edge=False, explicit_emissions=True):
    tmp_path.mkdir(parents=True, exist_ok=True)
    network = tmp_path / "external.net.xml"
    net = ET.Element("net")
    ET.SubElement(net, "edge", id="a")
    ET.SubElement(net, "edge", id="b")
    ET.SubElement(net, "connection", **{"from": "a", "to": "b"})
    ET.SubElement(net, "tlLogic", id="tls0", type="static", programID="0", offset="0")
    ET.ElementTree(net).write(network, encoding="utf-8", xml_declaration=True)

    route = tmp_path / "medium_17.rou.xml"
    routes = ET.Element("routes")
    attributes = {"id": "car"}
    if explicit_emissions:
        attributes["emissionClass"] = DEFAULT_EMISSION_CLASS
    ET.SubElement(routes, "vType", **attributes)
    ET.SubElement(routes, "route", id="r", edges="a missing" if bad_edge else "a b")
    for index in range(2):
        ET.SubElement(
            routes,
            "vehicle",
            id=f"v{index}",
            type="car",
            route="r",
            depart=str(index),
        )
    ET.ElementTree(routes).write(route, encoding="utf-8", xml_declaration=True)
    return network, route


def test_route_generation_is_seed_deterministic_and_explicitly_labeled(tmp_path):
    first = create_routes(tmp_path / "a.xml", 200, 300, 55)
    second = create_routes(tmp_path / "b.xml", 200, 300, 55)
    assert first.read_bytes() == second.read_bytes()
    root = ET.parse(first).getroot()
    assert len(root.findall("route")) == 4
    assert count_route_vehicles(first) == len(root.findall("vehicle")) > 0
    assert root.find("vType").get("emissionClass") == DEFAULT_EMISSION_CLASS


def test_reject_zero_demand(tmp_path):
    with pytest.raises(ValueError):
        create_routes(tmp_path / "a.xml", 0, 100, 42)


def test_external_routes_are_checked_against_network_before_use(tmp_path):
    network, route = _external_fixture(tmp_path)
    metadata = validate_routes_for_network(network, route, require_explicit_emissions=True)
    assert metadata["generated_vehicles"] == 2
    assert metadata["traffic_light_ids"] == ["tls0"]
    assert metadata["emission_classes"] == [DEFAULT_EMISSION_CLASS]

    _, bad_route = _external_fixture(tmp_path / "bad", bad_edge=True)
    with pytest.raises(ValueError, match="edges absent"):
        validate_routes_for_network(network, bad_route)


def test_external_input_template_and_tls_are_validated(tmp_path, monkeypatch):
    network, route = _external_fixture(tmp_path)
    monkeypatch.chdir(tmp_path)
    resolved, paths, metadata = validate_external_inputs(
        network,
        "{scenario}_{seed}.rou.xml",
        ["medium"],
        [17],
        tls_id="tls0",
        require_explicit_emissions=True,
    )
    assert resolved == network.resolve()
    assert paths[("medium", 17)] == route.resolve()
    assert metadata[("medium", 17)]["generated_vehicles"] == 2

    with pytest.raises(ValueError, match="exactly"):
        validate_external_inputs(network, "medium_17.rou.xml", ["medium"], [17])
    with pytest.raises(ValueError, match="not found"):
        validate_external_inputs(
            network,
            "{scenario}_{seed}.rou.xml",
            ["medium"],
            [17],
            tls_id="wrong",
        )


def test_external_emissions_and_non_explicit_demand_are_rejected(tmp_path):
    network, route = _external_fixture(tmp_path, explicit_emissions=False)
    with pytest.raises(ValueError, match="emissionClass"):
        validate_routes_for_network(network, route, require_explicit_emissions=True)

    root = ET.parse(route).getroot()
    for vehicle in list(root.findall("vehicle")):
        root.remove(vehicle)
    ET.SubElement(root, "flow", id="f", route="r", begin="0", end="60", number="10")
    ET.ElementTree(root).write(route, encoding="utf-8", xml_declaration=True)
    with pytest.raises(ValueError, match="explicit <vehicle>"):
        count_route_vehicles(route)
