"""SUMO network, demand generation, and external-input validation.

The built-in junction and its demand are abstract simulation fixtures. They are
not calibrated Cairo geometry or observed traffic.
"""
from __future__ import annotations

import os
import shutil
import string
import subprocess
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_EMISSION_CLASS = "HBEFA3/PC_G_EU4"


def binary(name: str) -> str:
    path = shutil.which(name)
    if not path and os.getenv("SUMO_HOME"):
        executable = name + (".exe" if os.name == "nt" else "")
        path = str(Path(os.environ["SUMO_HOME"]) / "bin" / executable)
    if not path or not Path(path).is_file():
        raise RuntimeError(
            f"SUMO binary {name!r} not found. Install Eclipse SUMO and add its bin "
            "directory to PATH (the Python traci package alone is insufficient)."
        )
    return path


def create_network(out_dir: str | Path) -> Path:
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    nodes = ET.Element("nodes")
    for identifier, x, y, kind in [
        ("C", 0, 0, "traffic_light"),
        ("N", 0, 250, "priority"),
        ("S", 0, -250, "priority"),
        ("E", 250, 0, "priority"),
        ("W", -250, 0, "priority"),
    ]:
        ET.SubElement(nodes, "node", id=identifier, x=str(x), y=str(y), type=kind)
    ET.ElementTree(nodes).write(
        output / "junction.nod.xml", encoding="utf-8", xml_declaration=True
    )
    edges = ET.Element("edges")
    for side in ("N", "S", "E", "W"):
        ET.SubElement(
            edges,
            "edge",
            id=f"{side}_C",
            **{"from": side, "to": "C"},
            numLanes="1",
            speed="13.89",
        )
        ET.SubElement(
            edges,
            "edge",
            id=f"C_{side}",
            **{"from": "C", "to": side},
            numLanes="1",
            speed="13.89",
        )
    ET.ElementTree(edges).write(
        output / "junction.edg.xml", encoding="utf-8", xml_declaration=True
    )
    network = output / "junction.net.xml"
    command = [
        binary("netconvert"),
        "--node-files",
        str(output / "junction.nod.xml"),
        "--edge-files",
        str(output / "junction.edg.xml"),
        "--output-file",
        str(network),
        "--no-turnarounds",
        "true",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0 or not network.is_file():
        raise RuntimeError(f"netconvert failed: {result.stderr[-2500:]}")
    return network


def create_routes(
    out: str | Path,
    rate_hour: float,
    sim_seconds: int,
    seed: int,
    emission_class: str = DEFAULT_EMISSION_CLASS,
) -> Path:
    if not np.isfinite(rate_hour) or rate_hour <= 0:
        raise ValueError("Rate must be a finite value greater than zero")
    if isinstance(sim_seconds, bool) or int(sim_seconds) != sim_seconds or sim_seconds <= 0:
        raise ValueError("sim_seconds must be a positive integer")
    if not emission_class:
        raise ValueError("emission_class must be explicit for generated routes")
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    root = ET.Element("routes")
    ET.SubElement(
        root,
        "vType",
        id="car",
        vClass="passenger",
        emissionClass=emission_class,
        accel="2.6",
        decel="4.5",
        sigma="0.5",
        length="5",
        minGap="2.5",
        maxSpeed="13.89",
    )
    routes = {
        "N_S": "N_C C_S",
        "S_N": "S_C C_N",
        "E_W": "E_C C_W",
        "W_E": "W_C C_E",
    }
    for route_id, route_edges in routes.items():
        ET.SubElement(root, "route", id=route_id, edges=route_edges)
    rng = np.random.default_rng(seed)
    arrivals: list[tuple[float, str]] = []
    for route_id in routes:
        count = int(rng.poisson(float(rate_hour) * int(sim_seconds) / 3600))
        arrivals.extend(
            (float(departure), route_id)
            for departure in rng.uniform(0, int(sim_seconds), size=count)
        )
    for sequence, (departure, route_id) in enumerate(sorted(arrivals)):
        ET.SubElement(
            root,
            "vehicle",
            id=f"v_{seed}_{sequence}",
            type="car",
            route=route_id,
            depart=f"{departure:.2f}",
            departLane="best",
            departSpeed="max",
        )
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    return path


def _xml_root(path: Path, description: str) -> ET.Element:
    try:
        return ET.parse(path).getroot()
    except (ET.ParseError, OSError) as exc:
        raise ValueError(f"Cannot parse {description} XML {path}: {exc}") from exc


def count_route_vehicles(routes: str | Path) -> int:
    """Count explicit vehicles, rejecting demand forms that cannot be counted exactly."""
    path = Path(routes)
    root = _xml_root(path, "route")
    unsupported = [
        element.tag for element in root if element.tag in {"flow", "trip", "personFlow"}
    ]
    if unsupported:
        raise ValueError(
            f"Route file {path} uses {sorted(set(unsupported))}; provide explicit <vehicle> "
            "elements so generated/pending counts remain auditable"
        )
    vehicles = root.findall("vehicle")
    if not vehicles:
        raise ValueError(f"Route file contains no explicit vehicles: {path}")
    identifiers = [vehicle.get("id") for vehicle in vehicles]
    if any(not identifier for identifier in identifiers) or len(set(identifiers)) != len(identifiers):
        raise ValueError(f"Route file has missing or duplicate vehicle IDs: {path}")
    return len(vehicles)


def validate_routes_for_network(
    network: str | Path,
    routes: str | Path,
    *,
    require_explicit_emissions: bool = False,
) -> dict[str, Any]:
    """Validate explicit route edges, connectivity, demand, and emission assumptions."""
    network_path = Path(network)
    routes_path = Path(routes)
    network_root = _xml_root(network_path, "network")
    route_root = _xml_root(routes_path, "route")
    edge_ids = {
        edge.get("id")
        for edge in network_root.findall("edge")
        if edge.get("id") and edge.get("function") != "internal" and not edge.get("id", "").startswith(":")
    }
    connections = {
        (connection.get("from"), connection.get("to"))
        for connection in network_root.findall("connection")
        if connection.get("from") and connection.get("to")
    }
    signal_ids = sorted(
        {logic.get("id") for logic in network_root.findall("tlLogic") if logic.get("id")}
    )
    if not edge_ids:
        raise ValueError(f"SUMO network contains no usable edges: {network_path}")
    route_definitions: dict[str, list[str]] = {}
    for route in route_root.findall("route"):
        identifier = route.get("id")
        values = (route.get("edges") or "").split()
        if not identifier or not values:
            raise ValueError(f"Every top-level route needs an ID and edge list: {routes_path}")
        route_definitions[identifier] = values
    count = count_route_vehicles(routes_path)
    used_edges: set[str] = set()
    used_types: set[str] = set()
    for vehicle in route_root.findall("vehicle"):
        route_id = vehicle.get("route")
        inline = vehicle.find("route")
        if route_id:
            if route_id not in route_definitions:
                raise ValueError(f"Vehicle references unknown route {route_id!r}: {routes_path}")
            sequence = route_definitions[route_id]
        elif inline is not None and inline.get("edges"):
            sequence = inline.get("edges", "").split()
        else:
            raise ValueError(f"Every vehicle must reference an explicit edge route: {routes_path}")
        missing = set(sequence) - edge_ids
        if missing:
            raise ValueError(
                f"Route file {routes_path} references edges absent from {network_path}: "
                f"{sorted(missing)}"
            )
        disconnected = [
            pair for pair in pairwise(sequence) if connections and pair not in connections
        ]
        if disconnected:
            raise ValueError(f"Route has disconnected edge pairs {disconnected}: {routes_path}")
        used_edges.update(sequence)
        used_types.add(vehicle.get("type", "DEFAULT_VEHTYPE"))
    type_emissions = {
        vehicle_type.get("id"): vehicle_type.get("emissionClass")
        for vehicle_type in route_root.findall("vType")
        if vehicle_type.get("id")
    }
    missing_types = sorted(
        type_id
        for type_id in used_types
        if type_id == "DEFAULT_VEHTYPE" or not type_emissions.get(type_id)
    )
    if require_explicit_emissions and missing_types:
        raise ValueError(
            "CO2 tracking for external routes requires explicit emissionClass on every used "
            f"vType; missing for {missing_types} in {routes_path}"
        )
    assumptions = sorted(
        {type_emissions[type_id] for type_id in used_types if type_emissions.get(type_id)}
    )
    return {
        "generated_vehicles": count,
        "used_edge_count": len(used_edges),
        "traffic_light_ids": signal_ids,
        "emission_classes": assumptions,
        "all_used_vtypes_have_explicit_emission_class": not missing_types,
    }


def _template_fields(template: str) -> set[str]:
    try:
        return {
            field_name
            for _, field_name, _, _ in string.Formatter().parse(template)
            if field_name is not None
        }
    except ValueError as exc:
        raise ValueError(f"Invalid routes template: {exc}") from exc


def validate_external_inputs(
    network_file: str | Path,
    routes_template: str,
    scenarios: Iterable[str],
    seeds: Iterable[int],
    *,
    tls_id: str | None = None,
    require_explicit_emissions: bool = False,
) -> tuple[Path, dict[tuple[str, int], Path], dict[tuple[str, int], dict[str, Any]]]:
    """Resolve and validate every external artifact before any generated-route work."""
    network = Path(network_file).expanduser().resolve()
    if not network.is_file():
        raise FileNotFoundError(f"External SUMO network not found: {network}")
    if not network.name.endswith(".net.xml"):
        raise ValueError("External network must be an uncompressed .net.xml file")
    if not isinstance(routes_template, str) or not routes_template:
        raise ValueError("External network requires --routes-template")
    fields = _template_fields(routes_template)
    if fields != {"scenario", "seed"}:
        raise ValueError("routes-template must contain exactly {scenario} and {seed} placeholders")
    paths: dict[tuple[str, int], Path] = {}
    metadata: dict[tuple[str, int], dict[str, Any]] = {}
    for scenario in scenarios:
        for seed_value in seeds:
            seed = int(seed_value)
            route_path = Path(
                routes_template.format(scenario=scenario, seed=seed)
            ).expanduser().resolve()
            if not route_path.is_file():
                raise FileNotFoundError(f"External SUMO route file not found: {route_path}")
            key = (str(scenario), seed)
            paths[key] = route_path
            metadata[key] = validate_routes_for_network(
                network,
                route_path,
                require_explicit_emissions=require_explicit_emissions,
            )
    signal_ids = sorted(
        {signal for record in metadata.values() for signal in record["traffic_light_ids"]}
    )
    if not signal_ids:
        raise ValueError(f"External network has no traffic-light program: {network}")
    if tls_id is not None and tls_id not in signal_ids:
        raise ValueError(
            f"TLS {tls_id!r} not found in external network; available IDs: {signal_ids}"
        )
    return network, paths, metadata


__all__ = [
    "DEFAULT_EMISSION_CLASS",
    "binary",
    "count_route_vehicles",
    "create_network",
    "create_routes",
    "validate_external_inputs",
    "validate_routes_for_network",
]
