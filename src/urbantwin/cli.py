"""Command-line interface for the reproducible UrbanTwin-AI workflows."""
from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

from urbantwin.common import ROOT, load_yaml, require_real_provenance, save_json, sha256_file

STUDY = ROOT / "configs/study_area.yaml"
EXPERIMENTS = ROOT / "configs/experiments.yaml"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="UrbanTwin-AI reproducible research CLI")
    commands = parser.add_subparsers(dest="cmd", required=True)
    commands.add_parser(
        "demo", help="Run a deterministic synthetic offline smoke test (not research evidence)"
    )
    real = commands.add_parser(
        "eo-fetch", help="Fetch real Sentinel-2 L2A and WorldCover 2021 for the Cairo AOI"
    )
    real.add_argument("--config", type=Path, default=STUDY)
    roads = commands.add_parser("roads", help="Download and characterize current real OSM roads")
    roads.add_argument("--config", type=Path, default=STUDY)
    geo = commands.add_parser(
        "geo-train", help="Run repeated grouped WorldCover agreement benchmarks"
    )
    geo.add_argument("--data-dir", type=Path, default=ROOT / "data/real")
    geo.add_argument("--config", type=Path, default=EXPERIMENTS)
    traffic = commands.add_parser(
        "traffic-run", help="Run SUMO comparisons on an abstract or user-supplied network"
    )
    traffic.add_argument("--config", type=Path, default=EXPERIMENTS)
    traffic.add_argument(
        "--quick", action="store_true", help="Use tiny equal optimizer budgets for smoke testing only"
    )
    traffic.add_argument("--network", type=Path, help="Optional external SUMO .net.xml file")
    traffic.add_argument(
        "--routes-template",
        help="External route path template containing both {scenario} and {seed}",
    )
    traffic.add_argument("--tls-id", help="Traffic-light ID in an external network")
    report = commands.add_parser("report", help="Generate a provenance-checked PDF report")
    report.add_argument(
        "--draft",
        action="store_true",
        help="Generate a methodology-only PDF without consuming experiment results",
    )
    commands.add_parser("status", help="Inspect artifact presence, source labels, and integrity")
    return parser


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _eo_bundle_state(directory: Path, expected_kind: str) -> tuple[str, str]:
    stack = directory / "eo_stack.npz"
    provenance_path = directory / "provenance.json"
    if not stack.exists() and not provenance_path.exists():
        return "NOT RUN", f"no {expected_kind} raster bundle"
    if not stack.is_file() or not provenance_path.is_file():
        return "INVALID", "raster stack and provenance are incomplete"
    try:
        provenance = _read_json(provenance_path)
        if provenance.get("data_kind") != expected_kind:
            raise ValueError(
                f"expected source label {expected_kind!r}, found {provenance.get('data_kind')!r}"
            )
        if expected_kind == "real":
            require_real_provenance(provenance)
        recorded_hash = provenance.get("eo_stack_sha256")
        if not isinstance(recorded_hash, str) or recorded_hash != sha256_file(stack):
            raise ValueError("eo_stack.npz does not match its recorded SHA-256")
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return "INVALID", str(exc)
    return "READY", f"source={expected_kind}"


def _print_status() -> None:
    entries: list[tuple[str, Path, Callable[[], tuple[str, str]]]] = [
        (
            "real EO",
            ROOT / "data/real/eo_stack.npz",
            lambda: _eo_bundle_state(ROOT / "data/real", "real"),
        ),
        (
            "synthetic demo data",
            ROOT / "data/synthetic/eo_stack.npz",
            lambda: _eo_bundle_state(ROOT / "data/synthetic", "synthetic"),
        ),
    ]
    for name, path, checker in entries:
        state, note = checker()
        print(f"{state}: {name} ({_relative(path)}) - {note}")

    roads_path = ROOT / "results/roads.json"
    if not roads_path.exists():
        print(f"NOT RUN: OSM roads ({_relative(roads_path)})")
    else:
        try:
            roads = _read_json(roads_path)
            if roads.get("data_kind") != "real":
                raise ValueError("road summary is not labeled real")
            print(f"READY: OSM roads ({_relative(roads_path)}) - source=real")
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            print(f"INVALID: OSM roads ({_relative(roads_path)}) - {exc}")

    from urbantwin.visualization.report_writer import (
        ArtifactIntegrityError,
        validate_geoai_artifacts,
        validate_traffic_artifacts,
    )

    validators = [
        ("real GeoAI results", ROOT / "results/geoai/geoai_metrics.csv", validate_geoai_artifacts),
        ("SUMO results", ROOT / "results/traffic/traffic_metrics.csv", validate_traffic_artifacts),
        (
            "SUMO quick smoke",
            ROOT / "results/traffic_quick/traffic_metrics.csv",
            lambda root: validate_traffic_artifacts(root, root / "results/traffic_quick"),
        ),
    ]
    for name, path, validator in validators:
        try:
            bundle = validator(ROOT)
            if bundle is None:
                print(f"NOT RUN: {name} ({_relative(path)})")
            else:
                source = bundle["provenance"].get(
                    "data_kind", bundle.get("details", {}).get("provenance_kind")
                )
                print(f"READY: {name} ({_relative(path)}) - source={source}")
        except ArtifactIntegrityError as exc:
            print(f"INVALID: {name} ({_relative(path)}) - {exc}")

    demo_metrics = ROOT / "results/demo/geoai_metrics.csv"
    demo_details = ROOT / "results/demo/geoai_details.json"
    if not demo_metrics.exists() and not demo_details.exists():
        print(f"NOT RUN: synthetic demo results ({_relative(demo_metrics)})")
    elif not demo_metrics.is_file() or not demo_details.is_file():
        print(f"INVALID: synthetic demo results ({_relative(demo_metrics)}) - incomplete bundle")
    else:
        try:
            details = _read_json(demo_details)
            if details.get("provenance_kind") != "synthetic":
                raise ValueError("demo results are not labeled synthetic")
            if details.get("metrics_sha256") != sha256_file(demo_metrics):
                raise ValueError("demo metrics hash mismatch")
            print(
                f"READY: synthetic demo results ({_relative(demo_metrics)}) - source=synthetic"
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            print(f"INVALID: synthetic demo results ({_relative(demo_metrics)}) - {exc}")


def _run_demo(_args: argparse.Namespace) -> None:
    from urbantwin.data.demo import write_demo
    from urbantwin.geoai.train import run_classification

    provenance = write_demo(ROOT / "data/synthetic")
    table = run_classification(
        ROOT / "data/synthetic",
        ROOT / "results/demo",
        EXPERIMENTS,
        allow_synthetic=True,
    )
    print(str(provenance["warning"]).replace("—", "-"))
    print("Source label: synthetic")
    print(table.to_string(index=False))


def _run_eo_fetch(args: argparse.Namespace) -> None:
    from urbantwin.data.earth_observation import fetch_real

    provenance = fetch_real(load_yaml(args.config), ROOT / "data/real")
    print("Downloaded real Sentinel-2 L2A and ESA WorldCover data.")
    print(
        json.dumps(
            {
                "data_kind": provenance["data_kind"],
                "sentinel_stac_item_ids": provenance["sentinel_stac_item_ids"],
                "worldcover_stac_item_ids": provenance["worldcover_stac_item_ids"],
                "eo_stack_sha256": provenance["eo_stack_sha256"],
            },
            indent=2,
        )
    )


def _run_roads(args: argparse.Namespace) -> None:
    from urbantwin.data.spatial_features import road_network_stats

    stats, edges = road_network_stats(load_yaml(args.config))
    summary_path = ROOT / "results/roads.json"
    save_json(stats, summary_path)
    output = ROOT / "data/osm"
    output.mkdir(parents=True, exist_ok=True)
    edges[["geometry"]].reset_index(drop=True).to_file(
        output / "roads.geojson", driver="GeoJSON"
    )
    print(json.dumps(stats, indent=2))


def _run_geo_train(args: argparse.Namespace) -> None:
    from urbantwin.geoai.evaluate import make_figures
    from urbantwin.geoai.train import run_classification

    output = ROOT / "results/geoai"
    table = run_classification(args.data_dir, output, args.config)
    figures = make_figures(output)
    print(table.to_string(index=False))
    print("Figures:", *[str(path) for path in figures], sep="\n  ")


def _quick_config(config_path: Path) -> Path:
    configuration = load_yaml(config_path)
    traffic = configuration.get("traffic")
    if not isinstance(traffic, dict):
        raise TypeError("Configuration must contain a traffic mapping")
    traffic.update(
        {
            "optimizer_training_seeds": [17],
            "evaluation_seeds": [101],
            "ga_population": 4,
            "ga_generations": 1,
            "pso_particles": 4,
            "pso_iterations": 1,
            "simulation_seconds": 240,
            "clearance_seconds": 60,
            "warmup_seconds": min(30, int(traffic.get("warmup_seconds", 0))),
            "vehicle_rates_per_approach_per_hour": {"medium": 200},
        }
    )
    output = ROOT / "results/quick_experiments.yaml"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(configuration, sort_keys=False), encoding="utf-8")
    return output


def _run_traffic(args: argparse.Namespace) -> None:
    from urbantwin.traffic.benchmark import run_benchmark
    from urbantwin.visualization.plots import traffic_figures

    config_path = args.config
    if args.quick:
        config_path = _quick_config(config_path)
        print("QUICK TEST - equal optimizer budgets are too small for scientific conclusions")
    output = ROOT / ("results/traffic_quick" if args.quick else "results/traffic")
    frame = run_benchmark(
        config_path,
        output,
        str(args.network) if args.network is not None else None,
        args.routes_template,
        args.tls_id,
    )
    figures = traffic_figures(output)
    preferred = [
        "scenario",
        "method",
        "seed",
        "data_kind",
        "generated_vehicles",
        "inserted_vehicles",
        "completed_vehicles",
        "unfinished_vehicles",
        "pending_vehicles",
        "mean_time_loss_s_completed",
        "simulated_co2_g_inserted_fleet",
    ]
    available = [column for column in preferred if column in frame.columns]
    print("Traffic and CO2 outputs below are simulator-only; source labels are retained.")
    print(frame[available].to_string(index=False))
    print("Figures:", *[str(path) for path in figures], sep="\n  ")


def _run_report(args: argparse.Namespace) -> None:
    from urbantwin.visualization.report_writer import make_report

    output = ROOT / "reports" / (
        "methodology_draft.pdf" if args.draft else "generated_research_report.pdf"
    )
    print(make_report(output, include_draft=args.draft))


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    handlers: dict[str, Callable[[argparse.Namespace], None]] = {
        "demo": _run_demo,
        "eo-fetch": _run_eo_fetch,
        "roads": _run_roads,
        "geo-train": _run_geo_train,
        "traffic-run": _run_traffic,
        "report": _run_report,
        "status": lambda _args: _print_status(),
    }
    try:
        handlers[args.cmd](args)
    except (FileNotFoundError, KeyError, RuntimeError, TypeError, ValueError) as exc:
        parser.exit(2, f"urbantwin {args.cmd} failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
