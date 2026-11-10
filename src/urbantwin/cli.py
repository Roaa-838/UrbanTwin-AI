"""Command-line interface for the reproducible UrbanTwin-AI workflows."""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

from urbantwin.common import ROOT, load_yaml, require_real_provenance, save_json, sha256_file

STUDY = ROOT / "configs/study_area.yaml"
EXPERIMENTS = ROOT / "configs/experiments.yaml"
OBSERVED_CONFIG = ROOT / "configs/observed_traffic.yaml"
OBSERVED_EO_CONFIG = ROOT / "configs/observed_eo_2024.yaml"


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
    observed = commands.add_parser(
        "observed-data-audit",
        help="Validate local 2024 observed traffic counts and survey geometries",
    )
    observed.add_argument("--counts", type=Path, required=True, help="Local counts CSV or ZIP")
    observed.add_argument(
        "--locations",
        type=Path,
        required=True,
        help="Local survey-locations GeoJSON or single-GeoJSON ZIP",
    )
    observed.add_argument("--expected-count-rows", type=int, default=6618)
    observed.add_argument("--expected-segments", type=int, default=145)
    prepare = commands.add_parser(
        "observed-data-prepare",
        help="Validate and normalize local TfC counts and survey geometry without overwriting",
    )
    prepare.add_argument("--counts", type=Path, required=True, help="Official counts CSV or ZIP")
    prepare.add_argument(
        "--locations", type=Path, required=True, help="Official survey GeoJSON or ZIP"
    )
    prepare.add_argument(
        "--source-retrieved-at-utc",
        required=True,
        help="Actual ISO-8601 UTC download/retrieval timestamp",
    )
    prepare.add_argument(
        "--output-dir", type=Path, default=ROOT / "data/observed/tfc/normalized_v1"
    )
    prepare.add_argument("--config", type=Path, default=OBSERVED_CONFIG)
    prepare.add_argument("--expected-count-rows", type=int, default=6618)
    prepare.add_argument("--expected-segments", type=int, default=145)
    observed_eo = commands.add_parser(
        "observed-eo-fetch",
        help="Fetch separate real 2024 Sentinel-2 context without replacing the 2021 EO bundle",
    )
    observed_eo.add_argument("--config", type=Path, default=OBSERVED_EO_CONFIG)
    observed_eo.add_argument(
        "--output-dir", type=Path, default=ROOT / "data/observed/eo_2024"
    )
    observed_roads = commands.add_parser(
        "observed-roads",
        help="Build checksum-validated current OSM context for the selected observed site",
    )
    observed_roads.add_argument(
        "--study-dir", type=Path, default=ROOT / "data/observed/tfc/normalized_v1"
    )
    observed_roads.add_argument(
        "--output-dir", type=Path, default=ROOT / "data/observed/osm/current_tahrir"
    )
    observed_roads.add_argument("--config", type=Path, default=OBSERVED_CONFIG)
    observed_roads.add_argument(
        "--pbf",
        type=Path,
        help="Optional Geofabrik Egypt PBF fallback (geometry-only, never treated as routable)",
    )
    observed_roads.add_argument("--retry-attempts", type=int, default=3)
    study = commands.add_parser(
        "observed-study",
        help="Run the frozen measured-link characterization and leakage-controlled benchmark",
    )
    study.add_argument(
        "--study-dir", type=Path, default=ROOT / "data/observed/tfc/normalized_v1"
    )
    study.add_argument(
        "--output-dir", type=Path, default=ROOT / "results/observed_traffic"
    )
    study.add_argument("--config", type=Path, default=OBSERVED_CONFIG)
    study.add_argument("--eo-dir", type=Path, default=ROOT / "data/observed/eo_2024")
    study.add_argument(
        "--road-dir", type=Path, default=ROOT / "data/observed/osm/current_tahrir"
    )
    study.add_argument(
        "--no-eo-context",
        action="store_true",
        help="Run without the separate near-survey Sentinel-2/WorldCover context bundle",
    )
    study.add_argument(
        "--resume",
        action="store_true",
        help="Reuse a complete strictly validated local study and compatible MLflow run",
    )
    study.add_argument(
        "--mlflow", action="store_true", help="Log validated aggregates to optional MLflow"
    )
    study.add_argument("--mlflow-tracking-uri", help="Optional MLflow tracking URI")
    observed_report = commands.add_parser(
        "observed-report",
        help="Build the provenance-checked measured-traffic research PDF",
    )
    observed_report.add_argument(
        "--output",
        type=Path,
        default=ROOT / "output/pdf/urbantwin_observed_traffic_study.pdf",
    )
    observed_report.add_argument(
        "--study-output", type=Path, default=ROOT / "results/observed_traffic"
    )
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
    traffic.add_argument(
        "--resume",
        action="store_true",
        help="Reuse compatible, validated per-simulation SUMO checkpoints",
    )
    traffic.add_argument(
        "--checkpoint-dir",
        type=Path,
        help="Override the local checkpoint directory (independent of final results)",
    )
    traffic.add_argument(
        "--mlflow",
        action="store_true",
        help="Log the completed benchmark to optional local/configured MLflow tracking",
    )
    traffic.add_argument(
        "--mlflow-tracking-uri",
        help="MLflow tracking URI (default: local SQLite under .urbantwin/mlflow)",
    )
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

    from urbantwin.observed.artifacts import (
        ObservedArtifactIntegrityError,
        validate_observed_artifacts,
    )

    observed_source = ROOT / "data/observed/tfc/normalized_v1/source_manifest.json"
    if not observed_source.is_file():
        print(f"NOT RUN: normalized observed traffic ({_relative(observed_source)})")
    else:
        try:
            source_manifest = _read_json(observed_source)
            if source_manifest.get("data_kind") != "real_observed_traffic":
                raise ValueError("normalized source has the wrong data_kind")
            print(
                "READY: normalized observed traffic "
                f"({_relative(observed_source)}) - source=real_observed_traffic"
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            print(f"INVALID: normalized observed traffic ({_relative(observed_source)}) - {exc}")

    from urbantwin.observed.roads import ObservedRoadError, validate_observed_road_bundle

    observed_roads = ROOT / "data/observed/osm/current_tahrir"
    try:
        road_bundle = validate_observed_road_bundle(observed_roads)
        if road_bundle is None:
            print(f"NOT RUN: observed-site current OSM ({_relative(observed_roads)})")
        else:
            connectivity = road_bundle.get("connectivity") or {}
            print(
                "READY: observed-site current OSM "
                f"({_relative(observed_roads / 'road_manifest.json')}) - "
                f"source={road_bundle['source_kind']}, nodes={road_bundle.get('nodes', 'n/a')}, "
                f"directed_edges={road_bundle.get('directed_edges', 'n/a')}, "
                f"weak_components={connectivity.get('weakly_connected_components', 'n/a')}"
            )
    except ObservedRoadError as exc:
        print(f"INVALID: observed-site current OSM ({_relative(observed_roads)}) - {exc}")
    observed_metrics = ROOT / "results/observed_traffic/observed_metrics.csv"
    try:
        observed_bundle = validate_observed_artifacts(ROOT)
        if observed_bundle is None:
            print(f"NOT RUN: observed traffic study ({_relative(observed_metrics)})")
        else:
            print(
                "READY: observed traffic study "
                f"({_relative(observed_metrics)}) - source=real_observed_traffic_analysis"
            )
    except ObservedArtifactIntegrityError as exc:
        print(f"INVALID: observed traffic study ({_relative(observed_metrics)}) - {exc}")

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


def _run_observed_data_audit(args: argparse.Namespace) -> None:
    from urbantwin.data.observed_traffic import audit_observed_traffic

    summary = audit_observed_traffic(
        args.counts,
        args.locations,
        expected_count_rows=args.expected_count_rows,
        expected_segments=args.expected_segments,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def _run_observed_data_prepare(args: argparse.Namespace) -> None:
    from urbantwin.data.observed_study import prepare_observed_data

    configuration = load_yaml(args.config)
    sources = configuration.get("sources")
    if not isinstance(sources, dict):
        raise TypeError("Observed config must contain a sources mapping")
    expected = {
        "counts": (sources.get("traffic_counts") or {}).get("expected_sha256"),
        "locations": (sources.get("survey_locations") or {}).get("expected_sha256"),
    }
    for role, path in (("counts", args.counts), ("locations", args.locations)):
        if not path.is_file():
            raise FileNotFoundError(f"{role} input is not a file: {path}")
        actual = sha256_file(path)
        if expected[role] != actual:
            raise ValueError(
                f"{role} source checksum does not match the audited release: {actual}"
            )
    license_config = sources.get("license") or {}
    bundle = prepare_observed_data(
        args.counts,
        args.locations,
        args.output_dir,
        source_retrieved_at_utc=args.source_retrieved_at_utc,
        license_status=str(license_config.get("status", "unspecified_by_publisher")),
        redistribution_permitted=bool(license_config.get("redistribution_permitted", False)),
        expected_count_rows=args.expected_count_rows,
        expected_segments=args.expected_segments,
    )
    manifest = bundle["manifest"]
    print(
        json.dumps(
            {
                "data_kind": manifest["data_kind"],
                "output_dir": str(bundle["root"]),
                "counts_wide_rows": manifest["row_counts"]["counts_wide"],
                "counts_long_rows": manifest["row_counts"]["counts_long"],
                "license": manifest["license"],
                "warnings": manifest["validation_warnings"],
            },
            indent=2,
        )
    )


def _run_observed_study(args: argparse.Namespace) -> None:
    from urbantwin.observed.artifacts import validate_observed_artifacts
    from urbantwin.observed.benchmark import run_observed_benchmark

    if args.mlflow_tracking_uri and not args.mlflow:
        raise ValueError("--mlflow-tracking-uri requires --mlflow")
    if args.output_dir.exists() and args.resume:
        verified = validate_observed_artifacts(ROOT, args.output_dir, source_dir=args.study_dir)
        if verified is None:
            raise RuntimeError("Observed output exists but is not a complete validated study")
        metrics = verified["metrics"]
        print("Reused the complete provenance-validated local observed study.")
    else:
        metrics = run_observed_benchmark(
            args.study_dir,
            args.output_dir,
            args.config,
            eo_dir=None if args.no_eo_context else args.eo_dir,
            road_dir=args.road_dir,
        )
    display = metrics[
        ["model", "feature_set", "information_budget", "rows", "mae", "rmse"]
    ].copy()
    print("Measured link-count study; persistence and measurement-free tasks are not comparable.")
    print(display.to_string(index=False))
    if args.mlflow:
        from urbantwin.observed.mlflow_tracking import track_observed_study

        result = track_observed_study(
            args.output_dir,
            project_root=ROOT,
            tracking_uri=args.mlflow_tracking_uri,
            resume=args.resume,
        )
        action = "reused" if result["already_completed"] else "logged"
        print(
            f"MLflow {action}: experiment=UrbanTwin-Observed-Traffic "
            f"run_id={result['run_id']} tracking_uri={result['tracking_uri']}"
        )
    print(
        "Measured-demand signal SUMO: NOT RUN. The source lacks turning movements, "
        "routes, and observed signal timing; see provenance.json."
    )


def _run_observed_eo_fetch(args: argparse.Namespace) -> None:
    from urbantwin.data.earth_observation import fetch_real

    if args.output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite observed EO directory: {args.output_dir}"
        )
    configuration = load_yaml(args.config)
    if configuration.get("worldcover_role") != (
        "historic_reference_product_not_2024_ground_truth"
    ):
        raise ValueError("Observed EO config must label WorldCover as a historic reference")
    provenance = fetch_real(configuration, args.output_dir)
    print(
        json.dumps(
            {
                "data_kind": provenance["data_kind"],
                "output_dir": str(args.output_dir),
                "sentinel_stac_item_ids": provenance["sentinel_stac_item_ids"],
                "sentinel_items": provenance["sentinel_items"],
                "worldcover_role": configuration["worldcover_role"],
                "eo_stack_sha256": provenance["eo_stack_sha256"],
            },
            indent=2,
        )
    )


def _run_observed_roads(args: argparse.Namespace) -> None:
    from urbantwin.observed.roads import prepare_observed_roads

    manifest = prepare_observed_roads(
        args.study_dir,
        args.output_dir,
        load_yaml(args.config),
        pbf_path=args.pbf,
        retry_attempts=args.retry_attempts,
    )
    print(
        json.dumps(
            {
                "data_kind": manifest["data_kind"],
                "source_kind": manifest["source_kind"],
                "output_dir": str(args.output_dir.resolve()),
                "routable_graph": manifest["routable_graph"],
                "nodes": manifest.get("nodes"),
                "directed_edges": manifest.get("directed_edges"),
                "connectivity": manifest.get("connectivity"),
                "map_matching": manifest.get("map_matching"),
                "simulation_readiness": manifest["simulation_readiness"],
                "attribution": manifest["attribution"],
                "license": manifest["license"],
            },
            indent=2,
            sort_keys=True,
        )
    )


def _run_observed_report(args: argparse.Namespace) -> None:
    from urbantwin.observed.report import make_observed_report

    output = make_observed_report(ROOT, args.output, observed_output=args.study_output)
    print(output)
    print(output.with_suffix(".provenance.json"))


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
    if args.mlflow_tracking_uri and not args.mlflow:
        raise ValueError("--mlflow-tracking-uri requires --mlflow")
    frame = run_benchmark(
        config_path,
        output,
        str(args.network) if args.network is not None else None,
        args.routes_template,
        args.tls_id,
        resume=args.resume,
        checkpoint_dir=args.checkpoint_dir,
    )
    figures = traffic_figures(output)
    mlflow_result = None
    mlflow_error = None
    if args.mlflow:
        from urbantwin.traffic.mlflow_tracking import (
            MlflowTrackingError,
            MlflowUnavailableError,
            track_benchmark,
        )

        try:
            mlflow_result = track_benchmark(
                output,
                config_path=config_path,
                tracking_uri=args.mlflow_tracking_uri,
                resume=args.resume,
                figure_paths=figures,
            )
        except MlflowUnavailableError as exc:
            mlflow_error = (
                "MLflow tracking was requested but the optional dependency is unavailable. "
                "Install it with `python -m pip install -e \".[mlflow]\"`. "
                f"Local SUMO artifacts remain available in {output}. Details: {exc}"
            )
        except MlflowTrackingError as exc:
            mlflow_error = (
                "MLflow tracking failed after local SUMO artifacts were validated and saved. "
                f"Local results remain available in {output}. Details: {exc}"
            )
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
    if mlflow_result is not None:
        action = (
            "reused compatible completed run"
            if mlflow_result.get("already_completed")
            else "logged completed run"
        )
        print(
            f"MLflow {action}: run_id={mlflow_result['run_id']} "
            f"tracking_uri={mlflow_result['tracking_uri']}"
        )
    if mlflow_error is not None:
        print(f"ERROR: {mlflow_error}", file=sys.stderr)
        raise RuntimeError(mlflow_error)


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
        "observed-data-audit": _run_observed_data_audit,
        "observed-data-prepare": _run_observed_data_prepare,
        "observed-eo-fetch": _run_observed_eo_fetch,
        "observed-roads": _run_observed_roads,
        "observed-report": _run_observed_report,
        "observed-study": _run_observed_study,
        "geo-train": _run_geo_train,
        "traffic-run": _run_traffic,
        "report": _run_report,
        "status": lambda _args: _print_status(),
    }
    try:
        handlers[args.cmd](args)
    except KeyboardInterrupt:
        message = "Interrupted by user. Completed local artifacts were preserved."
        if args.cmd == "traffic-run":
            message += " Completed SUMO checkpoints were preserved; rerun with --resume."
        print(message, file=sys.stderr)
        return 130
    except (FileNotFoundError, KeyError, RuntimeError, TypeError, ValueError) as exc:
        parser.exit(2, f"urbantwin {args.cmd} failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
