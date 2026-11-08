"""Run paired, source-labeled SUMO controller comparisons.

The generated network is an abstract simulator fixture. Even an external user
network is not treated as a calibrated Cairo model without separate evidence.
"""
from __future__ import annotations

import hashlib
import inspect
import subprocess
import tempfile
import time
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from urbantwin.common import ROOT, load_yaml, metadata, save_json, sha256_file
from urbantwin.traffic.baselines import install_tuned_program
from urbantwin.traffic.checkpoints import CHECKPOINT_SCHEMA, CheckpointStore, SimulationIdentity
from urbantwin.traffic.network import (
    DEFAULT_EMISSION_CLASS,
    binary,
    create_network,
    create_routes,
    validate_external_inputs,
    validate_routes_for_network,
)
from urbantwin.traffic.optimize import genetic_algorithm, particle_swarm, random_search

_RUNNER_CONTRACT = "urbantwin.sumo-runner/v1"


def _sumo_version() -> str:
    """Return the executable version recorded in every checkpoint identity."""
    try:
        completed = subprocess.run(
            [binary("sumo"), "--version"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"Unable to determine the installed SUMO version: {exc}") from exc
    output = "\n".join(part for part in (completed.stdout, completed.stderr) if part).strip()
    first_line = next((line.strip() for line in output.splitlines() if line.strip()), "")
    if not first_line:
        raise RuntimeError("SUMO --version returned no version information")
    return first_line


@dataclass
class _ProgressReporter:
    total_estimated: int
    started: float = field(default_factory=time.monotonic)
    completed: int = 0
    physical_executions: int = 0
    cache_hits: int = 0
    lookup_statuses: Counter[str] = field(default_factory=Counter)

    def _line(
        self,
        *,
        scenario: str,
        method: str,
        objective_evaluation: str,
        seed: int,
        status: str,
    ) -> str:
        elapsed = time.monotonic() - self.started
        return (
            "SUMO progress "
            f"scenario={scenario} method={method} "
            f"objective_evaluation={objective_evaluation} seed={seed} "
            f"completed_executions={self.completed}/{self.total_estimated} "
            f"executed_this_run={self.physical_executions} cache_hits={self.cache_hits} "
            f"elapsed={elapsed:.1f}s status={status}"
        )

    def announce(self, **context: Any) -> None:
        print(self._line(status="starting", **context), flush=True)

    def finish(self, *, cache_hit: bool, lookup_status: str, **context: Any) -> None:
        self.completed += 1
        self.lookup_statuses[lookup_status] += 1
        if cache_hit:
            self.cache_hits += 1
            status = "cache-hit"
        else:
            self.physical_executions += 1
            status = (
                "completed"
                if lookup_status in {"missing", "fresh"}
                else f"recomputed-{lookup_status}"
            )
        print(self._line(status=status, **context), flush=True)

    def interrupted(self) -> None:
        elapsed = time.monotonic() - self.started
        print(
            "SUMO interrupted: "
            f"completed_executions={self.completed}/{self.total_estimated} "
            f"executed_this_run={self.physical_executions} cache_hits={self.cache_hits} "
            f"elapsed={elapsed:.1f}s; completed checkpoints remain durable.",
            flush=True,
        )


def _load_traffic_lib():
    try:
        import traci
    except ImportError as exc:
        raise RuntimeError(
            "Install SUMO Python support with `python -m pip install -e \".[traffic]\"`"
        ) from exc
    binary("sumo")
    return traci


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def run_one(
    network: Path,
    routes: Path,
    plan: tuple[int, int],
    seed: int,
    config: dict[str, Any],
    tls_id: str | None = None,
    route_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one fixed-horizon simulation and expose all demand censoring counts."""
    traci = _load_traffic_lib()
    route_record = route_metadata or validate_routes_for_network(network, routes)
    generated = int(route_record["generated_vehicles"])
    start = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="urbantwin_sumo_") as temporary:
        trip_path = Path(temporary) / "tripinfos.xml"
        horizon = int(config["simulation_seconds"]) + int(config["clearance_seconds"])
        command = [
            binary("sumo"),
            "--net-file",
            str(network.resolve()),
            "--route-files",
            str(routes.resolve()),
            "--tripinfo-output",
            str(trip_path),
            "--no-step-log",
            "true",
            "--duration-log.disable",
            "true",
            "--time-to-teleport",
            "-1",
            "--seed",
            str(seed),
        ]
        started = False
        inserted = 0
        arrived = 0
        minimum_expected = 0
        queue_sum = 0.0
        queue_observations = 0
        co2_mg = 0.0
        try:
            try:
                traci.start(command, numRetries=10)
            except (traci.exceptions.FatalTraCIError, traci.exceptions.TraCIException) as exc:
                raise RuntimeError(
                    "SUMO started but TraCI could not establish its localhost connection. "
                    "Check firewall/socket permissions and retry with --resume; no incomplete "
                    "simulation checkpoint was written."
                ) from exc
            started = True
            signal_ids = list(traci.trafficlight.getIDList())
            signal = tls_id or (signal_ids[0] if signal_ids else None)
            if not signal:
                raise RuntimeError("No traffic light is available in the SUMO network")
            if signal not in signal_ids:
                raise ValueError(f"TLS {signal!r} is not available; found {signal_ids}")
            program = install_tuned_program(traci, signal, plan)
            lanes = sorted(
                {
                    lane
                    for lane in traci.trafficlight.getControlledLanes(signal)
                    if lane and not lane.startswith(":")
                }
            )
            if not lanes:
                raise ValueError(f"TLS {signal!r} controls no physical approach lanes")
            track_emissions = bool(config.get("track_emissions", False))
            for step in range(horizon):
                traci.simulationStep()
                inserted += int(traci.simulation.getDepartedNumber())
                arrived += int(traci.simulation.getArrivedNumber())
                if step >= int(config.get("warmup_seconds", 0)):
                    queue_sum += sum(
                        traci.lane.getLastStepHaltingNumber(lane) for lane in lanes
                    )
                    queue_observations += 1
                if track_emissions:
                    delta_seconds = float(traci.simulation.getDeltaT())
                    co2_mg += delta_seconds * sum(
                        traci.vehicle.getCO2Emission(vehicle)
                        for vehicle in traci.vehicle.getIDList()
                    )
            minimum_expected = int(traci.simulation.getMinExpectedNumber())
        finally:
            if started:
                traci.close()
        if not trip_path.is_file():
            raise RuntimeError("SUMO did not write tripinfo output")
        root = ET.parse(trip_path).getroot()
        trips = [trip for trip in root.findall("tripinfo") if "timeLoss" in trip.attrib]
        waiting = [float(trip.attrib["waitingTime"]) for trip in trips if "waitingTime" in trip.attrib]
        time_loss = [float(trip.attrib["timeLoss"]) for trip in trips]
        duration = [float(trip.attrib["duration"]) for trip in trips if "duration" in trip.attrib]

    completed = len(trips)
    if completed != arrived:
        raise RuntimeError(
            f"SUMO arrival/tripinfo mismatch: arrived={arrived}, tripinfo={completed}"
        )
    if not (0 <= completed <= inserted <= generated):
        raise RuntimeError(
            "Invalid SUMO demand accounting: expected completed <= inserted <= generated, got "
            f"{completed} <= {inserted} <= {generated}"
        )
    unfinished = inserted - completed
    pending = generated - inserted
    if minimum_expected != unfinished + pending:
        raise RuntimeError(
            "SUMO minimum-expected count does not match unfinished plus pending vehicles: "
            f"{minimum_expected} != {unfinished} + {pending}"
        )
    track_emissions = bool(config.get("track_emissions", False))
    co2_g = co2_mg / 1000 if track_emissions else None
    return {
        "plan_ns_ew_s": [int(plan[0]), int(plan[1])],
        "generated_vehicles": generated,
        "inserted_vehicles": inserted,
        "completed_vehicles": completed,
        "unfinished_vehicles": unfinished,
        "pending_vehicles": pending,
        "completion_fraction_generated": completed / generated,
        "insertion_fraction_generated": inserted / generated,
        "completed_throughput_vehicles_per_sim_hour": completed * 3600 / horizon,
        "mean_waiting_time_s_completed": _mean(waiting),
        "mean_time_loss_s_completed": _mean(time_loss),
        "mean_trip_duration_s_completed": _mean(duration),
        "mean_approach_queue_vehicles": (
            queue_sum / queue_observations if queue_observations else None
        ),
        "simulated_co2_g_inserted_fleet": co2_g,
        "simulated_co2_g_per_inserted_vehicle": (
            co2_g / inserted if co2_g is not None and inserted else None
        ),
        "simulated_co2_label": "SUMO model estimate; simulator-only, not observed pollution",
        "wall_time_s": time.monotonic() - start,
        "signal_program": program,
        "queue_note": (
            "Mean halted vehicles on unique TLS-controlled approach lanes after warmup; "
            "not a maximum queue-length measure"
        ),
        "censoring_note": (
            "Waiting/time-loss/duration means use completed trips only. Generated, inserted, "
            "completed, unfinished, and pending counts expose horizon censoring."
        ),
    }


def objective_metrics(row: dict[str, Any]) -> float:
    """Arbitrary tuning scalar; scientific comparisons retain separate KPIs."""
    if not row["completed_vehicles"] or row["mean_time_loss_s_completed"] is None:
        return 1_000_000.0 + 2 * (row["unfinished_vehicles"] + row["pending_vehicles"])
    return float(
        row["mean_time_loss_s_completed"]
        + 2 * row["mean_approach_queue_vehicles"]
        + 2 * (row["unfinished_vehicles"] + row["pending_vehicles"])
    )


def _simulation_runner_source_sha256() -> str:
    """Hash only code that can change one plan/route simulation payload."""
    source = "\n".join(
        inspect.getsource(function) for function in (_load_traffic_lib, _mean, run_one)
    )
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _validate_config(config: dict[str, Any]) -> None:
    if "yellow_seconds" in config:
        raise ValueError(
            "traffic.yellow_seconds is not applied. Remove it: non-green durations are read "
            "from and preserved from each SUMO signal program."
        )
    train = [int(value) for value in config["optimizer_training_seeds"]]
    evaluation = [int(value) for value in config["evaluation_seeds"]]
    if not train or not evaluation:
        raise ValueError("Traffic training and evaluation seed lists must both be non-empty")
    if len(train) != len(set(train)) or len(evaluation) != len(set(evaluation)):
        raise ValueError("Traffic seed lists must not contain duplicates")
    if set(train) & set(evaluation):
        raise ValueError("Optimizer training seeds must not overlap evaluation seeds")
    ga_budget = int(config["ga_population"]) * int(config["ga_generations"])
    pso_budget = int(config["pso_particles"]) * int(config["pso_iterations"])
    if ga_budget != pso_budget:
        raise ValueError(
            "GA and PSO must have equal objective-evaluation budgets for comparison: "
            f"{ga_budget} != {pso_budget}"
        )
    if int(config["simulation_seconds"]) <= 0 or int(config["clearance_seconds"]) < 0:
        raise ValueError("simulation_seconds must be positive and clearance_seconds non-negative")
    horizon = int(config["simulation_seconds"]) + int(config["clearance_seconds"])
    warmup = int(config.get("warmup_seconds", 0))
    if warmup < 0 or warmup >= horizon:
        raise ValueError("warmup_seconds must be non-negative and below the simulation horizon")
    bounds = [int(value) for value in config["green_seconds_bounds"]]
    baseline = [int(value) for value in config["baseline_green_seconds"]]
    if len(bounds) != 2 or not (0 < bounds[0] < bounds[1]):
        raise ValueError("green_seconds_bounds must be two increasing positive integers")
    if len(baseline) != 2 or any(not bounds[0] <= value <= bounds[1] for value in baseline):
        raise ValueError("Each baseline green duration must lie within green_seconds_bounds")


def _config_reference(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _paired_outputs(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = frame.copy()
    keys = ["scenario", "seed", "data_kind"]
    metrics = [
        "mean_time_loss_s_completed",
        "mean_waiting_time_s_completed",
        "mean_approach_queue_vehicles",
        "completed_throughput_vehicles_per_sim_hour",
        "completed_vehicles",
        "unfinished_vehicles",
        "pending_vehicles",
        "completion_fraction_generated",
        "simulated_co2_g_inserted_fleet",
    ]
    for metric in metrics:
        frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
    fixed = frame.loc[frame["method"] == "fixed"].copy()
    methods = frame.loc[frame["method"] != "fixed"].copy()
    baseline = fixed[keys + metrics].rename(
        columns={metric: f"{metric}__fixed" for metric in metrics}
    )
    paired = methods.merge(baseline, on=keys, how="left", validate="many_to_one")
    for metric in metrics:
        paired[f"delta_{metric}_method_minus_fixed"] = paired[metric] - paired[
            f"{metric}__fixed"
        ]
    keep = keys + ["method"] + [
        column for column in paired.columns if column.startswith("delta_")
    ]
    deltas = paired[keep].copy()
    delta_columns = [column for column in deltas if column.startswith("delta_")]
    aggregations: dict[str, tuple[str, str]] = {"runs": ("seed", "count")}
    for column in delta_columns:
        aggregations[f"mean_{column}"] = (column, "mean")
        aggregations[f"std_{column}"] = (column, "std")
    summary = (
        deltas.groupby(["scenario", "method", "data_kind"], sort=True)
        .agg(**aggregations)
        .reset_index()
    )
    return deltas, summary


def run_benchmark(
    config_file: str | Path,
    out_dir: str | Path,
    network_file: str | None = None,
    routes_template: str | None = None,
    tls_id: str | None = None,
    *,
    resume: bool = False,
    checkpoint_dir: str | Path | None = None,
) -> pd.DataFrame:
    config_path = Path(config_file)
    config_all = load_yaml(config_path)
    config = config_all.get("traffic")
    if not isinstance(config, dict):
        raise TypeError("Configuration must contain a traffic mapping")
    _validate_config(config)
    scenarios = config.get("vehicle_rates_per_approach_per_hour")
    if not isinstance(scenarios, dict) or not scenarios:
        raise ValueError("vehicle_rates_per_approach_per_hour must be a non-empty mapping")
    train_seeds = [int(value) for value in config["optimizer_training_seeds"]]
    evaluation_seeds = [int(value) for value in config["evaluation_seeds"]]
    all_seeds = sorted(set(train_seeds + evaluation_seeds))
    track_emissions = bool(config.get("track_emissions", False))
    external = network_file is not None or routes_template is not None
    if (network_file is None) != (routes_template is None):
        raise ValueError("External SUMO runs require both --network and --routes-template")
    if not external and tls_id is not None:
        raise ValueError("--tls-id is only valid with an external network")

    route_paths: dict[tuple[str, int], Path]
    route_records: dict[tuple[str, int], dict[str, Any]]
    if external:
        network, route_paths, route_records = validate_external_inputs(
            network_file,
            routes_template,
            scenarios,
            all_seeds,
            tls_id=tls_id,
            require_explicit_emissions=track_emissions,
        )
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    if not external:
        network = create_network(output / "synthetic_network")
        route_paths = {}
        route_records = {}
        emission_class = str(config.get("emission_class", DEFAULT_EMISSION_CLASS))
        for scenario, rate in scenarios.items():
            for seed in all_seeds:
                key = (str(scenario), seed)
                route_path = create_routes(
                    output / "synthetic_network" / f"routes_{scenario}_{seed}.rou.xml",
                    float(rate),
                    int(config["simulation_seconds"]),
                    seed,
                    emission_class,
                )
                route_paths[key] = route_path
                route_records[key] = validate_routes_for_network(
                    network,
                    route_path,
                    require_explicit_emissions=track_emissions,
                )

    bounds = tuple(int(value) for value in config["green_seconds_bounds"])
    baseline = tuple(int(value) for value in config["baseline_green_seconds"])
    if len(bounds) != 2 or len(baseline) != 2:
        raise ValueError("green_seconds_bounds and baseline_green_seconds need exactly two values")
    evaluation_budget = int(config["ga_population"]) * int(config["ga_generations"])
    base_optimizer_seed = int(config_all.get("random_seed", 42))
    comparison: list[dict[str, Any]] = []
    optimization: dict[str, Any] = {}
    evaluation_programs: dict[str, Any] = {}
    data_kind = "simulation_external_user_network" if external else "simulation_abstract"
    config_hash = sha256_file(config_path)
    network_hash = sha256_file(network)
    route_hashes = {key: sha256_file(path) for key, path in route_paths.items()}
    runner_source_hash = _simulation_runner_source_sha256()
    signal_adapter_hash = sha256_file(Path(__file__).with_name("baselines.py"))
    optimizer_source_hash = sha256_file(Path(__file__).with_name("optimize.py"))
    installed_sumo_version = _sumo_version()
    checkpoint_root = Path(
        checkpoint_dir or (ROOT / ".urbantwin" / "checkpoints" / "traffic")
    )
    resolved_checkpoint = checkpoint_root.resolve()
    resolved_output = output.resolve()
    if (
        resolved_checkpoint == resolved_output
        or resolved_checkpoint in resolved_output.parents
        or resolved_output in resolved_checkpoint.parents
    ):
        raise ValueError(
            "SUMO checkpoints must be stored separately from final research artifacts: "
            f"checkpoint_dir={resolved_checkpoint}, out_dir={resolved_output}"
        )
    checkpoint_store = CheckpointStore(checkpoint_root)
    session_results: dict[str, dict[str, Any]] = {}
    total_estimated = len(scenarios) * (
        3 * evaluation_budget * len(train_seeds) + 4 * len(evaluation_seeds)
    )
    progress = _ProgressReporter(total_estimated)

    def cached_run(
        *,
        scenario_key: str,
        method: str,
        objective_evaluation: str,
        plan: tuple[int, int],
        seed: int,
    ) -> dict[str, Any]:
        route_record = route_records[(scenario_key, seed)]
        identity = SimulationIdentity(
            config_sha256=config_hash,
            network_sha256=network_hash,
            route_sha256=route_hashes[(scenario_key, seed)],
            sumo_version=installed_sumo_version,
            scenario=scenario_key,
            seed=seed,
            signal_plan={
                "tls_id": tls_id or "__first_available__",
                "green_seconds": [int(plan[0]), int(plan[1])],
            },
            emission_settings={
                "track_emissions": track_emissions,
                "configured_emission_class": config.get("emission_class"),
                "route_emission_classes": sorted(route_record.get("emission_classes", [])),
                "co2_measurement": "TraCI vehicle CO2 rate integrated over simulation steps",
            },
            simulation_settings={
                "runner_contract": _RUNNER_CONTRACT,
                "runner_source_sha256": runner_source_hash,
                "signal_adapter_sha256": signal_adapter_hash,
                "simulation_seconds": int(config["simulation_seconds"]),
                "clearance_seconds": int(config["clearance_seconds"]),
                "warmup_seconds": int(config.get("warmup_seconds", 0)),
                "time_to_teleport_s": -1,
                "step_length": "SUMO network/default",
                "queue_scope": "unique non-internal TLS-controlled lanes",
            },
        )
        context = {
            "scenario": scenario_key,
            "method": method,
            "objective_evaluation": objective_evaluation,
            "seed": seed,
        }
        progress.announce(**context)
        if identity.fingerprint in session_results:
            result = session_results[identity.fingerprint]
            progress.finish(cache_hit=True, lookup_status="session", **context)
            return result

        if resume:
            lookup = checkpoint_store.lookup(identity)
            if lookup.hit:
                assert lookup.result is not None
                session_results[identity.fingerprint] = lookup.result
                progress.finish(cache_hit=True, lookup_status="hit", **context)
                return lookup.result
            lookup_status = lookup.status
        else:
            lookup_status = "fresh"

        result = run_one(
            network,
            route_paths[(scenario_key, seed)],
            plan,
            seed,
            config,
            tls_id,
            route_record,
        )
        checkpoint_store.save(identity, result, overwrite=not resume)
        session_results[identity.fingerprint] = result
        progress.finish(cache_hit=False, lookup_status=lookup_status, **context)
        return result

    try:
        for scenario, rate in scenarios.items():
            scenario_key = str(scenario)

            def make_score(method: str, score_scenario: str):
                objective_index = 0

                def score(candidate_plan: tuple[int, int]) -> float:
                    nonlocal objective_index
                    objective_index += 1
                    measures = [
                        cached_run(
                            scenario_key=score_scenario,
                            method=method,
                            objective_evaluation=f"{objective_index}/{evaluation_budget}",
                            plan=candidate_plan,
                            seed=seed,
                        )
                        for seed in train_seeds
                    ]
                    return float(np.mean([objective_metrics(measure) for measure in measures]))

                return score

            optimized: dict[str, tuple[dict[str, Any], float]] = {}
            optimizer_calls = [
                (
                    "GA",
                    genetic_algorithm,
                    {
                        "population": int(config["ga_population"]),
                        "generations": int(config["ga_generations"]),
                        "seed": base_optimizer_seed,
                    },
                ),
                (
                    "PSO",
                    particle_swarm,
                    {
                        "particles": int(config["pso_particles"]),
                        "iterations": int(config["pso_iterations"]),
                        "seed": base_optimizer_seed + 1,
                    },
                ),
                (
                    "random_search",
                    random_search,
                    {"evaluations": evaluation_budget, "seed": base_optimizer_seed + 2},
                ),
            ]
            for method, optimizer, optimizer_kwargs in optimizer_calls:
                started = time.monotonic()
                result = optimizer(make_score(method, scenario_key), bounds, **optimizer_kwargs)
                elapsed = time.monotonic() - started
                if int(result["objective_evaluations"]) != evaluation_budget:
                    raise RuntimeError(
                        f"{method} did not use the declared equal evaluation budget"
                    )
                optimized[method] = (result, elapsed)
            optimization[scenario_key] = {
                method: {
                    **result,
                    "optimization_wall_time_s": elapsed,
                    "runtime_scope": (
                        "current invocation; validated cache hits reduce resumed wall time"
                    ),
                }
                for method, (result, elapsed) in optimized.items()
            }
            optimization[scenario_key].update(
                {
                    "objective_evaluation_budget_per_method": evaluation_budget,
                    "training_seed_count_per_objective": len(train_seeds),
                    "training_sumo_runs_per_method": evaluation_budget * len(train_seeds),
                    "training_seeds": train_seeds,
                    "evaluation_seeds": evaluation_seeds,
                }
            )
            plans = [("fixed", baseline, 0.0)] + [
                (method, tuple(result["plan"]), elapsed)
                for method, (result, elapsed) in optimized.items()
            ]
            for method, plan, optimizer_seconds in plans:
                for seed in evaluation_seeds:
                    result = cached_run(
                        scenario_key=scenario_key,
                        method=method,
                        objective_evaluation="held-out",
                        plan=plan,
                        seed=seed,
                    )
                    evaluation_programs[f"{scenario_key}__{method}__{seed}"] = result[
                        "signal_program"
                    ]
                    comparison.append(
                        {
                            "scenario": scenario_key,
                            "demand_vph_per_approach": float(rate),
                            "method": method,
                            "seed": seed,
                            "train_seed_overlap": False,
                            "green_ns_s": int(plan[0]),
                            "green_ew_s": int(plan[1]),
                            "data_kind": data_kind,
                            "optimization_wall_time_s": optimizer_seconds,
                            **{
                                key: value
                                for key, value in result.items()
                                if not isinstance(value, (list, dict))
                            },
                        }
                    )
    except KeyboardInterrupt:
        progress.interrupted()
        print(f"Resume checkpoints: {checkpoint_root.resolve()}", flush=True)
        raise

    if progress.completed != total_estimated:
        raise RuntimeError(
            "SUMO execution accounting mismatch: "
            f"completed {progress.completed} of {total_estimated} estimated requests"
        )

    frame = pd.DataFrame(comparison)
    result_keys = ["scenario", "method", "seed", "data_kind"]
    duplicates = frame.duplicated(result_keys, keep=False)
    if duplicates.any():
        repeated = frame.loc[duplicates, result_keys].to_dict(orient="records")
        raise RuntimeError(f"Duplicate held-out traffic results were produced: {repeated}")
    leading_columns = [
        "scenario",
        "demand_vph_per_approach",
        "method",
        "seed",
        "train_seed_overlap",
        "green_ns_s",
        "green_ew_s",
        "data_kind",
        "optimization_wall_time_s",
    ]
    frame = frame[leading_columns + sorted(set(frame.columns) - set(leading_columns))]
    metrics_path = output / "traffic_metrics.csv"
    frame.to_csv(metrics_path, index=False)
    summary = (
        frame.groupby(["scenario", "method", "data_kind"], sort=True)
        .agg(
            mean_waiting_time_s_completed=("mean_waiting_time_s_completed", "mean"),
            std_waiting_time_s_completed=("mean_waiting_time_s_completed", "std"),
            mean_time_loss_s_completed=("mean_time_loss_s_completed", "mean"),
            std_time_loss_s_completed=("mean_time_loss_s_completed", "std"),
            mean_approach_queue_vehicles=("mean_approach_queue_vehicles", "mean"),
            mean_completed_throughput_vph=(
                "completed_throughput_vehicles_per_sim_hour",
                "mean",
            ),
            mean_generated_vehicles=("generated_vehicles", "mean"),
            mean_inserted_vehicles=("inserted_vehicles", "mean"),
            mean_completed_vehicles=("completed_vehicles", "mean"),
            mean_unfinished_vehicles=("unfinished_vehicles", "mean"),
            mean_pending_vehicles=("pending_vehicles", "mean"),
            mean_completion_fraction_generated=("completion_fraction_generated", "mean"),
            mean_simulated_co2_g_inserted_fleet=(
                "simulated_co2_g_inserted_fleet",
                "mean",
            ),
            runs=("seed", "count"),
        )
        .reset_index()
    )
    # Backward-compatible plot aliases remain explicitly completed-trip/horizon metrics.
    summary["mean_time_loss_s"] = summary["mean_time_loss_s_completed"]
    summary["mean_unfinished"] = summary["mean_unfinished_vehicles"]
    summary_path = output / "traffic_summary.csv"
    summary.to_csv(summary_path, index=False)
    paired_deltas, paired_summary = _paired_outputs(frame)
    paired_deltas_path = output / "traffic_paired_deltas.csv"
    paired_summary_path = output / "traffic_paired_summary.csv"
    paired_deltas.to_csv(paired_deltas_path, index=False)
    paired_summary.to_csv(paired_summary_path, index=False)

    route_artifacts = {
        f"{scenario}__{seed}": {
            "path": _config_reference(path),
            "sha256": sha256_file(path),
        }
        for (scenario, seed), path in sorted(route_paths.items())
    }
    emission_classes = sorted(
        {
            emission
            for record in route_records.values()
            for emission in record.get("emission_classes", [])
        }
    )
    record = metadata(
        data_kind,
        config_all,
        schema_version="urbantwin.traffic-benchmark/v2",
        config_path=_config_reference(config_path),
        config_sha256=sha256_file(config_path),
        network=_config_reference(network),
        network_sha256=sha256_file(network),
        network_source="external_user_network" if external else "generated_abstract_junction",
        route_artifacts_by_scenario_seed=route_artifacts,
        warning=(
            "Simulator-only experiment. The network and demand are not established as a "
            "calibrated Cairo digital twin."
        ),
        objective=(
            "mean completed-trip time loss + 2*mean approach queue + "
            "2*(unfinished inserted + pending insertion); tuning only"
        ),
        equal_objective_evaluation_budget=evaluation_budget,
        sumo_version=installed_sumo_version,
        optimizer_source_sha256=optimizer_source_hash,
        checkpointing={
            "schema_version": CHECKPOINT_SCHEMA,
            "runner_contract": _RUNNER_CONTRACT,
            "runner_source_sha256": runner_source_hash,
            "signal_adapter_sha256": signal_adapter_hash,
            "checkpoint_root": _config_reference(checkpoint_root),
            "separate_from_final_artifacts": True,
            "resume_requested": bool(resume),
            "estimated_logical_executions": total_estimated,
            "completed_logical_executions": progress.completed,
            "physical_executions_this_invocation": progress.physical_executions,
            "cache_hits": progress.cache_hits,
            "lookup_status_counts": dict(sorted(progress.lookup_statuses.items())),
        },
        optimization_runtime_interpretation=(
            "Wall time for the current optimizer invocation; validated checkpoint hits can "
            "reduce this orchestration time on resume, so it is not a traffic-performance KPI."
        ),
        optimization=optimization,
        evaluation_signal_programs=evaluation_programs,
        track_emissions=track_emissions,
        emission_class_assumptions=emission_classes,
        co2_accounting=(
            "At each simulation step, sum TraCI getCO2Emission (mg/s) for active vehicles "
            "and multiply by the reported simulation step length; includes inserted vehicles "
            "while active and excludes demand still pending insertion."
        ),
        co2_interpretation=(
            "SUMO/HBEFA simulator estimate under the recorded emission classes; "
            "not measured Cairo air pollution"
            if track_emissions
            else "CO2 tracking disabled; no emission result was evaluated"
        ),
        metrics_sha256=sha256_file(metrics_path),
        summary_sha256=sha256_file(summary_path),
        paired_deltas_sha256=sha256_file(paired_deltas_path),
        paired_summary_sha256=sha256_file(paired_summary_path),
    )
    save_json(record, output / "traffic_provenance.json")
    return frame


__all__ = ["objective_metrics", "run_benchmark", "run_one"]
