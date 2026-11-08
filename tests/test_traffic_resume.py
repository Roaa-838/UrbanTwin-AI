from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml

from urbantwin.traffic import benchmark


def _configuration(path: Path) -> Path:
    path.write_text(
        yaml.safe_dump(
            {
                "random_seed": 42,
                "traffic": {
                    "simulation_seconds": 30,
                    "clearance_seconds": 5,
                    "warmup_seconds": 0,
                    "vehicle_rates_per_approach_per_hour": {"medium": 100},
                    "green_seconds_bounds": [12, 55],
                    "baseline_green_seconds": [30, 30],
                    "track_emissions": True,
                    "emission_class": "HBEFA3/PC_G_EU4",
                    "optimizer_training_seeds": [17],
                    "evaluation_seeds": [101],
                    "ga_population": 4,
                    "ga_generations": 1,
                    "pso_particles": 4,
                    "pso_iterations": 1,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def _result(plan: tuple[int, int], seed: int) -> dict[str, Any]:
    loss = float(plan[0] + plan[1]) / 10 + seed / 10_000
    return {
        "plan_ns_ew_s": [int(plan[0]), int(plan[1])],
        "generated_vehicles": 10,
        "inserted_vehicles": 9,
        "completed_vehicles": 8,
        "unfinished_vehicles": 1,
        "pending_vehicles": 1,
        "completion_fraction_generated": 0.8,
        "insertion_fraction_generated": 0.9,
        "completed_throughput_vehicles_per_sim_hour": 720.0,
        "mean_waiting_time_s_completed": loss / 2,
        "mean_time_loss_s_completed": loss,
        "mean_trip_duration_s_completed": loss + 10,
        "mean_approach_queue_vehicles": 1.0,
        "simulated_co2_g_inserted_fleet": 90.0,
        "simulated_co2_g_per_inserted_vehicle": 10.0,
        "simulated_co2_label": (
            "SUMO model estimate; simulator-only, not observed pollution"
        ),
        "wall_time_s": 0.01,
        "signal_program": {
            "tls_id": "tls0",
            "green_seconds": [int(plan[0]), int(plan[1])],
        },
        "queue_note": "Mean halted vehicles on controlled lanes",
        "censoring_note": "Completed-trip metrics expose horizon censoring",
    }


def _patch_external_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, str]:
    network = tmp_path / "fixture.net.xml"
    network.write_text("<net/>", encoding="utf-8")
    route_paths: dict[tuple[str, int], Path] = {}
    records: dict[tuple[str, int], dict[str, Any]] = {}
    for seed in (17, 101):
        route = tmp_path / f"medium_{seed}.rou.xml"
        route.write_text(f"<routes seed='{seed}'/>", encoding="utf-8")
        route_paths[("medium", seed)] = route
        records[("medium", seed)] = {
            "generated_vehicles": 10,
            "emission_classes": ["HBEFA3/PC_G_EU4"],
            "traffic_light_ids": ["tls0"],
        }

    def validate(*_args, **_kwargs):
        return network, route_paths, records

    monkeypatch.setattr(benchmark, "validate_external_inputs", validate)
    monkeypatch.setattr(benchmark, "_sumo_version", lambda: "Eclipse SUMO 1.26.0")
    monkeypatch.setattr(benchmark, "_simulation_runner_source_sha256", lambda: "f" * 64)
    return network, str(tmp_path / "{scenario}_{seed}.rou.xml")


def _scientific_columns(frame: pd.DataFrame) -> pd.DataFrame:
    ignored = {"optimization_wall_time_s"}
    return frame[[column for column in frame if column not in ignored]].sort_values(
        ["scenario", "method", "seed"]
    ).reset_index(drop=True)


def test_interrupted_benchmark_resumes_without_duplicate_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    config = _configuration(tmp_path / "experiments.yaml")
    network, template = _patch_external_inputs(tmp_path, monkeypatch)
    checkpoints = tmp_path / "checkpoints"
    output = tmp_path / "results"
    attempted = 0

    def interrupting_run(_network, _routes, plan, seed, *_args, **_kwargs):
        nonlocal attempted
        attempted += 1
        if attempted == 3:
            raise KeyboardInterrupt
        return _result(plan, seed)

    monkeypatch.setattr(benchmark, "run_one", interrupting_run)
    with pytest.raises(KeyboardInterrupt):
        benchmark.run_benchmark(
            config,
            output,
            str(network),
            template,
            "tls0",
            resume=True,
            checkpoint_dir=checkpoints,
        )
    assert len(list(checkpoints.rglob("*.json"))) == 2
    assert not (output / "traffic_metrics.csv").exists()

    resumed_physical = 0

    def completing_run(_network, _routes, plan, seed, *_args, **_kwargs):
        nonlocal resumed_physical
        resumed_physical += 1
        return _result(plan, seed)

    monkeypatch.setattr(benchmark, "run_one", completing_run)
    first = benchmark.run_benchmark(
        config,
        output,
        str(network),
        template,
        "tls0",
        resume=True,
        checkpoint_dir=checkpoints,
    )
    assert resumed_physical > 0
    assert len(first) == 4
    assert not first.duplicated(["scenario", "method", "seed", "data_kind"]).any()

    def unexpected_run(*_args, **_kwargs):
        raise AssertionError("A completed compatible simulation was run again")

    monkeypatch.setattr(benchmark, "run_one", unexpected_run)
    second = benchmark.run_benchmark(
        config,
        output,
        str(network),
        template,
        "tls0",
        resume=True,
        checkpoint_dir=checkpoints,
    )
    pd.testing.assert_frame_equal(_scientific_columns(first), _scientific_columns(second))
    assert len(second) == 4
    progress = capsys.readouterr().out
    assert "completed_executions=16/16" in progress
    assert "cache_hits=" in progress
    assert "objective_evaluation=" in progress


def test_without_resume_recomputes_instead_of_using_prior_process_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    config = _configuration(tmp_path / "experiments.yaml")
    network, template = _patch_external_inputs(tmp_path, monkeypatch)
    checkpoints = tmp_path / "checkpoints"
    calls = 0

    def counted_run(_network, _routes, plan, seed, *_args, **_kwargs):
        nonlocal calls
        calls += 1
        return _result(plan, seed)

    monkeypatch.setattr(benchmark, "run_one", counted_run)
    for resume in (True, False):
        before = calls
        benchmark.run_benchmark(
            config,
            tmp_path / f"results_{resume}",
            str(network),
            template,
            "tls0",
            resume=resume,
            checkpoint_dir=checkpoints,
        )
        assert calls > before

