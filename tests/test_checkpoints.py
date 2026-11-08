import json

import pytest

from urbantwin.traffic.checkpoints import CheckpointStore, SimulationIdentity


def _identity(config_hash: str = "a" * 64) -> SimulationIdentity:
    return SimulationIdentity(
        config_sha256=config_hash,
        network_sha256="b" * 64,
        route_sha256="c" * 64,
        sumo_version="Eclipse SUMO 1.26.0",
        scenario="medium",
        seed=142,
        signal_plan={"tls_id": "J0", "green_seconds": [30, 35]},
        emission_settings={
            "track_emissions": True,
            "emission_classes": ["HBEFA3/PC_G_EU4"],
        },
        simulation_settings={
            "simulation_seconds": 120,
            "clearance_seconds": 30,
            "warmup_seconds": 10,
            "step_length_s": 1.0,
        },
    )


def _result() -> dict:
    return {
        "plan_ns_ew_s": [30, 35],
        "generated_vehicles": 10,
        "inserted_vehicles": 9,
        "completed_vehicles": 7,
        "unfinished_vehicles": 2,
        "pending_vehicles": 1,
        "completion_fraction_generated": 0.7,
        "insertion_fraction_generated": 0.9,
        "completed_throughput_vehicles_per_sim_hour": 168.0,
        "mean_waiting_time_s_completed": 2.5,
        "mean_time_loss_s_completed": 3.5,
        "mean_trip_duration_s_completed": 12.0,
        "mean_approach_queue_vehicles": 1.25,
        "simulated_co2_g_inserted_fleet": 45.0,
        "simulated_co2_g_per_inserted_vehicle": 5.0,
        "simulated_co2_label": (
            "SUMO model estimate; simulator-only, not observed pollution"
        ),
        "wall_time_s": 0.25,
        "signal_program": {"tls_id": "J0", "green_phase_indices": [0, 2]},
        "queue_note": "Mean halted approach vehicles after warmup",
        "censoring_note": "Completed-trip metrics expose horizon censoring",
    }


def test_identity_and_serialization_are_deterministic(tmp_path):
    reordered = SimulationIdentity(
        config_sha256="A" * 64,
        network_sha256="b" * 64,
        route_sha256="c" * 64,
        sumo_version="Eclipse SUMO 1.26.0",
        scenario="medium",
        seed=142,
        signal_plan={"green_seconds": [30, 35], "tls_id": "J0"},
        emission_settings={
            "emission_classes": ["HBEFA3/PC_G_EU4"],
            "track_emissions": True,
        },
        simulation_settings={
            "warmup_seconds": 10,
            "step_length_s": 1.0,
            "clearance_seconds": 30,
            "simulation_seconds": 120,
        },
    )
    identity = _identity()
    assert identity.fingerprint == reordered.fingerprint

    store = CheckpointStore(tmp_path / "checkpoints")
    path = store.save(identity, _result())
    lookup = store.lookup(identity)
    assert lookup.hit
    assert lookup.result == _result()
    assert json.loads(path.read_text(encoding="utf-8"))["state"] == "complete"
    assert not list(path.parent.glob("*.tmp"))


def test_deterministic_cache_reuse_computes_only_once(tmp_path):
    store = CheckpointStore(tmp_path / "checkpoints")
    calls = 0

    def compute():
        nonlocal calls
        calls += 1
        return _result()

    first = store.get_or_compute(_identity(), compute)
    second = store.get_or_compute(_identity(), compute)
    assert not first.cache_hit
    assert second.cache_hit
    assert first.result == second.result
    assert calls == 1
    assert len(list((tmp_path / "checkpoints").rglob("*.json"))) == 1


def test_explicit_fresh_save_atomically_replaces_older_payload(tmp_path):
    store = CheckpointStore(tmp_path / "checkpoints")
    identity = _identity()
    original = _result()
    refreshed = _result()
    refreshed["wall_time_s"] = 0.75
    refreshed["mean_time_loss_s_completed"] = 4.25

    store.save(identity, original)
    store.save(identity, refreshed)
    assert store.lookup(identity).result == original

    store.save(identity, refreshed, overwrite=True)
    assert store.lookup(identity).result == refreshed
    assert len(list((tmp_path / "checkpoints").rglob("*.json"))) == 1


def test_interruption_leaves_no_partial_checkpoint(tmp_path):
    store = CheckpointStore(tmp_path / "checkpoints")

    def interrupted():
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        store.get_or_compute(_identity(), interrupted)
    assert store.lookup(_identity()).status == "missing"
    assert not list((tmp_path / "checkpoints").rglob("*"))


def test_corrupt_checkpoint_is_detected_and_recomputed(tmp_path):
    store = CheckpointStore(tmp_path / "checkpoints")
    identity = _identity()
    path = store.path_for(identity)
    path.parent.mkdir(parents=True)
    path.write_text('{"state":', encoding="utf-8")
    assert store.lookup(identity).status == "corrupt"

    outcome = store.get_or_compute(identity, _result)
    assert not outcome.cache_hit
    assert outcome.prior_status == "corrupt"
    assert store.lookup(identity).hit


def test_incompatible_config_checkpoint_is_rejected(tmp_path):
    store = CheckpointStore(tmp_path / "checkpoints")
    original = _identity()
    store.save(original, _result())
    incompatible = _identity("d" * 64)
    wrong_path = store.path_for(incompatible)
    wrong_path.parent.mkdir(parents=True)
    wrong_path.write_bytes(store.path_for(original).read_bytes())

    lookup = store.lookup(incompatible)
    assert lookup.status == "incompatible"
    outcome = store.get_or_compute(incompatible, _result)
    assert not outcome.cache_hit
    assert outcome.prior_status == "incompatible"
    assert store.lookup(incompatible).hit
    assert len(list((tmp_path / "checkpoints").rglob("*.json"))) == 2


def test_invalid_or_incomplete_results_are_never_saved(tmp_path):
    store = CheckpointStore(tmp_path / "checkpoints")
    invalid = _result()
    invalid["unfinished_vehicles"] = 3
    with pytest.raises(ValueError, match="unfinished_vehicles"):
        store.save(_identity(), invalid)
    assert store.lookup(_identity()).status == "missing"
