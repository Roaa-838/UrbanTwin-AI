"""Durable, authoritative cache records for completed SUMO simulations.

Checkpoint identity describes simulator inputs, never an optimizer or reporting
method.  Consequently, two methods that select the same signal plan can safely
reuse one simulation.  Final benchmark CSV/provenance artifacts remain a
separate concern and must be regenerated from validated checkpoint payloads.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

CHECKPOINT_SCHEMA = "urbantwin.sumo-simulation-checkpoint/v1"
_MAX_CHECKPOINT_BYTES = 10 * 1024 * 1024
_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")


def _canonical_json(value: Any) -> str:
    """Return the stable JSON representation used for keys and checksums."""
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Checkpoint values must be finite, JSON-serializable data") from exc


def _json_copy(value: Any) -> Any:
    return json.loads(_canonical_json(value))


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _validated_sha256(name: str, value: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a 64-character SHA-256 hexadecimal digest")
    return value.lower()


@dataclass(frozen=True)
class SimulationIdentity:
    """All inputs that can change one SUMO simulation's reproducible result."""

    config_sha256: str
    network_sha256: str
    route_sha256: str
    sumo_version: str
    scenario: str
    seed: int
    signal_plan: Mapping[str, Any]
    emission_settings: Mapping[str, Any]
    simulation_settings: Mapping[str, Any]
    _identity_json: str = field(init=False, repr=False, compare=False)
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        config_hash = _validated_sha256("config_sha256", self.config_sha256)
        network_hash = _validated_sha256("network_sha256", self.network_sha256)
        route_hash = _validated_sha256("route_sha256", self.route_sha256)
        if not isinstance(self.sumo_version, str) or not self.sumo_version.strip():
            raise ValueError("sumo_version must be a non-empty version string")
        if not isinstance(self.scenario, str) or not self.scenario.strip():
            raise ValueError("scenario must be a non-empty string")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        for name, value in (
            ("signal_plan", self.signal_plan),
            ("emission_settings", self.emission_settings),
            ("simulation_settings", self.simulation_settings),
        ):
            if not isinstance(value, Mapping):
                raise TypeError(f"{name} must be a mapping")
            if name != "emission_settings" and not value:
                raise ValueError(f"{name} must not be empty")

        payload = {
            "config_sha256": config_hash,
            "network_sha256": network_hash,
            "route_sha256": route_hash,
            "sumo_version": self.sumo_version.strip(),
            "scenario": self.scenario.strip(),
            "seed": self.seed,
            "signal_plan": dict(self.signal_plan),
            "emission_settings": dict(self.emission_settings),
            "simulation_settings": dict(self.simulation_settings),
        }
        identity_json = _canonical_json(payload)
        object.__setattr__(self, "config_sha256", config_hash)
        object.__setattr__(self, "network_sha256", network_hash)
        object.__setattr__(self, "route_sha256", route_hash)
        object.__setattr__(self, "sumo_version", self.sumo_version.strip())
        object.__setattr__(self, "scenario", self.scenario.strip())
        object.__setattr__(self, "_identity_json", identity_json)
        object.__setattr__(
            self,
            "fingerprint",
            hashlib.sha256(identity_json.encode("utf-8")).hexdigest(),
        )

    def as_dict(self) -> dict[str, Any]:
        """Return a detached copy of the canonical identity payload."""
        return json.loads(self._identity_json)


LookupStatus = Literal["hit", "missing", "corrupt", "incompatible"]


@dataclass(frozen=True)
class CheckpointLookup:
    status: LookupStatus
    path: Path
    result: dict[str, Any] | None = None
    reason: str | None = None

    @property
    def hit(self) -> bool:
        return self.status == "hit"


@dataclass(frozen=True)
class CheckpointOutcome:
    result: dict[str, Any]
    cache_hit: bool
    checkpoint_path: Path
    prior_status: LookupStatus


def _finite_number(name: str, value: Any, *, allow_none: bool = False) -> float | None:
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"Checkpoint result {name} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Checkpoint result {name} must be finite")
    return number


def _validate_result(result: Any) -> dict[str, Any]:
    if not isinstance(result, dict):
        raise TypeError("Checkpoint result must be a JSON object")
    normalized = _json_copy(result)
    plan = normalized.get("plan_ns_ew_s")
    if (
        not isinstance(plan, list)
        or len(plan) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in plan)
    ):
        raise ValueError("Checkpoint result plan_ns_ew_s must contain two positive integers")
    required_counts = (
        "generated_vehicles",
        "inserted_vehicles",
        "completed_vehicles",
        "unfinished_vehicles",
        "pending_vehicles",
    )
    counts: dict[str, int] = {}
    for name in required_counts:
        value = normalized.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"Checkpoint result {name} must be a non-negative integer")
        counts[name] = value
    generated = counts["generated_vehicles"]
    inserted = counts["inserted_vehicles"]
    completed = counts["completed_vehicles"]
    if generated <= 0 or not 0 <= completed <= inserted <= generated:
        raise ValueError("Checkpoint result has inconsistent vehicle accounting")
    if counts["unfinished_vehicles"] != inserted - completed:
        raise ValueError("Checkpoint result unfinished_vehicles is inconsistent")
    if counts["pending_vehicles"] != generated - inserted:
        raise ValueError("Checkpoint result pending_vehicles is inconsistent")

    completion = _finite_number(
        "completion_fraction_generated", normalized.get("completion_fraction_generated")
    )
    insertion = _finite_number(
        "insertion_fraction_generated", normalized.get("insertion_fraction_generated")
    )
    if not math.isclose(float(completion), completed / generated, abs_tol=1e-12):
        raise ValueError("Checkpoint completion fraction is inconsistent")
    if not math.isclose(float(insertion), inserted / generated, abs_tol=1e-12):
        raise ValueError("Checkpoint insertion fraction is inconsistent")

    for name in (
        "completed_throughput_vehicles_per_sim_hour",
        "mean_waiting_time_s_completed",
        "mean_time_loss_s_completed",
        "mean_trip_duration_s_completed",
        "mean_approach_queue_vehicles",
        "simulated_co2_g_inserted_fleet",
        "simulated_co2_g_per_inserted_vehicle",
    ):
        value = _finite_number(name, normalized.get(name), allow_none=True)
        if value is not None and value < 0:
            raise ValueError(f"Checkpoint result {name} must be non-negative")
    wall_time = _finite_number("wall_time_s", normalized.get("wall_time_s"))
    if wall_time is None or wall_time < 0:
        raise ValueError("Checkpoint result wall_time_s must be non-negative")
    if not isinstance(normalized.get("signal_program"), dict):
        raise TypeError("Checkpoint result must include the installed signal_program")
    expected_label = "SUMO model estimate; simulator-only, not observed pollution"
    if normalized.get("simulated_co2_label") != expected_label:
        raise ValueError("Checkpoint result has an unsupported simulated CO2 label")
    for name in ("queue_note", "censoring_note"):
        if not isinstance(normalized.get(name), str) or not normalized[name]:
            raise ValueError(f"Checkpoint result must include {name}")
    return normalized


def _fsync_directory(path: Path) -> None:
    """Best-effort directory sync; Windows does not expose portable directory fsync."""
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json_atomic_durable(payload: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        indent=2,
    ) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


class CheckpointStore:
    """Validate and persist one complete checkpoint per simulation identity."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.simulations_dir = self.root / "simulations"

    def path_for(self, identity: SimulationIdentity) -> Path:
        return (
            self.simulations_dir
            / identity.fingerprint[:2]
            / f"{identity.fingerprint}.json"
        )

    def lookup(self, identity: SimulationIdentity) -> CheckpointLookup:
        path = self.path_for(identity)
        if not path.is_file():
            return CheckpointLookup("missing", path, reason="checkpoint does not exist")
        try:
            if path.stat().st_size > _MAX_CHECKPOINT_BYTES:
                raise ValueError("checkpoint exceeds the safe size limit")
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            return CheckpointLookup("corrupt", path, reason=str(exc))
        if not isinstance(record, dict):
            return CheckpointLookup("corrupt", path, reason="checkpoint root is not an object")
        if record.get("schema_version") != CHECKPOINT_SCHEMA:
            return CheckpointLookup("incompatible", path, reason="unsupported checkpoint schema")
        if record.get("state") != "complete":
            return CheckpointLookup("corrupt", path, reason="checkpoint is not complete")
        if record.get("fingerprint") != identity.fingerprint:
            return CheckpointLookup("incompatible", path, reason="fingerprint mismatch")
        if record.get("identity") != identity.as_dict():
            return CheckpointLookup("incompatible", path, reason="simulation identity mismatch")
        created = record.get("completed_utc")
        try:
            parsed = datetime.fromisoformat(str(created).replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                raise ValueError("timestamp is not timezone-aware")
        except ValueError as exc:
            return CheckpointLookup("corrupt", path, reason=f"invalid completed_utc: {exc}")
        try:
            result = _validate_result(record.get("result"))
        except (TypeError, ValueError) as exc:
            return CheckpointLookup("corrupt", path, reason=str(exc))
        expected_plan = identity.as_dict()["signal_plan"].get("green_seconds")
        if result["plan_ns_ew_s"] != expected_plan:
            return CheckpointLookup(
                "incompatible",
                path,
                reason="checkpoint result signal plan does not match its identity",
            )
        tracks_emissions = identity.as_dict()["emission_settings"].get("track_emissions")
        co2_value = result["simulated_co2_g_inserted_fleet"]
        if tracks_emissions is True and co2_value is None:
            return CheckpointLookup(
                "incompatible",
                path,
                reason="checkpoint is missing requested emissions output",
            )
        if tracks_emissions is False and co2_value is not None:
            return CheckpointLookup(
                "incompatible",
                path,
                reason="checkpoint contains emissions although tracking was disabled",
            )
        if record.get("result_sha256") != _sha256_json(result):
            return CheckpointLookup("corrupt", path, reason="result checksum mismatch")
        return CheckpointLookup("hit", path, result=result)

    def save(
        self,
        identity: SimulationIdentity,
        result: dict[str, Any],
        *,
        overwrite: bool = False,
    ) -> Path:
        """Atomically save a complete result; never create duplicate identity files.

        ``overwrite`` is reserved for an explicitly fresh benchmark invocation.
        Resume paths retain an already validated record, while a requested
        recomputation replaces that same fingerprint atomically so a later
        resume cannot silently return an older payload.
        """
        normalized = _validate_result(result)
        existing = self.lookup(identity)
        if existing.hit and not overwrite:
            return existing.path
        record = {
            "schema_version": CHECKPOINT_SCHEMA,
            "state": "complete",
            "fingerprint": identity.fingerprint,
            "identity": identity.as_dict(),
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "result_sha256": _sha256_json(normalized),
            "result": normalized,
        }
        path = self.path_for(identity)
        _write_json_atomic_durable(record, path)
        verified = self.lookup(identity)
        if not verified.hit:
            raise OSError(
                f"Checkpoint did not validate after durable write: {verified.reason or verified.status}"
            )
        return path

    def get_or_compute(
        self,
        identity: SimulationIdentity,
        compute: Callable[[], dict[str, Any]],
    ) -> CheckpointOutcome:
        """Reuse a valid checkpoint, otherwise compute then durably save one.

        ``BaseException`` is deliberately not caught.  In particular, Ctrl+C
        propagates while every checkpoint completed before it remains intact,
        and an interrupted computation never produces a partial checkpoint.
        """
        lookup = self.lookup(identity)
        if lookup.hit:
            assert lookup.result is not None
            return CheckpointOutcome(lookup.result, True, lookup.path, lookup.status)
        result = compute()
        path = self.save(identity, result)
        return CheckpointOutcome(_validate_result(result), False, path, lookup.status)


__all__ = [
    "CHECKPOINT_SCHEMA",
    "CheckpointLookup",
    "CheckpointOutcome",
    "CheckpointStore",
    "SimulationIdentity",
]
