"""Adapt a compatible SUMO signal while preserving non-green phases."""
from __future__ import annotations

from typing import Any


def green_phase_indices(phases) -> list[int]:
    """Return pure-green phases for the deliberately simple two-stage adapter."""
    return [
        index
        for index, phase in enumerate(phases)
        if ("G" in phase.state or "g" in phase.state)
        and not ("y" in phase.state or "Y" in phase.state)
    ]


def install_tuned_program(traci: Any, tls_id: str, green_seconds: tuple[int, int]) -> dict:
    durations = tuple(int(value) for value in green_seconds)
    if len(durations) != 2 or any(value <= 0 for value in durations):
        raise ValueError("Exactly two positive green durations are required")
    programs = traci.trafficlight.getAllProgramLogics(tls_id)
    if not programs:
        raise ValueError(f"No traffic program for TLS {tls_id!r}")
    current_id = traci.trafficlight.getProgram(tls_id)
    logic = next((program for program in programs if program.programID == current_id), programs[0])
    phases = logic.phases
    indices = green_phase_indices(phases)
    if len(indices) != 2:
        raise ValueError(
            "Expected exactly 2 pure-green phases for this simplified controller; "
            f"found {len(indices)}. Use a verified phase adapter for complex networks."
        )
    updated = []
    original_durations = [float(phase.duration) for phase in phases]
    for index, phase in enumerate(phases):
        duration = durations[indices.index(index)] if index in indices else float(phase.duration)
        updated.append(
            traci.trafficlight.Phase(
                duration=duration,
                state=phase.state,
                minDur=duration,
                maxDur=duration,
                name=getattr(phase, "name", ""),
            )
        )
    tuned = traci.trafficlight.Logic(
        programID="urbantwin_fixed_two_green",
        type=0,
        currentPhaseIndex=0,
        phases=updated,
    )
    traci.trafficlight.setProgramLogic(tls_id, tuned)
    traci.trafficlight.setProgram(tls_id, tuned.programID)
    non_green = [index for index in range(len(phases)) if index not in indices]
    return {
        "tls_id": tls_id,
        "source_program_id": current_id,
        "green_phase_indices": indices,
        "original_phase_durations_s": original_durations,
        "tuned_phase_durations_s": [float(phase.duration) for phase in updated],
        "preserved_non_green_phase_indices": non_green,
        "yellow_configuration_note": (
            "Yellow and every other non-green phase keep the network program's existing duration; "
            "no separate yellow_seconds setting is applied."
        ),
    }
