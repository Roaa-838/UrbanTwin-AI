"""Control an existing SUMO traffic signal using two green-phase durations."""
from __future__ import annotations


def green_phase_indices(phases) -> list[int]:
    # For a simple two-stage junction. Safely refuse more elaborate signal plans.
    return [i for i,p in enumerate(phases)
            if ("G" in p.state or "g" in p.state) and not ("y" in p.state or "Y" in p.state)]


def install_tuned_program(traci, tls_id: str, green_seconds: tuple[int,int]):
    programs=traci.trafficlight.getAllProgramLogics(tls_id)
    if not programs:
        raise ValueError(f"No traffic program for TLS '{tls_id}'")
    logic=next((x for x in programs if x.programID==traci.trafficlight.getProgram(tls_id)),programs[0])
    phases=logic.phases
    indices=green_phase_indices(phases)
    if len(indices)!=2:
        raise ValueError(f"Expected exactly 2 main green phases for this simplified controller; found {len(indices)}. "
                         "Use an appropriate adapter/phase mapping for complex SUMOFlow networks.")
    updated=[]
    for i,p in enumerate(phases):
        duration=int(green_seconds[indices.index(i)]) if i in indices else float(p.duration)
        updated.append(traci.trafficlight.Phase(duration=duration,state=p.state,
                                                minDur=duration,maxDur=duration,
                                                name=getattr(p,"name","")))
    tuned=traci.trafficlight.Logic(programID="urbantwin_fixed_two_green",type=0,
                                   currentPhaseIndex=0,phases=updated)
    traci.trafficlight.setProgramLogic(tls_id,tuned)
    traci.trafficlight.setProgram(tls_id,tuned.programID)
    return {"tls_id":tls_id,"phase_indices":indices,"all_phase_durations_s":[p.duration for p in updated]}
