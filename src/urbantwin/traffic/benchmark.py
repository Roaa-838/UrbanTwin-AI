"""Run controlled SUMO comparisons using consistent scenarios and held-out seeds.

The built-in simulation is a toy cross-junction, *not* a calibrated model of Cairo.
If a real SUMOFlow network is provided, users must supply compatible routes,
and mapping for its traffic-light phase structure.
"""
from __future__ import annotations
import json
import time
import tempfile
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd
from urbantwin.common import load_yaml,metadata,save_json
from urbantwin.traffic.network import binary,create_network,create_routes
from urbantwin.traffic.baselines import install_tuned_program
from urbantwin.traffic.optimize import genetic_algorithm,particle_swarm


def _load_traffic_lib():
    try:import traci
    except ImportError as exc:raise RuntimeError("Install SUMO Python support: pip install -e '.[traffic]'") from exc
    binary("sumo");binary("netconvert")
    return traci


def run_one(network: Path,routes: Path,plan: tuple[int,int],seed: int,
            config: dict, tls_id: str | None=None) -> dict:
    traci=_load_traffic_lib()
    t0=time.monotonic()
    with tempfile.TemporaryDirectory(prefix="urbantwin_sumo_") as tmp:
        trip=Path(tmp)/"tripinfos.xml"
        horizon=int(config["simulation_seconds"])+int(config["clearance_seconds"])
        cmd=[binary("sumo"),"--net-file",str(network.resolve()),"--route-files",str(routes.resolve()),
             "--tripinfo-output",str(trip),"--no-step-log","true","--duration-log.disable","true",
             "--time-to-teleport","-1","--seed",str(seed)]
        started=False
        try:
            traci.start(cmd)
            started=True
            ids=list(traci.trafficlight.getIDList())
            signal=tls_id or (ids[0] if ids else None)
            if not signal:raise RuntimeError("No traffic light in SUMO network")
            plan_metadata=install_tuned_program(traci,signal,plan)
            queue_sum=0.;q_observations=0
            co2_mg=0.;arrivals=0;departures=0
            lanes=sorted(set(l for l in traci.trafficlight.getControlledLanes(signal)
                             if l and not l.startswith(":")))
            if not lanes:
                raise ValueError(f"TLS {signal} controls no physical approach lanes")
            for step in range(horizon):
                traci.simulationStep()
                if step>=int(config.get("warmup_seconds",0)):
                    queue_sum+=sum(traci.lane.getLastStepHaltingNumber(lane) for lane in lanes)
                    q_observations+=1
                arrivals+=traci.simulation.getArrivedNumber()
                departures+=traci.simulation.getDepartedNumber()
                if bool(config.get("track_emissions",False)):
                    co2_mg+=sum(traci.vehicle.getCO2Emission(veh) for veh in traci.vehicle.getIDList())
            unfinished=max(0,departures-arrivals)
        finally:
            if started:traci.close()
        if not trip.exists():raise RuntimeError("SUMO did not write tripinfo output")
        root=ET.parse(trip).getroot()
        # Only completed trips belong to the following mean waiting/delay measures.
        trips=[x for x in root.findall("tripinfo") if "timeLoss" in x.attrib]
        waiting=[float(x.attrib["waitingTime"]) for x in trips if "waitingTime" in x.attrib]
        delay=[float(x.attrib["timeLoss"]) for x in trips]
        duration=[float(x.attrib["duration"]) for x in trips if "duration" in x.attrib]
    result={
        "plan_ns_ew_s":[int(plan[0]),int(plan[1])],
        "completed_trips":len(trips),"departed_vehicles":departures,"unfinished_vehicles":unfinished,
        "throughput_vehicles_per_hour":len(trips)*3600/horizon,
        "mean_waiting_time_s_completed":float(np.mean(waiting)) if waiting else None,
        "mean_time_loss_s_completed":float(np.mean(delay)) if delay else None,
        "mean_trip_duration_s_completed":float(np.mean(duration)) if duration else None,
        "mean_approach_queue_vehicles":queue_sum/q_observations if q_observations else None,
        "simulated_co2_g":co2_mg/1000 if bool(config.get("track_emissions",False)) else None,
        "wall_time_s":time.monotonic()-t0,"signal_program":plan_metadata,
        "queue_note":"Mean halted vehicles on unique TLS-controlled approach lanes after warmup; not maximum queue length",
        "censoring_note":"Time-loss/waiting means are for completed trips only; unfinished count reported separately",
    }
    return result


def objective_metrics(row: dict) -> float:
    if not row["completed_trips"]:return 1e6
    # Explicit arbitrary scalarization for tuning only; report independent metrics.
    return float(row["mean_time_loss_s_completed"]+2*row["mean_approach_queue_vehicles"]+2*row["unfinished_vehicles"])


def run_benchmark(config_file: str | Path,out_dir: str | Path,network_file: str | None=None,
                  routes_template: str | None=None, tls_id: str | None=None) -> pd.DataFrame:
    cfg_all=load_yaml(config_file)
    cfg=cfg_all["traffic"]
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    network=Path(network_file).resolve() if network_file else create_network(out/"synthetic_network")
    if not network.exists():raise FileNotFoundError(network)
    scenarios=cfg["vehicle_rates_per_approach_per_hour"]
    train_seeds=list(cfg["optimizer_training_seeds"])
    test_seeds=list(cfg["evaluation_seeds"])
    if set(train_seeds)&set(test_seeds):raise ValueError("Optimizer training seeds must not overlap evaluation seeds")
    all_seeds=set(train_seeds+test_seeds)
    route_paths={}
    for scenario,rate in scenarios.items():
        for seed in all_seeds:
            if routes_template:
                file=Path(routes_template.format(scenario=scenario,seed=seed)).resolve()
                if not file.exists():raise FileNotFoundError(file)
            else:
                file=create_routes(out/"synthetic_network"/f"routes_{scenario}_{seed}.rou.xml",
                                   float(rate),int(cfg["simulation_seconds"]),int(seed))
            route_paths[(scenario,seed)]=file
    if network_file and not routes_template:
        raise ValueError("For external SUMO network, also supply compatible routes_template")
    bounds=tuple(int(x) for x in cfg["green_seconds_bounds"])
    base=tuple(int(x) for x in cfg["baseline_green_seconds"])
    comparison=[];optimization={}
    for scenario,rate in scenarios.items():
        @lru_cache(maxsize=256)
        def score(plan):
            measures=[run_one(network,route_paths[(scenario,s)],plan,s,cfg,tls_id) for s in train_seeds]
            return float(np.mean([objective_metrics(x) for x in measures]))
        t0=time.monotonic()
        ga=genetic_algorithm(score,bounds,population=int(cfg["ga_population"]),generations=int(cfg["ga_generations"]),seed=42)
        ga_seconds=time.monotonic()-t0
        t0=time.monotonic()
        pso=particle_swarm(score,bounds,particles=int(cfg["pso_particles"]),iterations=int(cfg["pso_iterations"]),seed=43)
        pso_seconds=time.monotonic()-t0
        optimization[scenario]={"GA":{**ga,"optimization_wall_time_s":ga_seconds},
                                "PSO":{**pso,"optimization_wall_time_s":pso_seconds},
                                "training_seeds":train_seeds,"evaluation_seeds":test_seeds}
        for method,plan,optimizer_seconds in [("fixed",base,0.),("GA",tuple(ga["plan"]),ga_seconds),("PSO",tuple(pso["plan"]),pso_seconds)]:
            for s in test_seeds:
                r=run_one(network,route_paths[(scenario,s)],plan,int(s),cfg,tls_id)
                comparison.append({"scenario":scenario,"demand_vph_per_approach":rate,
                    "method":method,"seed":s,"train_seed_overlap":False,
                    "green_ns_s":plan[0],"green_ew_s":plan[1],
                    "data_kind":"simulation_abstract" if not network_file else "simulation_external_user_network",
                    "optimization_wall_time_s":optimizer_seconds,
                    **{k:v for k,v in r.items() if not isinstance(v,(list,dict))}})
    frame=pd.DataFrame(comparison)
    frame.to_csv(out/"traffic_metrics.csv",index=False)
    summary=(frame.groupby(["scenario","method"],sort=True)
             .agg(mean_waiting_s=("mean_waiting_time_s_completed","mean"),
                  std_waiting_s=("mean_waiting_time_s_completed","std"),
                  mean_time_loss_s=("mean_time_loss_s_completed","mean"),
                  mean_queue_vehicles=("mean_approach_queue_vehicles","mean"),
                  mean_throughput_vph=("throughput_vehicles_per_hour","mean"),
                  mean_unfinished=("unfinished_vehicles","mean"),
                  runs=("seed","count")).reset_index())
    summary.to_csv(out/"traffic_summary.csv",index=False)
    save_json(metadata("simulation_abstract" if not network_file else "simulation_external_user_network",
                       cfg_all,network=str(network),
                       warning="Synthetic intersection is not a geometrically or demand-calibrated Cairo traffic model.",
                       objective="mean time loss completed + 2*mean queue + 2*unfinished count",
                       optimization=optimization),out/"traffic_provenance.json")
    return frame
