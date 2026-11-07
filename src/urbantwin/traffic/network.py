"""Generate an *abstract*, one-junction SUMO simulation. Not a Cairo digital twin.

SUMOFlow network integration is possible via external_network/external_routes_template
in experiments.yaml if routes and traffic-light programming are compatible.
"""
from __future__ import annotations
import subprocess
import shutil
import os
from pathlib import Path
import numpy as np
import xml.etree.ElementTree as ET


def binary(name: str) -> str:
    path=shutil.which(name)
    if not path and os.getenv("SUMO_HOME"):
        exe=name+(".exe" if os.name=="nt" else "")
        path=str(Path(os.environ["SUMO_HOME"])/"bin"/exe)
    if not path or not Path(path).exists():
        raise RuntimeError(f"SUMO binary '{name}' not found. Install Eclipse SUMO and put its bin directory on PATH")
    return path


def create_network(out_dir: str | Path) -> Path:
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    nodes=ET.Element("nodes")
    for ident,x,y,t in [("C",0,0,"traffic_light"),("N",0,250,"priority"),
                        ("S",0,-250,"priority"),("E",250,0,"priority"),
                        ("W",-250,0,"priority")]:
        ET.SubElement(nodes,"node",id=ident,x=str(x),y=str(y),type=t)
    ET.ElementTree(nodes).write(out/"junction.nod.xml",encoding="utf-8",xml_declaration=True)
    edges=ET.Element("edges")
    for side in ["N","S","E","W"]:
        ET.SubElement(edges,"edge",id=f"{side}_C",**{"from":side,"to":"C"},numLanes="1",speed="13.89")
        ET.SubElement(edges,"edge",id=f"C_{side}",**{"from":"C","to":side},numLanes="1",speed="13.89")
    ET.ElementTree(edges).write(out/"junction.edg.xml",encoding="utf-8",xml_declaration=True)
    net=out/"junction.net.xml"
    cmd=[binary("netconvert"),"--node-files",str(out/"junction.nod.xml"),
         "--edge-files",str(out/"junction.edg.xml"),"--output-file",str(net),
         "--no-turnarounds","true"]
    result=subprocess.run(cmd,capture_output=True,text=True,check=False)
    if result.returncode!=0 or not net.exists():
        raise RuntimeError(f"netconvert failed: {result.stderr[-2500:]}")
    return net


def create_routes(out: str | Path, rate_hour: float, sim_seconds: int, seed: int) -> Path:
    if rate_hour<=0: raise ValueError("Rate must be >0")
    path=Path(out);path.parent.mkdir(parents=True,exist_ok=True)
    root=ET.Element("routes")
    ET.SubElement(root,"vType",id="car",vClass="passenger",accel="2.6",decel="4.5",
                  sigma="0.5",length="5",minGap="2.5",maxSpeed="13.89")
    routes={"N_S":"N_C C_S","S_N":"S_C C_N","E_W":"E_C C_W","W_E":"W_C C_E"}
    for rid,ed in routes.items():ET.SubElement(root,"route",id=rid,edges=ed)
    rng=np.random.default_rng(seed)
    arrivals=[]
    for rid in routes:
        n=int(rng.poisson(rate_hour*sim_seconds/3600))
        arrivals.extend((float(t),rid,j) for j,t in enumerate(rng.uniform(0,sim_seconds,size=n)))
    for seq,(t,rid,j) in enumerate(sorted(arrivals)):
        ET.SubElement(root,"vehicle",id=f"v_{seed}_{seq}",type="car",route=rid,
                      depart=f"{t:.2f}",departLane="best",departSpeed="max")
    ET.ElementTree(root).write(path,encoding="utf-8",xml_declaration=True)
    return path
