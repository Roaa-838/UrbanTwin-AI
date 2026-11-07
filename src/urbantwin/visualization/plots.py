"""Figure generation from experiment CSV files; never invent measurement values."""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def traffic_figures(directory: str | Path) -> list[Path]:
    base=Path(directory)
    csv=base/"traffic_summary.csv"
    if not csv.exists():raise FileNotFoundError("No verified SUMO runs: traffic_summary.csv missing")
    df=pd.read_csv(csv)
    if df.empty:raise ValueError("Empty traffic summary")
    out=base/"figures";out.mkdir(parents=True,exist_ok=True)
    pivot=df.pivot(index="scenario",columns="method",values="mean_time_loss_s")
    ax=pivot.plot.bar(rot=0,figsize=(8,5),title="SUMO: mean completed-trip time loss by scenario")
    ax.set_ylabel("Mean time loss, seconds (completed trips only)")
    ax.figure.tight_layout();p=out/"traffic_time_loss.png";ax.figure.savefig(p,dpi=170);plt.close(ax.figure)
    pivot2=df.pivot(index="scenario",columns="method",values="mean_unfinished")
    ax=pivot2.plot.bar(rot=0,figsize=(8,5),title="SUMO: unfinished vehicles at evaluation horizon")
    ax.set_ylabel("Unfinished vehicles (mean across held-out seeds)")
    ax.figure.tight_layout();p2=out/"traffic_unfinished.png";ax.figure.savefig(p2,dpi=170);plt.close(ax.figure)
    return [p,p2]
