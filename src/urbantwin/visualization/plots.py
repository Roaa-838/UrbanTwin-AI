"""Figure generation from recorded experiment files; never invent values."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from urbantwin.common import sha256_file


def traffic_figures(directory: str | Path) -> list[Path]:
    base = Path(directory)
    summary_path = base / "traffic_summary.csv"
    provenance_path = base / "traffic_provenance.json"
    if not summary_path.is_file() or not provenance_path.is_file():
        raise FileNotFoundError("No provenance-complete SUMO run is available for plotting")
    frame = pd.read_csv(summary_path)
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if frame.empty:
        raise ValueError("Traffic summary is empty")
    if provenance.get("summary_sha256") != sha256_file(summary_path):
        raise ValueError("Traffic summary does not match its provenance hash")
    required = {
        "scenario",
        "method",
        "data_kind",
        "mean_time_loss_s_completed",
        "mean_unfinished_vehicles",
        "mean_pending_vehicles",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Traffic summary is missing columns: {sorted(missing)}")
    source_labels = set(frame["data_kind"].astype(str))
    if source_labels != {str(provenance.get("data_kind"))}:
        raise ValueError("Traffic summary and provenance source labels disagree")
    source_label = next(iter(source_labels))

    output = base / "figures"
    output.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    pivot = frame.pivot(
        index="scenario", columns="method", values="mean_time_loss_s_completed"
    )
    axis = pivot.plot.bar(
        rot=0,
        figsize=(9, 5),
        title=f"SUMO completed-trip time loss (source: {source_label})",
    )
    axis.set_ylabel("Mean time loss, seconds (completed trips only)")
    axis.figure.tight_layout()
    path = output / "traffic_time_loss.png"
    axis.figure.savefig(path, dpi=170)
    plt.close(axis.figure)
    paths.append(path)

    censoring = frame.set_index(["scenario", "method"])[
        ["mean_unfinished_vehicles", "mean_pending_vehicles"]
    ].rename(
        columns={
            "mean_unfinished_vehicles": "unfinished after insertion",
            "mean_pending_vehicles": "pending insertion",
        }
    )
    axis = censoring.plot.bar(
        rot=25,
        figsize=(10, 5),
        title=f"SUMO horizon censoring counts (source: {source_label})",
    )
    axis.set_ylabel("Mean vehicles across paired held-out seeds")
    axis.figure.tight_layout()
    path = output / "traffic_horizon_counts.png"
    axis.figure.savefig(path, dpi=170)
    plt.close(axis.figure)
    paths.append(path)

    co2_column = "mean_simulated_co2_g_inserted_fleet"
    if co2_column in frame and frame[co2_column].notna().any():
        pivot = frame.pivot(index="scenario", columns="method", values=co2_column)
        axis = pivot.plot.bar(
            rot=0,
            figsize=(9, 5),
            title=f"Simulator-only SUMO CO2 estimate (source: {source_label})",
        )
        axis.set_ylabel("Mean modeled CO2, g (inserted fleet over fixed horizon)")
        axis.figure.tight_layout()
        path = output / "traffic_simulated_co2.png"
        axis.figure.savefig(path, dpi=170)
        plt.close(axis.figure)
        paths.append(path)
    return paths
