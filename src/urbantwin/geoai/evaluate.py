"""Render figures from recorded, hash-matched model evaluations."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from urbantwin.common import sha256_file


def make_figures(results_dir: str | Path) -> list[Path]:
    base = Path(results_dir)
    metrics_path = base / "geoai_metrics.csv"
    details_path = base / "geoai_details.json"
    if not metrics_path.is_file() or not details_path.is_file():
        raise FileNotFoundError("Run GeoAI training before plotting")
    data = pd.read_csv(metrics_path)
    details = json.loads(details_path.read_text(encoding="utf-8"))
    if data.empty:
        raise ValueError("GeoAI metrics are empty")
    if details.get("metrics_sha256") != sha256_file(metrics_path):
        raise ValueError("GeoAI metrics do not match their detailed result provenance")
    required = {"method", "features", "data_kind", "macro_f1"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"GeoAI metrics are missing columns: {sorted(missing)}")
    source_labels = set(data["data_kind"].astype(str))
    if len(source_labels) != 1 or source_labels != {str(details.get("provenance_kind"))}:
        raise ValueError("GeoAI figure inputs have inconsistent source labels")
    source_label = next(iter(source_labels))

    output = base / "figures"
    output.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(9, 4.5))
    names = (data["method"] + "\n" + data["features"]).to_list()
    error = data["macro_f1_std"] if "macro_f1_std" in data.columns else None
    axis.bar(names, data["macro_f1"], yerr=error, capsize=3 if error is not None else 0)
    axis.set(
        ylabel="Mean macro F1 against WorldCover pseudo-labels",
        ylim=(0, 1),
        title=f"Repeated grouped spatial evaluation (source: {source_label})",
    )
    figure.tight_layout()
    comparison_path = output / "geoai_model_comparison.png"
    figure.savefig(comparison_path, dpi=170)
    plt.close(figure)

    best = data.loc[data.macro_f1.idxmax()]
    key = f"{best['method']}__{best['features']}"
    experiment = details.get("experiments", {}).get(key)
    if not isinstance(experiment, dict) or "confusion_matrix" not in experiment:
        raise ValueError(f"Missing detailed confusion matrix for {key}")
    matrix = np.asarray(experiment["confusion_matrix"])
    class_names = details.get("class_names")
    if (
        matrix.ndim != 2
        or matrix.shape[0] != matrix.shape[1]
        or not isinstance(class_names, list)
        or len(class_names) != matrix.shape[0]
    ):
        raise ValueError("Confusion matrix and class labels are inconsistent")
    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    axis.set_yticks(range(len(class_names)), class_names)
    axis.set(
        xlabel="Predicted WorldCover class",
        ylabel="WorldCover class",
        title=f"Aggregated per-split confusion counts (source: {source_label})",
    )
    figure.colorbar(image, ax=axis, label="Evaluated pixels across splits")
    figure.tight_layout()
    confusion_path = output / "geoai_confusion_matrix.png"
    figure.savefig(confusion_path, dpi=170)
    plt.close(figure)
    return [comparison_path, confusion_path]
