"""Render plots from actual recorded model evaluations, never fake scores."""
from __future__ import annotations
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def make_figures(results_dir: str | Path) -> list[Path]:
    base=Path(results_dir)
    csv=base/"geoai_metrics.csv"
    js=base/"geoai_details.json"
    if not csv.exists() or not js.exists():
        raise FileNotFoundError("Run geoai training before plotting")
    data=pd.read_csv(csv)
    details=json.loads(js.read_text(encoding="utf-8"))
    out=base/"figures";out.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(figsize=(9,4.5))
    names=(data["method"]+"\n"+data["features"]).to_list()
    ax.bar(names,data["macro_f1"])
    ax.set(ylabel="Macro F1 against WorldCover pseudo-labels",ylim=(0,1),title="Spatial holdout: land-cover model comparison")
    fig.tight_layout(); p=out/"geoai_model_comparison.png";fig.savefig(p,dpi=170);plt.close(fig)
    # Confusion matrix for top observed macro F1
    best=data.loc[data.macro_f1.idxmax()]
    key=best["method"]+"__"+best["features"]
    matrix=np.array(details["experiments"][key]["confusion_matrix"])
    fig,ax=plt.subplots(figsize=(7,6));im=ax.imshow(matrix,cmap="Blues")
    classes=details["class_names"]
    ax.set_xticks(range(len(classes)),classes,rotation=45,ha="right")
    ax.set_yticks(range(len(classes)),classes)
    ax.set(xlabel="Predicted WorldCover class",ylabel="WorldCover class",title="Spatial holdout confusion matrix")
    fig.colorbar(im,ax=ax,label="Pixels")
    fig.tight_layout();p2=out/"geoai_confusion_matrix.png";fig.savefig(p2,dpi=170);plt.close(fig)
    return [p,p2]
