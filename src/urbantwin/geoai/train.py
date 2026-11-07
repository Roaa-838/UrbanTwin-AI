"""Spatial-blocked classification of ESA WorldCover pseudo-labels.

This measures agreement with a raster label product, NOT surveyed map accuracy.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.model_selection import GroupShuffleSplit
from urbantwin.common import load_yaml, save_json, require_real_provenance
from urbantwin.data.earth_observation import compute_indices, WORLDCOVER_CLASSES


def construct_samples(bands: np.ndarray, labels: np.ndarray, block_pixels: int = 20):
    if bands.shape[0] != 5 or bands.shape[1:] != labels.shape:
        raise ValueError("Expected bands (5,H,W) and labels (H,W)")
    ndvi, ndbi = compute_indices(bands)
    all_features = np.concatenate([bands,ndvi[None],ndbi[None]],axis=0)
    valid = np.isin(labels, sorted(WORLDCOVER_CLASSES)) & np.all(np.isfinite(all_features), axis=0)
    rows, cols = np.indices(labels.shape)
    # Spatial groups are disjoint rectangular blocks, rather than random pixels.
    nblock_col = int(np.ceil(labels.shape[1]/block_pixels))
    groups = (rows//block_pixels)*nblock_col + (cols//block_pixels)
    return all_features[:,valid].T, labels[valid], groups[valid], valid


def split_groups(X, y, groups, test_fraction=0.25, seed=42):
    s = GroupShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed)
    train_idx, test_idx = next(s.split(X,y,groups=groups))
    if set(groups[train_idx]) & set(groups[test_idx]):
        raise AssertionError("Spatial group leakage")
    return train_idx,test_idx


def subsample_training(train_idx, y, max_per_class, seed):
    rng = np.random.default_rng(seed)
    chosen = []
    for label in np.unique(y[train_idx]):
        ids = train_idx[y[train_idx] == label]
        chosen.extend(rng.choice(ids, size=min(max_per_class, len(ids)), replace=False).tolist())
    return np.array(sorted(chosen),dtype=int)


def run_classification(data_dir: str | Path, out_dir: str | Path,
                       config_file: str | Path, allow_synthetic: bool=False) -> pd.DataFrame:
    directory = Path(data_dir)
    provenance = json.loads((directory/"provenance.json").read_text(encoding="utf-8"))
    if not allow_synthetic:
        require_real_provenance(provenance)
    cfg = load_yaml(config_file)["geoai"]
    arr = np.load(directory/"eo_stack.npz")
    X,y,groups,valid = construct_samples(arr["bands"],arr["labels"],int(cfg["spatial_block_pixels"]))
    train,test = split_groups(X,y,groups,float(cfg["test_fraction"]))
    min_train,min_test = int(cfg["min_train_per_class"]),int(cfg["min_test_per_class"])
    valid_classes = [int(c) for c in np.unique(y)
                     if (y[train]==c).sum()>=min_train and (y[test]==c).sum()>=min_test]
    if len(valid_classes)<2:
        raise RuntimeError("Not enough supported classes under spatial holdout; enlarge AOI or lower thresholds explicitly")
    train=train[np.isin(y[train],valid_classes)]
    test=test[np.isin(y[test],valid_classes)]
    train=subsample_training(train,y,int(cfg["max_train_per_class"]),42)
    if len(test)==0:
        raise RuntimeError("Empty test set")
    out=Path(out_dir); out.mkdir(parents=True,exist_ok=True)
    rows=[]
    detailed={"provenance_kind":provenance["data_kind"],"evaluation_reference":"WorldCover pseudo-label agreement",
              "spatial_block_pixels":int(cfg["spatial_block_pixels"]),
              "train_group_count":len(set(groups[train])),"test_group_count":len(set(groups[test])),
              "train_n":len(train),"test_n":len(test),"class_ids":valid_classes,
              "class_names":[WORLDCOVER_CLASSES[c] for c in valid_classes],"experiments":{}}
    for ablation in cfg["ablation"]:
        subset=slice(0,5 if ablation=="bands_only" else 7)
        for classifier in cfg["classifiers"]:
            if classifier=="random_forest":
                model=RandomForestClassifier(n_estimators=120,min_samples_leaf=2,n_jobs=-1,random_state=42)
            elif classifier=="hist_gradient_boosting":
                model=HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=15,learning_rate=.08,random_state=42)
            else:
                raise ValueError(f"Unknown model {classifier}")
            model.fit(X[train,subset],y[train])
            pred=model.predict(X[test,subset])
            key=f"{classifier}__{ablation}"
            row={"method":classifier,"features":ablation,"data_kind":provenance["data_kind"],
                 "train_n":len(train),"test_n":len(test),
                 "macro_f1":float(f1_score(y[test],pred,labels=valid_classes,average="macro",zero_division=0)),
                 "weighted_f1":float(f1_score(y[test],pred,labels=valid_classes,average="weighted",zero_division=0)),
                 "accuracy_vs_worldcover":float(accuracy_score(y[test],pred))}
            rows.append(row)
            detailed["experiments"][key]={"row":row,
                "per_class":classification_report(y[test],pred,labels=valid_classes,output_dict=True,zero_division=0),
                "confusion_matrix":confusion_matrix(y[test],pred,labels=valid_classes).tolist()}
    table=pd.DataFrame(rows)
    table.to_csv(out/"geoai_metrics.csv",index=False)
    save_json(detailed,out/"geoai_details.json")
    return table
