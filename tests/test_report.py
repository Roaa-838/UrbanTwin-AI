import json

import numpy as np
import pandas as pd
import pytest

from urbantwin.common import sha256_file
from urbantwin.visualization.report_writer import (
    ArtifactIntegrityError,
    make_report,
    validate_geoai_artifacts,
    validate_traffic_artifacts,
)


def _geo_bundle(root):
    data = root / "data/real"
    results = root / "results/geoai"
    configs = root / "configs"
    data.mkdir(parents=True)
    results.mkdir(parents=True)
    configs.mkdir(parents=True)
    provenance_path = data / "provenance.json"
    stack_path = data / "eo_stack.npz"
    np.savez_compressed(
        stack_path,
        bands=np.ones((5, 2, 2), dtype=np.float32),
        labels=np.full((2, 2), 50, dtype=np.uint8),
    )
    provenance_path.write_text(
        json.dumps(
            {
                "schema_version": "urbantwin.eo-provenance/v1",
                "data_kind": "real",
                "generated_utc": "2026-01-01T00:00:00+00:00",
                "sentinel_stac_collection": "sentinel-2-l2a",
                "sentinel_stac_item_ids": ["S2_TEST_ITEM"],
                "worldcover_stac_collection": "esa-worldcover",
                "worldcover_stac_item_ids": ["WORLDCOVER_2021_TEST_TILE"],
                "band_order": ["B02", "B03", "B04", "B08", "B11"],
                "crs": "EPSG:32636",
                "transform": [10, 0, 300000, 0, -10, 3330000],
                "width": 2,
                "height": 2,
                "processing": {
                    "continuous_resampling": "bilinear",
                    "categorical_resampling": "nearest",
                    "clear_scl_classes": [4, 5, 6, 7],
                },
                "eo_stack_sha256": sha256_file(stack_path),
                "signed_asset_urls_recorded": False,
            }
        ),
        encoding="utf-8",
    )
    config_path = configs / "experiments.yaml"
    config_path.write_text("geoai: {}\n", encoding="utf-8")
    row = {
        "method": "random_forest",
        "features": "bands_only",
        "data_kind": "real",
        "split_count": 1,
        "train_n": 3,
        "test_n": 1,
        "macro_f1": 0.5,
        "weighted_f1": 0.5,
        "accuracy_vs_worldcover": 0.5,
    }
    metrics_path = results / "geoai_metrics.csv"
    pd.DataFrame([row]).to_csv(metrics_path, index=False)
    split_path = results / "geoai_split_metrics.csv"
    split_row = {key: value for key, value in row.items() if key != "split_count"}
    split_row["split_seed"] = 42
    pd.DataFrame([split_row]).to_csv(split_path, index=False)
    details = {
        "schema_version": 2,
        "generated_utc": "2026-01-02T00:00:00+00:00",
        "provenance_kind": "real",
        "class_selection_protocol": "training_partition_only",
        "input_provenance_sha256": sha256_file(provenance_path),
        "input_stack_sha256": sha256_file(stack_path),
        "config_path": "configs/experiments.yaml",
        "config_sha256": sha256_file(config_path),
        "metrics_sha256": sha256_file(metrics_path),
        "split_metrics_sha256": sha256_file(split_path),
        "experiments": {
            "random_forest__bands_only": {
                "row": row,
                "runs": [{"seed": 42, "row": split_row}],
            }
        },
    }
    (results / "geoai_details.json").write_text(json.dumps(details), encoding="utf-8")
    return metrics_path


def _traffic_bundle(root):
    results = root / "results/traffic"
    configs = root / "configs"
    source = results / "synthetic_network"
    results.mkdir(parents=True)
    configs.mkdir(parents=True, exist_ok=True)
    source.mkdir(parents=True)
    config_path = configs / "experiments.yaml"
    if not config_path.exists():
        config_path.write_text("traffic: {}\n", encoding="utf-8")
    network_path = source / "junction.net.xml"
    network_path.write_text("<net/>\n", encoding="utf-8")
    route_path = source / "routes_medium_101.rou.xml"
    route_path.write_text("<routes/>\n", encoding="utf-8")

    methods = ["fixed", "GA", "PSO", "random_search"]
    rows = []
    for index, method in enumerate(methods):
        rows.append(
            {
                "scenario": "medium",
                "method": method,
                "seed": 101,
                "data_kind": "simulation_abstract",
                "train_seed_overlap": False,
                "generated_vehicles": 10,
                "inserted_vehicles": 9,
                "completed_vehicles": 8,
                "unfinished_vehicles": 1,
                "pending_vehicles": 1,
                "completion_fraction_generated": 0.8,
                "mean_time_loss_s_completed": 20.0 - index,
                "simulated_co2_g_inserted_fleet": 100.0 + index,
                "simulated_co2_label": (
                    "SUMO model estimate; simulator-only, not observed pollution"
                ),
            }
        )
    metrics_path = results / "traffic_metrics.csv"
    metrics = pd.DataFrame(rows)
    metrics.to_csv(metrics_path, index=False)
    summary = pd.DataFrame(
        [
            {
                "scenario": row["scenario"],
                "method": row["method"],
                "data_kind": row["data_kind"],
                "runs": 1,
                "mean_time_loss_s_completed": row["mean_time_loss_s_completed"],
                "mean_generated_vehicles": row["generated_vehicles"],
                "mean_inserted_vehicles": row["inserted_vehicles"],
                "mean_completed_vehicles": row["completed_vehicles"],
                "mean_unfinished_vehicles": row["unfinished_vehicles"],
                "mean_pending_vehicles": row["pending_vehicles"],
            }
            for row in rows
        ]
    )
    summary_path = results / "traffic_summary.csv"
    summary.to_csv(summary_path, index=False)
    fixed_loss = rows[0]["mean_time_loss_s_completed"]
    paired = pd.DataFrame(
        [
            {
                "scenario": row["scenario"],
                "method": row["method"],
                "seed": row["seed"],
                "data_kind": row["data_kind"],
                "delta_mean_time_loss_s_completed_method_minus_fixed": (
                    row["mean_time_loss_s_completed"] - fixed_loss
                ),
            }
            for row in rows[1:]
        ]
    )
    paired_path = results / "traffic_paired_deltas.csv"
    paired.to_csv(paired_path, index=False)
    paired_summary = paired[["scenario", "method", "data_kind"]].copy()
    paired_summary["runs"] = 1
    paired_summary_path = results / "traffic_paired_summary.csv"
    paired_summary.to_csv(paired_summary_path, index=False)
    provenance = {
        "schema_version": "urbantwin.traffic-benchmark/v2",
        "data_kind": "simulation_abstract",
        "generated_utc": "2026-01-02T00:00:00+00:00",
        "config_path": "configs/experiments.yaml",
        "config_sha256": sha256_file(config_path),
        "network": "results/traffic/synthetic_network/junction.net.xml",
        "network_sha256": sha256_file(network_path),
        "route_artifacts_by_scenario_seed": {
            "medium__101": {
                "path": "results/traffic/synthetic_network/routes_medium_101.rou.xml",
                "sha256": sha256_file(route_path),
            }
        },
        "metrics_sha256": sha256_file(metrics_path),
        "summary_sha256": sha256_file(summary_path),
        "paired_deltas_sha256": sha256_file(paired_path),
        "paired_summary_sha256": sha256_file(paired_summary_path),
        "track_emissions": True,
        "emission_class_assumptions": ["HBEFA3/PC_G_EU4"],
    }
    (results / "traffic_provenance.json").write_text(
        json.dumps(provenance), encoding="utf-8"
    )
    return route_path


def test_draft_report_renders_without_consuming_results(tmp_path):
    invalid = tmp_path / "results/geoai"
    invalid.mkdir(parents=True)
    (invalid / "geoai_metrics.csv").write_text("not,a,valid,bundle\n", encoding="utf-8")
    pdf = tmp_path / "report.pdf"
    make_report(pdf, include_draft=True, root=tmp_path)
    sidecar = json.loads(pdf.with_suffix(".provenance.json").read_text(encoding="utf-8"))
    assert pdf.exists() and pdf.stat().st_size > 1500
    assert sidecar["data_kind"] == "methodology_only"
    assert sidecar["inputs"] == {"geoai": None, "traffic": None}
    assert sidecar["report_sha256"] == sha256_file(pdf)


def test_valid_real_bundle_can_be_reported(tmp_path):
    _geo_bundle(tmp_path)
    pdf = tmp_path / "verified.pdf"
    make_report(pdf, root=tmp_path)
    sidecar = json.loads(pdf.with_suffix(".provenance.json").read_text(encoding="utf-8"))
    assert sidecar["data_kind"] == "verified_experiment_report"
    assert sidecar["inputs"]["geoai"]["metrics"]
    assert sidecar["inputs"]["traffic"] is None


def test_tampered_geoai_csv_is_rejected(tmp_path):
    metrics_path = _geo_bundle(tmp_path)
    metrics = pd.read_csv(metrics_path)
    metrics.loc[0, "macro_f1"] = 0.99
    metrics.to_csv(metrics_path, index=False)
    with pytest.raises(ArtifactIntegrityError, match="hash mismatch"):
        validate_geoai_artifacts(tmp_path)


def test_mismatched_geoai_detail_row_is_rejected(tmp_path):
    _geo_bundle(tmp_path)
    details_path = tmp_path / "results/geoai/geoai_details.json"
    details = json.loads(details_path.read_text(encoding="utf-8"))
    details["experiments"]["random_forest__bands_only"]["row"]["macro_f1"] = 0.9
    details_path.write_text(json.dumps(details), encoding="utf-8")
    with pytest.raises(ArtifactIntegrityError, match="disagree"):
        validate_geoai_artifacts(tmp_path)


def test_mislabeled_geoai_split_metrics_are_rejected(tmp_path):
    _geo_bundle(tmp_path)
    split_path = tmp_path / "results/geoai/geoai_split_metrics.csv"
    split = pd.read_csv(split_path)
    split.loc[0, "data_kind"] = "synthetic"
    split.to_csv(split_path, index=False)
    details_path = tmp_path / "results/geoai/geoai_details.json"
    details = json.loads(details_path.read_text(encoding="utf-8"))
    details["split_metrics_sha256"] = sha256_file(split_path)
    details_path.write_text(json.dumps(details), encoding="utf-8")
    with pytest.raises(ArtifactIntegrityError, match="split metric rows"):
        validate_geoai_artifacts(tmp_path)


def test_incomplete_geoai_bundle_is_rejected(tmp_path):
    result = tmp_path / "results/geoai"
    result.mkdir(parents=True)
    (result / "geoai_metrics.csv").write_text("method,features\na,b\n", encoding="utf-8")
    with pytest.raises(ArtifactIntegrityError, match="Incomplete GeoAI artifacts"):
        validate_geoai_artifacts(tmp_path)


def test_valid_traffic_bundle_preserves_counts_pairs_and_source_hashes(tmp_path):
    _traffic_bundle(tmp_path)
    bundle = validate_traffic_artifacts(tmp_path)
    assert bundle["provenance"]["data_kind"] == "simulation_abstract"
    assert set(bundle["metrics"]["method"]) == {"fixed", "GA", "PSO", "random_search"}


def test_tampered_traffic_route_is_rejected(tmp_path):
    route_path = _traffic_bundle(tmp_path)
    route_path.write_text("<routes><vehicle/></routes>\n", encoding="utf-8")
    with pytest.raises(ArtifactIntegrityError, match="hash mismatch"):
        validate_traffic_artifacts(tmp_path)


def test_report_accepts_geoai_and_simulator_labeled_traffic_bundles(tmp_path):
    _geo_bundle(tmp_path)
    _traffic_bundle(tmp_path)
    output = tmp_path / "combined.pdf"
    make_report(output, root=tmp_path)
    sidecar = json.loads(output.with_suffix(".provenance.json").read_text(encoding="utf-8"))
    assert output.stat().st_size > 1500
    assert sidecar["inputs"]["traffic"]["network"]
    assert sidecar["inputs"]["traffic"]["routes"]["medium__101"]
