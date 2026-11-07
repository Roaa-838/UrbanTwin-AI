"""Build provenance-checked research reports from executed experiments.

The report generator treats an experiment as an indivisible bundle: metrics,
detailed results, source data and their recorded hashes must all agree. File
presence alone is not evidence that two artifacts belong to the same run.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from urbantwin.common import require_real_provenance, save_json, sha256_file

ROOT = Path(__file__).resolve().parents[3]


class ArtifactIntegrityError(RuntimeError):
    """Raised when result artifacts cannot be tied to one reproducible run."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactIntegrityError(f"Cannot read valid JSON provenance: {path}") from exc
    if not isinstance(value, dict):
        raise ArtifactIntegrityError(f"Expected a JSON object: {path}")
    return value


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        raise ArtifactIntegrityError(f"Cannot read result CSV: {path}") from exc
    if frame.empty:
        raise ArtifactIntegrityError(f"Result CSV is empty: {path}")
    return frame


def _require_columns(frame: pd.DataFrame, columns: set[str], path: Path) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ArtifactIntegrityError(f"Missing columns in {path}: {', '.join(missing)}")


def _require_hash(record: dict[str, Any], key: str, path: Path) -> None:
    expected = record.get(key)
    if (
        not isinstance(expected, str)
        or len(expected) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in expected)
    ):
        raise ArtifactIntegrityError(f"Missing or invalid {key} in experiment provenance")
    actual = sha256_file(path)
    if actual != expected:
        raise ArtifactIntegrityError(
            f"Artifact hash mismatch for {path.name}; rerun the experiment before reporting"
        )


def _parse_utc(value: Any, field: str) -> datetime:
    if not isinstance(value, str):
        raise ArtifactIntegrityError(f"Missing {field} timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise ArtifactIntegrityError(f"Invalid {field} timestamp") from exc
    if parsed.tzinfo is None:
        raise ArtifactIntegrityError(f"{field} timestamp must include a timezone")
    return parsed.astimezone(timezone.utc)


def _same_scalar(left: Any, right: Any) -> bool:
    """Compare a pandas CSV scalar with its JSON representation."""
    if pd.isna(left):
        return right is None or bool(pd.isna(right))
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-12)
    return str(left) == str(right)


def _require_matching_row(csv_row: dict[str, Any], detail_row: Any, key: str) -> None:
    if not isinstance(detail_row, dict):
        raise ArtifactIntegrityError(f"Metric row has no matching detail record: {key}")
    missing = set(csv_row) - set(detail_row)
    if missing:
        raise ArtifactIntegrityError(
            f"Detailed metric row {key} is missing fields: {', '.join(sorted(missing))}"
        )
    mismatched = [
        field
        for field, value in csv_row.items()
        if not _same_scalar(value, detail_row.get(field))
    ]
    if mismatched:
        raise ArtifactIntegrityError(
            f"CSV and detailed metric row disagree for {key}: {', '.join(sorted(mismatched))}"
        )


def validate_geoai_artifacts(root: str | Path = ROOT) -> dict[str, Any] | None:
    """Return one verified GeoAI result bundle, or ``None`` when it was not run."""

    root = Path(root)
    metrics_path = root / "results/geoai/geoai_metrics.csv"
    split_metrics_path = root / "results/geoai/geoai_split_metrics.csv"
    details_path = root / "results/geoai/geoai_details.json"
    result_paths = (metrics_path, split_metrics_path, details_path)
    present = [path.exists() for path in result_paths]
    if not any(present):
        return None
    if not all(present):
        raise ArtifactIntegrityError(
            "Incomplete GeoAI artifacts: aggregate metrics, split metrics and details are required"
        )

    provenance_path = root / "data/real/provenance.json"
    stack_path = root / "data/real/eo_stack.npz"
    for path in (provenance_path, stack_path):
        if not path.is_file():
            raise ArtifactIntegrityError(f"GeoAI source artifact is missing: {path}")

    details = _read_json(details_path)
    provenance = _read_json(provenance_path)
    metrics = _read_csv(metrics_path)
    split_metrics = _read_csv(split_metrics_path)
    config_value = details.get("config_path")
    if not isinstance(config_value, str) or not config_value:
        raise ArtifactIntegrityError("GeoAI details are missing config_path")
    config_path = Path(config_value)
    if not config_path.is_absolute():
        config_path = root / config_path
    if not config_path.is_file():
        raise ArtifactIntegrityError(f"GeoAI configuration artifact is missing: {config_path}")
    _require_columns(
        metrics,
        {"method", "features", "data_kind", "macro_f1", "test_n"},
        metrics_path,
    )
    _require_columns(
        split_metrics,
        {"method", "features", "data_kind", "split_seed", "macro_f1", "test_n"},
        split_metrics_path,
    )
    try:
        require_real_provenance(provenance)
    except ValueError as exc:
        raise ArtifactIntegrityError(f"Invalid real EO provenance: {exc}") from exc
    if details.get("provenance_kind") != "real":
        raise ArtifactIntegrityError("GeoAI report input must have real source provenance")
    if set(metrics["data_kind"].astype(str)) != {"real"}:
        raise ArtifactIntegrityError("GeoAI metric rows are not exclusively labeled real")
    if set(split_metrics["data_kind"].astype(str)) != {"real"}:
        raise ArtifactIntegrityError("GeoAI split metric rows are not exclusively labeled real")

    _require_hash(details, "input_provenance_sha256", provenance_path)
    _require_hash(details, "input_stack_sha256", stack_path)
    if provenance.get("eo_stack_sha256") != sha256_file(stack_path):
        raise ArtifactIntegrityError("EO provenance and raster stack hashes do not match")
    _require_hash(details, "config_sha256", config_path)
    _require_hash(details, "metrics_sha256", metrics_path)
    _require_hash(details, "split_metrics_sha256", split_metrics_path)
    result_time = _parse_utc(details.get("generated_utc"), "GeoAI generated_utc")
    source_time = _parse_utc(provenance.get("generated_utc"), "EO generated_utc")
    if result_time < source_time:
        raise ArtifactIntegrityError("GeoAI results predate their Earth-observation source")

    experiments = details.get("experiments")
    if not isinstance(experiments, dict):
        raise ArtifactIntegrityError("GeoAI details do not contain experiment records")
    if details.get("class_selection_protocol") != "training_partition_only":
        raise ArtifactIntegrityError(
            "GeoAI results do not use the required training-only class-selection protocol"
        )
    for column in ("macro_f1", "weighted_f1", "accuracy_vs_worldcover"):
        values = pd.to_numeric(metrics[column], errors="coerce")
        if values.isna().any() or not values.between(0, 1).all():
            raise ArtifactIntegrityError(f"GeoAI metric {column} must be finite and within [0, 1]")
    csv_keys: list[str] = []
    for row in metrics.to_dict(orient="records"):
        key = f"{row['method']}__{row['features']}"
        csv_keys.append(key)
        detail_row = experiments.get(key, {}).get("row")
        _require_matching_row(row, detail_row, key)
        if str(detail_row.get("data_kind")) != "real":
            raise ArtifactIntegrityError(f"GeoAI detail record has mismatched source label: {key}")
    if len(csv_keys) != len(set(csv_keys)):
        raise ArtifactIntegrityError("GeoAI CSV contains duplicate experiment rows")
    if set(csv_keys) != set(experiments):
        raise ArtifactIntegrityError("GeoAI CSV and detailed experiment sets do not match")
    split_keys = {
        f"{row.method}__{row.features}"
        for row in split_metrics[["method", "features"]].itertuples(index=False)
    }
    if split_keys != set(csv_keys):
        raise ArtifactIntegrityError("GeoAI aggregate and split experiment sets do not match")
    for key, experiment in experiments.items():
        runs = experiment.get("runs")
        if not isinstance(runs, list):
            raise ArtifactIntegrityError(f"GeoAI detailed runs are missing for {key}")
        method, features = key.split("__", maxsplit=1)
        rows = split_metrics.loc[
            (split_metrics["method"].astype(str) == method)
            & (split_metrics["features"].astype(str) == features)
        ].to_dict(orient="records")
        if len(rows) != len(runs):
            raise ArtifactIntegrityError(f"GeoAI split run count differs from details for {key}")
        run_by_seed = {
            int(run["seed"]): run
            for run in runs
            if isinstance(run, dict) and "seed" in run
        }
        if len(run_by_seed) != len(runs):
            raise ArtifactIntegrityError(f"GeoAI detailed split seeds are missing or duplicated for {key}")
        for row in rows:
            seed = int(row["split_seed"])
            run = run_by_seed.get(seed)
            if run is None:
                raise ArtifactIntegrityError(
                    f"GeoAI split seed {seed} has no detailed record for {key}"
                )
            _require_matching_row(row, run.get("row"), f"{key} split seed {seed}")
        declared_count = experiment["row"].get("split_count")
        if declared_count is not None and int(declared_count) != len(rows):
            raise ArtifactIntegrityError(f"GeoAI aggregate split_count is inconsistent for {key}")

    return {
        "metrics": metrics,
        "split_metrics": split_metrics,
        "details": details,
        "provenance": provenance,
        "hashes": {
            "metrics": sha256_file(metrics_path),
            "split_metrics": sha256_file(split_metrics_path),
            "details": sha256_file(details_path),
            "eo_stack": sha256_file(stack_path),
            "eo_provenance": sha256_file(provenance_path),
            "config": sha256_file(config_path),
        },
    }


def validate_traffic_artifacts(
    root: str | Path = ROOT,
    results_dir: str | Path | None = None,
) -> dict[str, Any] | None:
    """Return one verified SUMO result bundle, or ``None`` when it was not run."""

    root = Path(root)
    result_base = Path(results_dir) if results_dir is not None else root / "results/traffic"
    if not result_base.is_absolute():
        result_base = root / result_base
    metrics_path = result_base / "traffic_metrics.csv"
    summary_path = result_base / "traffic_summary.csv"
    paired_deltas_path = result_base / "traffic_paired_deltas.csv"
    paired_summary_path = result_base / "traffic_paired_summary.csv"
    provenance_path = result_base / "traffic_provenance.json"
    paths = (
        metrics_path,
        summary_path,
        paired_deltas_path,
        paired_summary_path,
        provenance_path,
    )
    present = [path.exists() for path in paths]
    if not any(present):
        return None
    if not all(present):
        raise ArtifactIntegrityError(
            "Incomplete traffic artifacts: metrics, summary, paired tables and provenance "
            "are all required"
        )

    provenance = _read_json(provenance_path)
    metrics = _read_csv(metrics_path)
    summary = _read_csv(summary_path)
    paired_deltas = _read_csv(paired_deltas_path)
    paired_summary = _read_csv(paired_summary_path)
    if provenance.get("schema_version") != "urbantwin.traffic-benchmark/v2":
        raise ArtifactIntegrityError("Traffic provenance has an unsupported schema version")
    _require_columns(
        metrics,
        {
            "scenario",
            "method",
            "seed",
            "data_kind",
            "generated_vehicles",
            "inserted_vehicles",
            "completed_vehicles",
            "unfinished_vehicles",
            "pending_vehicles",
            "completion_fraction_generated",
            "mean_time_loss_s_completed",
            "train_seed_overlap",
            "simulated_co2_g_inserted_fleet",
            "simulated_co2_label",
        },
        metrics_path,
    )
    _require_columns(
        summary,
        {
            "scenario",
            "method",
            "data_kind",
            "runs",
            "mean_time_loss_s_completed",
            "mean_generated_vehicles",
            "mean_inserted_vehicles",
            "mean_completed_vehicles",
            "mean_unfinished_vehicles",
            "mean_pending_vehicles",
        },
        summary_path,
    )
    _require_columns(
        paired_deltas,
        {"scenario", "method", "seed", "data_kind"},
        paired_deltas_path,
    )
    _require_columns(
        paired_summary,
        {"scenario", "method", "data_kind", "runs"},
        paired_summary_path,
    )
    data_kind = provenance.get("data_kind")
    if data_kind not in {"simulation_abstract", "simulation_external_user_network"}:
        raise ArtifactIntegrityError("Traffic provenance has an unsupported source label")
    for frame, label in (
        (metrics, "metric"),
        (summary, "summary"),
        (paired_deltas, "paired-delta"),
        (paired_summary, "paired-summary"),
    ):
        if set(frame["data_kind"].astype(str)) != {data_kind}:
            raise ArtifactIntegrityError(
                f"Traffic {label} source labels do not match provenance"
            )
    _parse_utc(provenance.get("generated_utc"), "traffic generated_utc")
    _require_hash(provenance, "metrics_sha256", metrics_path)
    _require_hash(provenance, "summary_sha256", summary_path)
    _require_hash(provenance, "paired_deltas_sha256", paired_deltas_path)
    _require_hash(provenance, "paired_summary_sha256", paired_summary_path)

    config_path_value = provenance.get("config_path")
    if not isinstance(config_path_value, str) or not config_path_value:
        raise ArtifactIntegrityError("Traffic provenance is missing config_path")
    config_path = Path(config_path_value)
    if not config_path.is_absolute():
        config_path = root / config_path
    if not config_path.is_file():
        raise ArtifactIntegrityError(f"Traffic configuration artifact is missing: {config_path}")
    _require_hash(provenance, "config_sha256", config_path)

    network_value = provenance.get("network")
    if not isinstance(network_value, str) or not network_value:
        raise ArtifactIntegrityError("Traffic provenance is missing its network path")
    network_path = Path(network_value)
    if not network_path.is_absolute():
        network_path = root / network_path
    if not network_path.is_file():
        raise ArtifactIntegrityError(f"Traffic network artifact is missing: {network_path}")
    _require_hash(provenance, "network_sha256", network_path)
    route_artifacts = provenance.get("route_artifacts_by_scenario_seed")
    if not isinstance(route_artifacts, dict) or not route_artifacts:
        raise ArtifactIntegrityError("Traffic provenance has no route artifacts")
    route_hashes: dict[str, str] = {}
    for key, record in route_artifacts.items():
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise ArtifactIntegrityError(f"Invalid traffic route artifact record: {key}")
        route_path = Path(record["path"])
        if not route_path.is_absolute():
            route_path = root / route_path
        if not route_path.is_file():
            raise ArtifactIntegrityError(f"Traffic route artifact is missing: {route_path}")
        _require_hash(record, "sha256", route_path)
        route_hashes[str(key)] = sha256_file(route_path)

    required_methods = {"fixed", "GA", "PSO", "random_search"}
    observed_methods = set(metrics["method"].astype(str))
    if observed_methods != required_methods:
        raise ArtifactIntegrityError(
            f"Traffic methods must be {sorted(required_methods)}, found {sorted(observed_methods)}"
        )
    if metrics.duplicated(["scenario", "method", "seed", "data_kind"]).any():
        raise ArtifactIntegrityError("Traffic metrics contain duplicate paired-run keys")
    overlap_values = metrics["train_seed_overlap"].astype(str).str.lower()
    if not overlap_values.isin({"false", "0"}).all():
        raise ArtifactIntegrityError("Traffic evaluation rows overlap optimizer training seeds")

    count_columns = [
        "generated_vehicles",
        "inserted_vehicles",
        "completed_vehicles",
        "unfinished_vehicles",
        "pending_vehicles",
    ]
    counts = metrics[count_columns].apply(pd.to_numeric, errors="coerce")
    if counts.isna().any().any() or (counts < 0).any().any():
        raise ArtifactIntegrityError("Traffic vehicle counts must be finite and non-negative")
    if not (
        counts["inserted_vehicles"]
        == counts["completed_vehicles"] + counts["unfinished_vehicles"]
    ).all():
        raise ArtifactIntegrityError("Inserted traffic counts do not equal completed + unfinished")
    if not (
        counts["generated_vehicles"]
        == counts["inserted_vehicles"] + counts["pending_vehicles"]
    ).all():
        raise ArtifactIntegrityError("Generated traffic counts do not equal inserted + pending")
    fractions = pd.to_numeric(metrics["completion_fraction_generated"], errors="coerce")
    if fractions.isna().any() or not fractions.between(0, 1).all():
        raise ArtifactIntegrityError("Traffic completion fractions must be within [0, 1]")
    expected_fractions = counts["completed_vehicles"] / counts["generated_vehicles"]
    if not (fractions - expected_fractions).abs().le(1e-12).all():
        raise ArtifactIntegrityError("Traffic completion fractions disagree with vehicle counts")

    for scenario, scenario_frame in metrics.groupby("scenario"):
        baseline_seeds = set(
            scenario_frame.loc[scenario_frame["method"] == "fixed", "seed"].astype(int)
        )
        if not baseline_seeds:
            raise ArtifactIntegrityError(f"Traffic scenario {scenario} has no fixed baseline")
        for method, method_frame in scenario_frame.groupby("method"):
            if set(method_frame["seed"].astype(int)) != baseline_seeds:
                raise ArtifactIntegrityError(
                    f"Traffic method {method} does not use paired seeds in scenario {scenario}"
                )

    metric_groups = metrics.groupby(
        ["scenario", "method", "data_kind"], dropna=False
    ).size().to_dict()
    summary_groups = {
        (row["scenario"], row["method"], row["data_kind"]): int(row["runs"])
        for row in summary.to_dict(orient="records")
    }
    if metric_groups != summary_groups:
        raise ArtifactIntegrityError("Traffic summary run counts do not match traffic metrics")

    nonbaseline_groups = {
        key: count for key, count in metric_groups.items() if key[1] != "fixed"
    }
    paired_groups = paired_deltas.groupby(
        ["scenario", "method", "data_kind"], dropna=False
    ).size().to_dict()
    paired_summary_groups = {
        (row["scenario"], row["method"], row["data_kind"]): int(row["runs"])
        for row in paired_summary.to_dict(orient="records")
    }
    if paired_groups != nonbaseline_groups or paired_summary_groups != nonbaseline_groups:
        raise ArtifactIntegrityError("Traffic paired-seed tables do not match evaluation runs")

    delta_columns = [
        column
        for column in paired_deltas.columns
        if column.startswith("delta_") and column.endswith("_method_minus_fixed")
    ]
    if not delta_columns:
        raise ArtifactIntegrityError("Traffic paired-delta table contains no KPI deltas")
    metric_index = metrics.set_index(["scenario", "method", "seed", "data_kind"])
    for row in paired_deltas.to_dict(orient="records"):
        method_key = (row["scenario"], row["method"], row["seed"], row["data_kind"])
        fixed_key = (row["scenario"], "fixed", row["seed"], row["data_kind"])
        if method_key not in metric_index.index or fixed_key not in metric_index.index:
            raise ArtifactIntegrityError("Traffic paired delta references an unknown run")
        method_row = metric_index.loc[method_key]
        fixed_row = metric_index.loc[fixed_key]
        for column in delta_columns:
            metric = column[len("delta_") : -len("_method_minus_fixed")]
            if metric not in metrics.columns:
                raise ArtifactIntegrityError(f"Traffic paired delta references unknown KPI {metric}")
            expected = float(method_row[metric]) - float(fixed_row[metric])
            if not _same_scalar(row[column], expected):
                raise ArtifactIntegrityError(
                    f"Traffic paired delta disagrees with metric rows for {method_key}: {metric}"
                )

    track_emissions = provenance.get("track_emissions") is True
    co2 = pd.to_numeric(metrics["simulated_co2_g_inserted_fleet"], errors="coerce")
    if track_emissions:
        assumptions = provenance.get("emission_class_assumptions")
        if not isinstance(assumptions, list) or not assumptions:
            raise ArtifactIntegrityError("Tracked traffic emissions lack class assumptions")
        if co2.isna().any() or (co2 < 0).any():
            raise ArtifactIntegrityError("Tracked simulator-only CO2 values are missing or invalid")
    elif co2.notna().any():
        raise ArtifactIntegrityError("Traffic CO2 values exist although emissions tracking is disabled")
    if not metrics["simulated_co2_label"].astype(str).str.contains(
        "simulator-only", regex=False
    ).all():
        raise ArtifactIntegrityError("Traffic CO2 outputs lack an explicit simulator-only label")

    return {
        "metrics": metrics,
        "summary": summary,
        "paired_deltas": paired_deltas,
        "paired_summary": paired_summary,
        "provenance": provenance,
        "hashes": {
            "metrics": sha256_file(metrics_path),
            "summary": sha256_file(summary_path),
            "paired_deltas": sha256_file(paired_deltas_path),
            "paired_summary": sha256_file(paired_summary_path),
            "provenance": sha256_file(provenance_path),
            "config": sha256_file(config_path),
            "network": sha256_file(network_path),
            "routes": route_hashes,
        },
    }


def _table(data: list[list[str]], widths: list[int]) -> Table:
    table = Table(data, repeatRows=1, hAlign="LEFT", colWidths=widths)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e5e7eb")),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.grey),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    return table


def make_report(
    output: str | Path,
    include_draft: bool = False,
    *,
    root: str | Path | None = None,
) -> str:
    """Create a methodology-only draft or a provenance-checked research report."""

    report_root = Path(root) if root is not None else ROOT
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Draft mode intentionally never consumes result files. This prevents a draft
    # from accidentally presenting an unverified or partially copied experiment.
    geo = None if include_draft else validate_geoai_artifacts(report_root)
    traffic = None if include_draft else validate_traffic_artifacts(report_root)
    if not include_draft and geo is None:
        raise RuntimeError(
            "No provenance-verified real GeoAI output. Run the real experiment, or use "
            "--draft for a methodology-only report."
        )

    sheet = getSampleStyleSheet()
    sheet.add(
        ParagraphStyle(
            name="MainTitle",
            parent=sheet["Title"],
            fontSize=18,
            leading=23,
            alignment=TA_CENTER,
            spaceAfter=18,
        )
    )
    story = [
        Paragraph("UrbanTwin-AI", sheet["MainTitle"]),
        Paragraph(
            "Geospatial Intelligence and Traffic Optimization for Sustainable Cities",
            sheet["Heading2"],
        ),
        Paragraph("Independent research prototype | Cairo area of interest", sheet["Normal"]),
        Spacer(1, 14),
        Paragraph("Research question", sheet["Heading2"]),
        Paragraph(
            "Can spatial indicators derived from Earth observation be analyzed alongside "
            "controlled traffic-signal simulations to support reproducible urban research?",
            sheet["BodyText"],
        ),
        Spacer(1, 8),
        Paragraph("Scope and integrity", sheet["Heading2"]),
        Paragraph(
            "The satellite and ESA WorldCover measurements refer to a small Cairo study area. "
            "The repository's generated SUMO intersection is synthetic, not a reconstruction "
            "of Cairo street geometry, measured traffic flows, or measured emissions. The two "
            "components are reported separately.",
            sheet["BodyText"],
        ),
        Spacer(1, 8),
        Paragraph("Methods", sheet["Heading2"]),
        Paragraph(
            "Sentinel-2 L2A imagery is SCL-screened and aligned in a projected CRS, then "
            "compared with ESA WorldCover 2021 reference labels. Classifiers use repeated "
            "grouped spatial evaluation. Traffic optimizers use training seeds distinct from paired "
            "evaluation seeds.",
            sheet["BodyText"],
        ),
        Spacer(1, 8),
    ]

    if include_draft:
        story.extend(
            [
                Paragraph("Experimental status", sheet["Heading2"]),
                Paragraph(
                    "METHODOLOGY-ONLY DRAFT: this document contains no experiment metrics. "
                    "Do not cite it as an empirical finding.",
                    sheet["BodyText"],
                ),
            ]
        )
    else:
        data = geo["metrics"]
        story.append(Paragraph("Observed GeoAI results", sheet["Heading2"]))
        cells = [["Model", "Ablation", "Macro F1 mean ± SD", "Mean test pixels/split"]]
        for row in data.to_dict(orient="records"):
            macro_std = float(row.get("macro_f1_std", 0.0))
            cells.append(
                [
                    str(row["method"]),
                    str(row["features"]),
                    f"{float(row['macro_f1']):.3f} ± {macro_std:.3f}",
                    f"{float(row['test_n']):.1f}",
                ]
            )
        story.extend(
            [
                _table(cells, [116, 120, 105, 105]),
                Spacer(1, 8),
                Paragraph(
                    "These metrics assess agreement with WorldCover pseudo-labels, not "
                    "independently surveyed land-cover accuracy.",
                    sheet["BodyText"],
                ),
            ]
        )

    story.append(Spacer(1, 8))
    if traffic is not None:
        data = traffic["summary"]
        story.append(Paragraph("Observed simulator-only SUMO results", sheet["Heading2"]))
        cells = [
            [
                "Scenario",
                "Method",
                "Generated / inserted / completed",
                "Unfinished / pending",
                "Time loss (s)",
                "Modeled CO2 (g)",
            ]
        ]
        for row in data.to_dict(orient="records"):
            co2 = row.get("mean_simulated_co2_g_inserted_fleet")
            cells.append(
                [
                    str(row["scenario"]),
                    str(row["method"]),
                    (
                        f"{float(row['mean_generated_vehicles']):.1f} / "
                        f"{float(row['mean_inserted_vehicles']):.1f} / "
                        f"{float(row['mean_completed_vehicles']):.1f}"
                    ),
                    (
                        f"{float(row['mean_unfinished_vehicles']):.1f} / "
                        f"{float(row['mean_pending_vehicles']):.1f}"
                    ),
                    f"{float(row['mean_time_loss_s_completed']):.2f}",
                    "disabled" if pd.isna(co2) else f"{float(co2):.1f}",
                ]
            )
        story.extend(
            [
                _table(cells, [65, 70, 115, 85, 72, 82]),
                Spacer(1, 8),
                Paragraph(
                    "All values above are simulator-only and use paired held-out seeds. "
                    "Time loss covers completed trips only; generated, inserted, completed, "
                    "unfinished and pending counts expose horizon censoring. Modeled CO2 uses "
                    "the recorded SUMO emissions classes and is not observed air pollution.",
                    sheet["BodyText"],
                ),
            ]
        )
    else:
        story.extend(
            [
                Paragraph("SUMO status", sheet["Heading2"]),
                Paragraph(
                    "NOT VERIFIED: no provenance-checked SUMO benchmark is included.",
                    sheet["BodyText"],
                ),
            ]
        )

    story.extend(
        [
            Spacer(1, 8),
            Paragraph("Limitations and future work", sheet["Heading2"]),
            Paragraph(
                "A single AOI, WorldCover pseudo-labels, residual spatial correlation, and "
                "abstract simulation demand constrain generalization. A calibrated Cairo "
                "digital twin would additionally require validated road geometry, signal "
                "programs, vehicle composition, traffic counts and independent evaluation.",
                sheet["BodyText"],
            ),
            Spacer(1, 8),
            Paragraph("Data attribution", sheet["Heading2"]),
            Paragraph(
                "Contains modified Copernicus Sentinel data; ESA WorldCover © ESA; road data "
                "© OpenStreetMap contributors (ODbL) when used.",
                sheet["BodyText"],
            ),
        ]
    )
    pdf = SimpleDocTemplate(
        str(out),
        pagesize=(595, 842),
        rightMargin=48,
        leftMargin=48,
        topMargin=48,
        bottomMargin=48,
    )
    pdf.build(story)

    report_record = {
        "data_kind": "methodology_only" if include_draft else "verified_experiment_report",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "report_sha256": sha256_file(out),
        "report_file": out.name,
        "inputs": {
            "geoai": None if geo is None else geo["hashes"],
            "traffic": None if traffic is None else traffic["hashes"],
        },
        "warning": (
            "Methodology-only draft; contains no empirical results."
            if include_draft
            else "GeoAI measures WorldCover agreement; traffic and CO2 values are simulator-only."
        ),
    }
    save_json(report_record, out.with_suffix(".provenance.json"))
    return str(out)
