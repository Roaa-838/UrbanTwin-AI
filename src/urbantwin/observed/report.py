"""Publication-style PDF for the provenance-validated observed-traffic study."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from urbantwin.common import sha256_file
from urbantwin.observed.artifacts import validate_observed_artifacts
from urbantwin.observed.mlflow_tracking import observed_run_association
from urbantwin.visualization.report_writer import (
    ArtifactIntegrityError,
    validate_geoai_artifacts,
    validate_traffic_artifacts,
)


def _paragraph(value: Any, style: ParagraphStyle) -> Paragraph:
    return Paragraph(str(value), style)


def _table(
    rows: list[list[Any]],
    widths: list[float],
    body: ParagraphStyle,
    *,
    repeat_rows: int = 1,
) -> Table:
    values = [[_paragraph(cell, body) for cell in row] for row in rows]
    table = Table(values, colWidths=widths, repeatRows=repeat_rows, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17324D")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#A8B4BF")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F6F8")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def _image(path: Path, max_width: float, max_height: float) -> Image:
    image = Image(str(path))
    scale = min(max_width / image.imageWidth, max_height / image.imageHeight)
    image.drawWidth = image.imageWidth * scale
    image.drawHeight = image.imageHeight * scale
    image.hAlign = "CENTER"
    return image


def _page(canvas, document) -> None:
    canvas.saveState()
    width, _ = landscape(A4)
    canvas.setStrokeColor(colors.HexColor("#C9D2D9"))
    canvas.line(16 * mm, 11 * mm, width - 16 * mm, 11 * mm)
    canvas.setFillColor(colors.HexColor("#536675"))
    canvas.setFont("Helvetica", 8)
    canvas.drawString(16 * mm, 6.5 * mm, "UrbanTwin-AI research prototype — source categories kept separate")
    canvas.drawRightString(width - 16 * mm, 6.5 * mm, f"Page {document.page}")
    canvas.restoreState()


def _figure_block(
    path: Path,
    caption: str,
    styles: dict[str, ParagraphStyle],
    *,
    height: float = 126 * mm,
) -> KeepTogether:
    return KeepTogether(
        [
            _image(path, 235 * mm, height),
            Spacer(1, 2 * mm),
            Paragraph(caption, styles["caption"]),
        ]
    )


def make_observed_report(
    project_root: str | Path,
    output_path: str | Path,
    *,
    observed_output: str | Path | None = None,
) -> Path:
    """Build a landscape PDF only from strictly validated local artifacts."""
    root = Path(project_root).resolve()
    observed_dir = (
        Path(observed_output).resolve()
        if observed_output is not None
        else root / "results" / "observed_traffic"
    )
    verified = validate_observed_artifacts(root, observed_dir)
    if verified is None:
        raise FileNotFoundError("No validated observed-traffic study is available")
    provenance = verified["provenance"]
    split = verified["split_manifest"]
    metrics = verified["metrics"]
    classes = verified["vehicle_classes"]
    characterization = verified["characterization"]
    eo_context = json.loads(
        (observed_dir / "historic_eo_context.json").read_text(encoding="utf-8")
    )
    if not isinstance(eo_context, list):
        raise TypeError("Observed EO context must be a JSON list")
    association = observed_run_association(observed_dir, project_root=root)
    road_context = provenance.get("road_context")

    try:
        geoai = validate_geoai_artifacts(root)
    except ArtifactIntegrityError:
        geoai = None
    try:
        abstract_traffic = validate_traffic_artifacts(root)
    except ArtifactIntegrityError:
        abstract_traffic = None

    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    styles_base = getSampleStyleSheet()
    styles = {
        "title": ParagraphStyle(
            "StudyTitle",
            parent=styles_base["Title"],
            fontName="Helvetica-Bold",
            fontSize=23,
            leading=27,
            textColor=colors.HexColor("#17324D"),
            alignment=TA_CENTER,
            spaceAfter=8 * mm,
        ),
        "subtitle": ParagraphStyle(
            "Subtitle",
            parent=styles_base["Normal"],
            fontSize=11,
            leading=15,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#536675"),
            spaceAfter=8 * mm,
        ),
        "h1": ParagraphStyle(
            "H1",
            parent=styles_base["Heading1"],
            fontSize=17,
            leading=21,
            textColor=colors.HexColor("#17324D"),
            spaceBefore=4 * mm,
            spaceAfter=3 * mm,
        ),
        "h2": ParagraphStyle(
            "H2",
            parent=styles_base["Heading2"],
            fontSize=13,
            leading=16,
            textColor=colors.HexColor("#2B6F8A"),
            spaceBefore=3 * mm,
            spaceAfter=2 * mm,
        ),
        "body": ParagraphStyle(
            "Body",
            parent=styles_base["BodyText"],
            fontSize=9.2,
            leading=13,
            alignment=TA_LEFT,
            spaceAfter=2.5 * mm,
        ),
        "small": ParagraphStyle(
            "Small",
            parent=styles_base["BodyText"],
            fontSize=7.6,
            leading=10,
        ),
        "caption": ParagraphStyle(
            "Caption",
            parent=styles_base["BodyText"],
            fontSize=8,
            leading=10.5,
            textColor=colors.HexColor("#536675"),
            alignment=TA_CENTER,
            spaceAfter=3 * mm,
        ),
        "callout": ParagraphStyle(
            "Callout",
            parent=styles_base["BodyText"],
            fontSize=10,
            leading=14,
            textColor=colors.HexColor("#7A3318"),
            borderColor=colors.HexColor("#D7A27E"),
            borderWidth=0.8,
            borderPadding=7,
            backColor=colors.HexColor("#FFF6EE"),
            spaceBefore=2 * mm,
            spaceAfter=4 * mm,
        ),
    }
    document = SimpleDocTemplate(
        str(output),
        pagesize=landscape(A4),
        leftMargin=16 * mm,
        rightMargin=16 * mm,
        topMargin=14 * mm,
        bottomMargin=16 * mm,
        title="Earth Observation, Observed Traffic Demand and Counterfactual Signal Optimization in Greater Cairo",
        author="Roaa Raafat",
        subject="Evidence-bounded Greater Cairo geospatial and measured traffic study",
    )
    story: list[Any] = []
    story.append(
        Paragraph(
            "Earth Observation, Observed Traffic Demand and Counterfactual Signal "
            "Optimization in Greater Cairo",
            styles["title"],
        )
    )
    story.append(
        Paragraph(
            "Evidence-based research prototype · measured link counts and real remote sensing · "
            "no calibrated Cairo digital-twin claim",
            styles["subtitle"],
        )
    )
    generated = datetime.now(ZoneInfo("Africa/Cairo")).isoformat(timespec="seconds")
    run_reference = association.get("run_id") if association else "No compatible local MLflow association"
    story.append(
        _table(
            [
                ["Evidence category", "Verified role", "Status"],
                ["Measured traffic", "TfC directional 20-minute link counts, 2024", "Executed locally"],
                ["Remote sensing", "Sentinel-2 L2A scenes near 31 Oct 2024", "Executed locally"],
                ["Historic reference", "ESA WorldCover 2021", "Not 2024 ground truth"],
                [
                    "Map geometry",
                    "TfC-released survey lines + checksum-validated current OSM graph",
                    "Third-party context",
                ],
                ["Measured-demand SUMO", "Turning/routes/signals unavailable", "NOT RUN — blocked"],
                ["Existing abstract SUMO", "Synthetic one-junction experiment", "Separate simulator evidence"],
            ],
            [42 * mm, 126 * mm, 72 * mm],
            styles["small"],
        )
    )
    story.extend(
        [
            Spacer(1, 5 * mm),
            Paragraph(
                f"Generated {generated}. Code revision {provenance['git']['commit']} "
                f"(dirty={provenance['git']['dirty']}). MLflow run: {run_reference}.",
                styles["caption"],
            ),
            Paragraph("Abstract", styles["h1"]),
            Paragraph(
                "This study validates and normalizes Transport for Cairo directional traffic "
                "counts, characterizes measured demand, and evaluates bounded 20-minute volume "
                "estimation under a frozen joint future-date/unseen-site holdout. It also maps "
                "the selected observed links against real Sentinel-2 L2A context. The data do "
                "not contain turning movements, complete routes, or signal timing; consequently "
                "no measured-demand signal-control simulation was run. Existing abstract SUMO "
                "results are retained as a separate counterfactual software experiment.",
                styles["body"],
            ),
            Paragraph(
                "The TfC portal did not state an explicit reuse license at audit time. This PDF "
                "is a local research artifact and should not be redistributed until TfC confirms "
                "reuse terms.",
                styles["callout"],
            ),
        ]
    )

    story.extend([PageBreak(), Paragraph("Data, provenance and study design", styles["h1"])])
    total_vehicles = int(classes["observed_vehicles"].sum())
    story.append(
        _table(
            [
                ["Item", "Verified value", "Interpretation"],
                ["Directional intervals", "6,618", "Published 20-minute records"],
                ["Observed vehicles", f"{total_vehicles:,}", "Sum across 12 source classes"],
                ["Survey dates", "30 (21 Jul–5 Nov 2024)", "Every directional segment appears on one date"],
                ["Released survey lines", "145", "142 have count rows"],
                ["Inferred analytical sites", "60", "250 m connected components; not official IDs"],
                ["Ambiguous OSM joins", "5 segments / 156 rows", "Excluded from primary spatial benchmark"],
                ["Timezone", "Not provided", "Clock values retained without UTC conversion"],
                ["License", "Unspecified by publisher", "Source and derived row-level data kept outside Git"],
            ],
            [54 * mm, 66 * mm, 120 * mm],
            styles["small"],
        )
    )
    story.extend(
        [
            Paragraph("Site selection", styles["h2"]),
            Paragraph(
                "The 6th of October Bridge pair was selected mechanically because it intersects "
                "the pre-existing Tahrir AOI and has full-day same-date coverage in both direction "
                "codes on 31 October 2024. It is suitable for link-flow and EO context, but it is "
                "a motorway bridge rather than a verified signalized intersection.",
                styles["body"],
            ),
            Paragraph("Evaluation protocol", styles["h2"]),
            Paragraph(
                f"Training uses {split['train_rows']:,} unambiguous intervals across "
                f"{split['train_inferred_sites']} inferred sites and the first 24 survey dates. "
                f"The frozen holdout uses {split['test_rows']:,} intervals across "
                f"{split['test_inferred_sites']} disjoint inferred sites on "
                f"{', '.join(split['test_dates'])}. Five-fold GroupKFold validation is confined "
                "to training. Site-level bootstrap resampling supplies 95% uncertainty intervals.",
                styles["body"],
            ),
            Paragraph(split["warning"], styles["callout"]),
        ]
    )
    map_path = observed_dir / "figures" / "survey_segments_measured_flow.png"
    story.append(
        _figure_block(
            map_path,
            "TfC survey-line geometries colored by measured totals. Geometry is OSM-derived "
            "context released by TfC; this is not an independently validated 2024 road network.",
            styles,
            height=115 * mm,
        )
    )

    if isinstance(road_context, dict):
        connectivity = road_context["connectivity"]
        matching = road_context["map_matching"]
        story.extend(
            [
                PageBreak(),
                Paragraph("Current OSM road context", styles["h1"]),
                _table(
                    [
                        ["Property", "Verified value", "Scientific role"],
                        ["Source", road_context["source_kind"], "Current third-party map data"],
                        ["Directed graph", f"{road_context['nodes']:,} nodes / {road_context['directed_edges']:,} edges", "GIS and conservative map matching"],
                        ["Weak components", str(connectivity["weakly_connected_components"]), f"Largest contains {connectivity['largest_component_node_fraction']:.2%} of nodes"],
                        ["Selected-AOI matches", f"{matching['spatially_reliable']} reliable geometry matches", "No routing direction assigned"],
                        ["Simulation readiness", road_context["simulation_readiness"], "Signal study remains blocked"],
                        ["License", road_context["license"], road_context["attribution"]],
                    ],
                    [56 * mm, 82 * mm, 102 * mm],
                    styles["small"],
                ),
                Paragraph(
                    "The current directed drive graph is checksum-validated and the selected "
                    "TfC links have strong projected geometry support. The publisher's direction "
                    "codes 0/1 have no semantic travel-direction legend, so the matching does not "
                    "create legal routes, turning proportions, or signal observations.",
                    styles["callout"],
                ),
                _figure_block(
                    observed_dir / "figures" / "selected_link_current_osm_context.png",
                    "Selected measured links over the current OSM drive graph. © OpenStreetMap "
                    "contributors, ODbL 1.0. This is map context, not a 2024 traffic observation "
                    "or a calibrated SUMO network.",
                    styles,
                    height=105 * mm,
                ),
            ]
        )

    story.extend([PageBreak(), Paragraph("Measured demand", styles["h1"])])
    story.append(
        _figure_block(
            observed_dir / "figures" / "selected_link_measured_flow.png",
            "Actual measured 20-minute volumes for the selected directional link pair. Direction "
            "codes 0 and 1 are retained because the publisher does not define their semantic names.",
            styles,
        )
    )
    class_rows = [["Vehicle class", "Observed vehicles", "Share"]]
    for row in classes.sort_values("observed_vehicles", ascending=False).itertuples(index=False):
        class_rows.append(
            [row.vehicle_class, f"{int(row.observed_vehicles):,}", f"{row.share_of_observed_vehicles:.2%}"]
        )
    story.extend(
        [
            Paragraph("Observed class composition", styles["h2"]),
            _table(class_rows, [78 * mm, 64 * mm, 48 * mm], styles["small"]),
            Paragraph(
                f"The daily characterization table contains {len(characterization)} survey-date "
                "rows. These are descriptive measurements, not annualized demand estimates.",
                styles["body"],
            ),
        ]
    )

    story.extend([PageBreak(), Paragraph("Earth-observation context", styles["h1"])])
    if eo_context:
        eo_rows = [["Buffer", "Valid coverage", "Mean NDVI", "Mean NDBI", "Built-up (WC21)", "Vegetated (WC21)"]]
        for row in eo_context:
            eo_rows.append(
                [
                    f"{row['buffer_m']} m",
                    f"{row['valid_pixel_coverage']:.2%}",
                    f"{row['mean_ndvi']:.3f}",
                    f"{row['mean_ndbi']:.3f}",
                    f"{row['worldcover_built_up_fraction']:.2%}",
                    f"{row['worldcover_vegetated_fraction']:.2%}",
                ]
            )
        story.append(_table(eo_rows, [30 * mm] * 6, styles["small"]))
        first = eo_context[0]
        story.append(
            Paragraph(
                "Sentinel-2 items: " + ", ".join(first["sentinel_item_ids"]) + ". "
                "WorldCover 2021 is used only as a historic reference product. NDVI/NDBI and "
                "land-cover fractions are descriptive spatial associations, not causal predictors "
                "of traffic demand.",
                styles["body"],
            )
        )
        story.append(
            _figure_block(
                observed_dir / "figures" / "selected_link_historic_eo_context.png",
                "Selected measured links over the real near-date Sentinel-2 NDVI composite. "
                "The filename retains 'historic' because WorldCover 2021 remains the label reference.",
                styles,
            )
        )

    story.extend([PageBreak(), Paragraph("Predictive benchmark", styles["h1"])])
    measurement = metrics.loc[
        metrics["information_budget"] == "measurement_free_estimation"
    ].sort_values("mae")
    metric_rows = [["Model", "Features", "MAE (95% site CI)", "RMSE (95% site CI)", "Mean bias"]]
    for row in measurement.itertuples(index=False):
        metric_rows.append(
            [
                row.model.replace("_", " "),
                row.feature_set.replace("_", " "),
                f"{row.mae:.2f} ({row.mae_ci95_low:.2f}, {row.mae_ci95_high:.2f})",
                f"{row.rmse:.2f} ({row.rmse_ci95_low:.2f}, {row.rmse_ci95_high:.2f})",
                f"{row.mean_bias:.2f}",
            ]
        )
    story.append(
        _table(metric_rows, [52 * mm, 40 * mm, 58 * mm, 58 * mm, 34 * mm], styles["small"])
    )
    best = measurement.iloc[0]
    story.append(
        Paragraph(
            f"The lowest measurement-free holdout MAE was {best['mae']:.2f} vehicles per "
            f"20-minute interval for {best['model'].replace('_', ' ')} with "
            f"{best['feature_set'].replace('_', ' ')} features. This is a bounded prototype "
            "result under a joint date/site shift; it is not evidence of causal EO contribution "
            "or general deployment accuracy.",
            styles["body"],
        )
    )
    story.append(
        _figure_block(
            observed_dir / "figures" / "observed_model_comparison.png",
            "Measurement-free models only. Error bars are 95% site-bootstrap intervals on the "
            "frozen joint future-date/unseen-site holdout.",
            styles,
            height=108 * mm,
        )
    )
    story.append(
        _figure_block(
            observed_dir / "figures" / "observed_vs_estimated.png",
            "Observed versus estimated volume for the best measurement-free model. Exact date, "
            "site IDs, road names, current counts, and opposite-direction counts are excluded.",
            styles,
            height=108 * mm,
        )
    )

    story.extend([PageBreak(), Paragraph("Online persistence is a different task", styles["h1"])])
    persistence = metrics.loc[metrics["model"] == "persistence"].iloc[0]
    story.append(
        Paragraph(
            f"Persistence uses the previous measured interval and is available for "
            f"{int(persistence['rows']):,} exactly adjacent holdout records. Its MAE was "
            f"{persistence['mae']:.2f} and RMSE {persistence['rmse']:.2f}. It is not a fair "
            "head-to-head comparison with measurement-free estimation because it has access to "
            "a recent observation. No gaps were filled.",
            styles["callout"],
        )
    )
    story.append(
        _figure_block(
            observed_dir / "figures" / "online_persistence_diagnostic.png",
            "One-step online persistence for exactly adjacent 20-minute observations. This panel "
            "is deliberately separated from the measurement-free model comparison.",
            styles,
        )
    )

    story.extend([PageBreak(), Paragraph("Measured-demand signal simulation gate", styles["h1"])])
    blockers = provenance["study_gate"]["blockers"]
    story.append(
        Paragraph(
            "No observed-demand-informed signal optimization was executed. The measured links "
            "cannot identify a complete intersection demand matrix, and the selected site is not "
            "a verified signalized junction.",
            styles["callout"],
        )
    )
    story.append(
        _table(
            [["Blocker", "Consequence"]]
            + [[blocker, "Required before a defensible measured-demand controller study"] for blocker in blockers],
            [112 * mm, 128 * mm],
            styles["small"],
        )
    )
    story.append(Paragraph("Existing abstract SUMO experiment", styles["h2"]))
    if abstract_traffic is not None:
        story.append(
            Paragraph(
                "The existing full abstract benchmark remains provenance-valid and separate. It "
                "uses a synthetic one-junction network and generated demand. Its modeled CO₂ and "
                "traffic KPIs are simulator outputs, not measured Cairo environmental or mobility "
                "performance.",
                styles["body"],
            )
        )
        for filename, caption in (
            ("traffic_time_loss.png", "Abstract SUMO completed-trip time loss; simulator-only."),
            ("traffic_simulated_co2.png", "Abstract SUMO modeled CO₂; not an environmental measurement."),
        ):
            path = root / "results" / "traffic" / "figures" / filename
            if path.is_file():
                story.append(_figure_block(path, caption, styles, height=102 * mm))
    else:
        story.append(Paragraph("No validated abstract SUMO bundle was available.", styles["body"]))

    story.extend([PageBreak(), Paragraph("Existing real EO / WorldCover benchmark", styles["h1"])])
    if geoai is not None:
        story.append(
            Paragraph(
                "The separate 2021 GeoAI experiment measures agreement with ESA WorldCover "
                "pseudo-labels under repeated grouped spatial evaluation. It is not independent "
                "land-cover accuracy and is not a traffic-demand model.",
                styles["body"],
            )
        )
        confusion = root / "results" / "geoai" / "figures" / "geoai_confusion_matrix.png"
        comparison = root / "results" / "geoai" / "figures" / "geoai_model_comparison.png"
        if confusion.is_file():
            story.append(
                _figure_block(
                    confusion,
                    "Per-class confusion matrices against WorldCover 2021 reference labels.",
                    styles,
                    height=110 * mm,
                )
            )
        if comparison.is_file():
            story.append(
                _figure_block(
                    comparison,
                    "Grouped spatial agreement benchmark; separate from observed traffic analysis.",
                    styles,
                    height=105 * mm,
                )
            )

    story.extend(
        [
            PageBreak(),
            Paragraph("Limitations, ethics and reproducibility", styles["h1"]),
            _table(
                [
                    ["Topic", "Evidence-bounded statement"],
                    ["Traffic coverage", "One survey date per directional segment; date and site effects are confounded."],
                    ["Site identity", "The 60 groups are reproducible 250 m clusters, not official TfC location IDs."],
                    ["Map matching", "Five source OSM-ID conflicts are excluded. Current OSM supports selected-link geometry, but direction codes remain unresolved."],
                    ["Earth observation", "2024 Sentinel-2 is descriptive; WorldCover 2021 remains a historic reference."],
                    ["Causality", "No causal effect of EO/road context on traffic is inferred."],
                    ["Signal control", "No turning counts or observed signal plan; measured-demand SUMO is blocked."],
                    ["Privacy", "The release contains aggregate directional counts, not identifiable trajectories."],
                    ["Reuse", "TfC license is unspecified; row-level and derived local artifacts remain outside Git."],
                ],
                [52 * mm, 188 * mm],
                styles["small"],
            ),
            Paragraph("Reproducibility identifiers", styles["h2"]),
            _table(
                [
                    ["Identifier", "Value"],
                    ["Count archive SHA-256", provenance["source_counts_sha256"]],
                    ["Location archive SHA-256", provenance["source_locations_sha256"]],
                    ["Configuration SHA-256", provenance["config_sha256"]],
                    ["Split manifest SHA-256", provenance["split_manifest_sha256"]],
                    [
                        "Current OSM manifest SHA-256",
                        road_context["manifest_sha256"] if isinstance(road_context, dict) else "Not included",
                    ],
                    ["Code revision", provenance["git"]["commit"]],
                    ["MLflow run", run_reference],
                ],
                [62 * mm, 178 * mm],
                styles["small"],
            ),
            Paragraph("Sources and attribution", styles["h2"]),
            Paragraph(
                'Transport for Cairo: <link href="https://data.transportforcairo.com/catalogue/#/document/190">traffic counts</link> and '
                '<link href="https://data.transportforcairo.com/catalogue/#/dataset/193">survey locations</link>. '
                'Contains modified Copernicus Sentinel data; ESA WorldCover 2021 © ESA, CC BY 4.0. '
                'Road geometry is OSM-derived and attributable to OpenStreetMap contributors under ODbL. '
                'Abstract simulation uses Eclipse SUMO. Source-data terms remain separate from the MIT code license.',
                styles["body"],
            ),
        ]
    )

    document.build(story, onFirstPage=_page, onLaterPages=_page)
    included_paths = [
        observed_dir / relative for relative in provenance["artifact_hashes"]
    ] + [root / "configs" / "observed_traffic.yaml"]
    if isinstance(provenance.get("eo_context"), dict):
        eo_root = root / provenance["eo_context"]["input_directory"]
        included_paths.extend([eo_root / "provenance.json", eo_root / "eo_stack.npz"])
    if isinstance(road_context, dict):
        road_root = root / road_context["input_directory"]
        included_paths.append(road_root / "road_manifest.json")
    sidecar = {
        "schema_version": "urbantwin.observed-report/v1",
        "generated_cairo_time": generated,
        "data_kind": "research_report_mixed_sources_explicitly_separated",
        "observed_provenance_sha256": sha256_file(observed_dir / "provenance.json"),
        "input_hashes": {
            str(path.relative_to(root)).replace("\\", "/"): sha256_file(path)
            for path in included_paths
            if path.is_file()
        },
        "mlflow_run_id": association.get("run_id") if association else None,
        "report_sha256": sha256_file(output),
        "redistribution_warning": (
            "TfC license status is unspecified; do not redistribute this local measured-data "
            "report until reuse terms are confirmed."
        ),
    }
    sidecar_path = output.with_suffix(".provenance.json")
    sidecar_path.write_text(
        json.dumps(sidecar, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output


__all__ = ["make_observed_report"]
