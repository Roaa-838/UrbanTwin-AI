"""Run with: ``streamlit run src/urbantwin/visualization/dashboard.py``."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from pyproj import Transformer

from urbantwin.common import require_real_provenance, sha256_file
from urbantwin.observed.artifacts import (
    ObservedArtifactIntegrityError,
    validate_observed_artifacts,
)
from urbantwin.observed.roads import ObservedRoadError, validate_observed_road_bundle
from urbantwin.visualization.report_writer import (
    ArtifactIntegrityError,
    validate_geoai_artifacts,
    validate_traffic_artifacts,
)

ROOT = Path(__file__).resolve().parents[3]
OBSERVED_OUTPUT = ROOT / "results/observed_traffic"
OBSERVED_SOURCE = ROOT / "data/observed/tfc/normalized_v1"
OBSERVED_ROADS = ROOT / "data/observed/osm/current_tahrir"
COLORS = {
    10: [0, 100, 0],
    20: [255, 187, 34],
    30: [255, 255, 76],
    40: [240, 150, 255],
    50: [250, 0, 0],
    60: [180, 180, 180],
    70: [240, 240, 240],
    80: [0, 100, 200],
    90: [0, 150, 160],
    95: [0, 207, 117],
    100: [250, 230, 160],
}


@st.cache_data(show_spinner=False)
def _load_raster(path_text: str, expected_kind: str):
    path = Path(path_text)
    stack_path = path / "eo_stack.npz"
    provenance_path = path / "provenance.json"
    if not stack_path.is_file() and not provenance_path.is_file():
        return None
    if not stack_path.is_file() or not provenance_path.is_file():
        raise ValueError("Raster stack and provenance must be generated together")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if not isinstance(provenance, dict):
        raise TypeError("Raster provenance must be a JSON object")
    if provenance.get("data_kind") != expected_kind:
        raise ValueError(
            f"Expected {expected_kind!r} provenance, found {provenance.get('data_kind')!r}"
        )
    if expected_kind == "real":
        require_real_provenance(provenance)
    recorded_hash = provenance.get("eo_stack_sha256")
    if not isinstance(recorded_hash, str) or recorded_hash != sha256_file(stack_path):
        raise ValueError("Raster stack does not match its recorded provenance hash")
    with np.load(stack_path, allow_pickle=False) as archive:
        bands = archive["bands"].copy()
        labels = archive["labels"].copy()
    if bands.ndim != 3 or bands.shape[0] != 5 or labels.ndim != 2:
        raise ValueError("Expected five raster bands and one two-dimensional label grid")
    if bands.shape[1:] != labels.shape:
        raise ValueError("Raster bands and labels are not aligned")
    if provenance.get("band_order") != ["B02", "B03", "B04", "B08", "B11"]:
        raise ValueError("Raster provenance has an unsupported band order")
    if provenance.get("height") != labels.shape[0] or provenance.get("width") != labels.shape[1]:
        raise ValueError("Raster dimensions do not match provenance")
    return bands, labels, provenance


@st.cache_data(show_spinner=False)
def _load_observed_bundle(
    output_text: str,
    source_text: str,
    provenance_mtime_ns: int,
    source_mtime_ns: int,
) -> dict[str, Any] | None:
    del provenance_mtime_ns, source_mtime_ns
    return validate_observed_artifacts(
        ROOT,
        Path(output_text),
        source_dir=Path(source_text),
    )


def _observed_bundle() -> dict[str, Any] | None:
    provenance = OBSERVED_OUTPUT / "provenance.json"
    source = OBSERVED_SOURCE / "source_manifest.json"
    marker = provenance.stat().st_mtime_ns if provenance.is_file() else 0
    source_marker = source.stat().st_mtime_ns if source.is_file() else 0
    return _load_observed_bundle(
        str(OBSERVED_OUTPUT),
        str(OBSERVED_SOURCE),
        marker,
        source_marker,
    )


def _load_verified_observed() -> dict[str, Any] | None:
    try:
        return _observed_bundle()
    except (ObservedArtifactIntegrityError, OSError, TypeError, ValueError) as exc:
        st.error(f"Observed-traffic artifacts rejected: {exc}", icon=":material/error:")
        return None


def _render_eo_map(labels: np.ndarray, provenance: dict[str, Any]) -> None:
    try:
        import pydeck as pdk
        from rasterio.transform import Affine

        transform = Affine(*provenance["transform"])
        to_ll = Transformer.from_crs(provenance["crs"], "EPSG:4326", always_xy=True)
        skip = max(1, int(max(labels.shape) / 70))
        samples = []
        for row in range(0, labels.shape[0], skip):
            for column in range(0, labels.shape[1], skip):
                class_id = int(labels[row, column])
                if class_id not in COLORS:
                    continue
                x, y = transform * (column + 0.5, row + 0.5)
                longitude, latitude = to_ll.transform(x, y)
                samples.append(
                    {
                        "lon": longitude,
                        "lat": latitude,
                        "color": COLORS[class_id],
                        "class_id": str(class_id),
                    }
                )
        if not samples:
            raise ValueError("No supported WorldCover classes are available to map")
        deck = pdk.Deck(
            map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
            initial_view_state=pdk.ViewState(
                latitude=float(np.mean([sample["lat"] for sample in samples])),
                longitude=float(np.mean([sample["lon"] for sample in samples])),
                zoom=13,
            ),
            layers=[
                pdk.Layer(
                    "ScatterplotLayer",
                    samples,
                    get_position="[lon, lat]",
                    get_fill_color="color",
                    get_radius=9,
                    pickable=True,
                )
            ],
            tooltip={"text": "WorldCover class code: {class_id}"},
        )
        st.pydeck_chart(
            deck,
            width="stretch",
            alt="Tahrir-area WorldCover 2021 reference classes on a geographic basemap",
        )
        st.caption(
            "Basemap © OpenStreetMap contributors and CARTO. WorldCover colors are an "
            "approximate display of the 2021 categorical reference product."
        )
    except Exception as exc:  # noqa: BLE001 - optional map failure stays within this panel
        st.warning(f"Map rendering unavailable: {exc}", icon=":material/map:")


def _observed_eo_section() -> None:
    st.header("Observed Earth observation", icon=":material/satellite_alt:")
    st.badge("Real remote sensing", color="green", icon=":material/verified:")
    st.caption(
        "Real Sentinel-2 L2A surface reflectance covers the configured Tahrir-area AOI. "
        "ESA WorldCover 2021 remains a historic reference product, not 2024 ground truth."
    )
    raster_choice = st.segmented_control(
        "EO bundle",
        ["Near-survey Sentinel-2 (2024)", "Historic GeoAI bundle (2021)"],
        default="Near-survey Sentinel-2 (2024)",
        key="eo_bundle",
        width="stretch",
    )
    raster_path = (
        ROOT / "data/observed/eo_2024"
        if raster_choice.startswith("Near-survey")
        else ROOT / "data/real"
    )
    try:
        raster = _load_raster(str(raster_path), "real")
    except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raster = None
        st.error(f"Raster artifact integrity error: {exc}", icon=":material/error:")
    if raster is None:
        st.warning(
            "No complete provenance-verified raster bundle. Run `urbantwin observed-eo-fetch` "
            "for 2024 context or `urbantwin eo-fetch` for the historic GeoAI bundle.",
            icon=":material/download:",
        )
        return

    _bands, labels, provenance = raster
    valid_count = int(provenance["valid_pixel_count"])
    grid_count = int(provenance["grid_pixel_count"])
    metrics = st.columns(4)
    period = provenance["configuration"].get("sentinel_period", ["unknown", "unknown"])
    metrics[0].metric("Sentinel period", f"{period[0]} to {period[1]}")
    metrics[1].metric("Valid-pixel coverage", f"{valid_count / grid_count:.1%}")
    metrics[2].metric("Projected grid", f"{provenance['width']} × {provenance['height']}")
    metrics[3].metric("CRS", str(provenance["crs"]))

    rgb = np.zeros((*labels.shape, 3), dtype=np.uint8)
    for class_id, color in COLORS.items():
        rgb[labels == class_id] = color
    st.image(
        rgb,
        caption="ESA WorldCover 2021 historic reference raster",
        width="stretch",
        alt="WorldCover 2021 land-cover classes for the Tahrir-area raster grid",
    )
    _render_eo_map(labels, provenance)

    try:
        geo = validate_geoai_artifacts(ROOT)
    except ArtifactIntegrityError as exc:
        geo = None
        st.error(f"GeoAI artifacts rejected: {exc}", icon=":material/error:")
    if geo is not None:
        import altair as alt

        frame = geo["metrics"].copy()
        st.subheader("Separate 2021 WorldCover agreement benchmark")
        st.caption(
            "Source: the validated 2021 Sentinel-2 bundle, regardless of the map choice above. "
            "Evaluation reference: WorldCover 2021 "
            "pseudo-labels. Agreement is not independent land-cover accuracy."
        )
        st.dataframe(
            frame,
            width="stretch",
            hide_index=True,
            alt="Repeated grouped spatial WorldCover agreement metrics",
        )
        chart = (
            alt.Chart(frame)
            .mark_bar()
            .encode(
                x=alt.X("method:N", title="Model"),
                y=alt.Y("macro_f1:Q", title="Macro F1 versus WorldCover"),
                color=alt.Color("features:N", title="Feature set"),
                xOffset="features:N",
                tooltip=["method:N", "features:N", alt.Tooltip("macro_f1:Q", format=".3f")],
            )
            .properties(title="Repeated grouped spatial evaluation")
        )
        st.altair_chart(
            chart,
            width="stretch",
            alt="Macro F1 by model and feature set against WorldCover 2021 pseudo-labels",
        )

    observed = _load_verified_observed()
    if observed is not None:
        context_path = Path(observed["output_dir"]) / "historic_eo_context.json"
        context = pd.DataFrame(json.loads(context_path.read_text(encoding="utf-8")))
        st.subheader("Selected observed-link context")
        st.caption(
            "Buffers around the selected 6th of October Bridge links use real Sentinel-2 L2A "
            "scenes from 24 October to 3 November 2024. WorldCover 2021 remains historic; "
            "these descriptors are not causal traffic predictors."
        )
        st.dataframe(
            context[
                [
                    "buffer_m",
                    "valid_pixel_coverage",
                    "mean_ndvi",
                    "mean_ndbi",
                    "worldcover_built_up_fraction",
                    "worldcover_vegetated_fraction",
                ]
            ],
            width="stretch",
            hide_index=True,
            alt="Near-survey 2024 Sentinel and historic WorldCover summaries for selected links",
        )


def _observed_traffic_section() -> None:
    st.header("Observed TfC traffic", icon=":material/traffic:")
    st.badge("Measured directional link counts", color="green", icon=":material/verified:")
    bundle = _load_verified_observed()
    if bundle is None:
        st.warning(
            "No complete provenance-verified observed study. Prepare the official local files "
            "and run `urbantwin observed-study`.",
            icon=":material/folder_off:",
        )
        return

    provenance = bundle["provenance"]
    split = bundle["split_manifest"]
    license_record = bundle["source_manifest"]["license"]
    st.caption(
        "Source label: measured TfC directional link counts (2024). Geometry: TfC-released, "
        "OSM-derived survey lines from a 2026 extract."
    )
    st.warning(
        "TfC attribution is recorded, but the portal did not state an explicit reuse license. "
        "Row-level data and derived artifacts remain local and untracked.",
        icon=":material/license:",
    )
    summary = st.columns(4)
    summary[0].metric("Measured intervals", "6,618")
    summary[1].metric("Observed vehicles", "2,961,483")
    summary[2].metric("Inferred site groups", "60")
    summary[3].metric("License", str(license_record["status"]).replace("_", " "))

    with st.container(border=True):
        st.subheader("Frozen evaluation protocol")
        st.markdown(
            f"Training uses **{split['train_rows']:,}** eligible intervals from "
            f"**{split['train_inferred_sites']}** inferred sites. The frozen holdout uses "
            f"**{split['test_rows']:,}** intervals from **{split['test_inferred_sites']}** "
            "unseen inferred sites."
        )
        st.warning(str(split["warning"]), icon=":material/warning:")
        st.caption(
            f"{split['excluded_ambiguous_rows']} rows attached to five conflicting OSM joins "
            "were excluded from the primary spatial benchmark, but retained in descriptive totals."
        )

    metrics = bundle["metrics"].copy()
    measurement_free = metrics.loc[
        metrics["information_budget"] == "measurement_free_estimation"
    ].sort_values("mae")
    persistence = metrics.loc[metrics["model"] == "persistence"]
    st.subheader("Measurement-free volume estimation")
    st.caption(
        "Models use clock/time and static released road-geometry attributes only. Current-period "
        "counts, opposite-direction counts, IDs, names, exact dates and future rows are excluded."
    )
    display_columns = [
        "model",
        "feature_set",
        "rows",
        "mae",
        "mae_ci95_low",
        "mae_ci95_high",
        "rmse",
        "rmse_ci95_low",
        "rmse_ci95_high",
        "mean_bias",
    ]
    st.dataframe(
        measurement_free[display_columns],
        width="stretch",
        hide_index=True,
        alt="Frozen holdout metrics for measurement-free directional volume estimation",
    )
    import altair as alt

    chart_frame = measurement_free.assign(
        model_feature=lambda value: value["model"] + " · " + value["feature_set"]
    )
    bars = (
        alt.Chart(chart_frame)
        .mark_bar()
        .encode(
            x=alt.X("model_feature:N", sort="-y", title="Model and feature set"),
            y=alt.Y("mae:Q", title="MAE (vehicles / 20 min)"),
            color=alt.Color("feature_set:N", title="Feature set"),
            tooltip=[
                "model:N",
                "feature_set:N",
                alt.Tooltip("mae:Q", format=".2f"),
                alt.Tooltip("mae_ci95_low:Q", format=".2f"),
                alt.Tooltip("mae_ci95_high:Q", format=".2f"),
            ],
        )
    )
    errors = (
        alt.Chart(chart_frame)
        .mark_rule(color="black")
        .encode(
            x=alt.X("model_feature:N", sort="-y"),
            y="mae_ci95_low:Q",
            y2="mae_ci95_high:Q",
        )
    )
    st.altair_chart(
        (bars + errors).properties(title="Joint future-date and unseen-site holdout"),
        width="stretch",
        alt="Measurement-free MAE with site-bootstrap 95 percent intervals",
    )

    with st.container(border=True):
        st.subheader("Online one-step persistence")
        st.caption(
            "This separate task uses the immediately previous measured interval and is therefore "
            "not directly comparable with measurement-free estimation. Gaps are never filled."
        )
        if len(persistence) == 1:
            row = persistence.iloc[0]
            cards = st.columns(3)
            cards[0].metric("Eligible intervals", f"{int(row['rows']):,}")
            cards[1].metric("Persistence MAE", f"{float(row['mae']):.2f}")
            cards[2].metric("Persistence RMSE", f"{float(row['rmse']):.2f}")

    figure_root = Path(bundle["output_dir"]) / "figures"
    visualizations = [
        (
            "selected_link_measured_flow.png",
            "Measured 20-minute directional flow at the selected observed-link pair",
        ),
        (
            "survey_segments_measured_flow.png",
            "TfC survey segments colored by measured vehicle totals",
        ),
        (
            "selected_link_current_osm_context.png",
            "Selected TfC observed links over the checksum-validated current OSM drive graph",
        ),
    ]
    for filename, alt_text in visualizations:
        path = figure_root / filename
        if path.is_file():
            st.image(path, width="stretch", alt=alt_text)
    st.caption(
        f"Configuration hash: `{provenance['config_sha256']}` · Code revision: "
        f"`{provenance['git']['commit']}` · MLflow is optional and does not alter local evidence."
    )


def _measured_simulation_section() -> None:
    st.header("Measured-demand simulation", icon=":material/block:")
    st.badge("Not run", color="orange", icon=":material/pending:")
    st.warning(
        "A measured-demand-informed signal experiment is scientifically blocked. The selected "
        "observed site is a motorway link, and the release has no turning movements or signal "
        "phase/timing observations.",
        title="No calibrated Cairo signal model",
        icon=":material/warning:",
    )
    blockers = [
        "No intersection turning movements",
        "No observed signal phases or timings",
        "No defensible complete route or turning-demand matrix",
        "Selected observed link is a motorway bridge, not a verified signalized junction",
    ]
    st.markdown("\n".join(f"- {item}" for item in blockers))
    try:
        roads = validate_observed_road_bundle(OBSERVED_ROADS)
    except ObservedRoadError as exc:
        roads = None
        st.error(f"Current OSM road bundle rejected: {exc}", icon=":material/error:")
    if roads is not None:
        connectivity = roads["connectivity"]
        road_cards = st.columns(4)
        road_cards[0].metric("Current OSM nodes", f"{int(roads['nodes']):,}")
        road_cards[1].metric("Directed edges", f"{int(roads['directed_edges']):,}")
        road_cards[2].metric(
            "Weak components", f"{int(connectivity['weakly_connected_components']):,}"
        )
        road_cards[3].metric(
            "Largest component",
            f"{float(connectivity['largest_component_node_fraction']):.1%}",
        )
        st.caption(
            "Current map evidence: © OpenStreetMap contributors, ODbL 1.0. The selected "
            "survey lines have projected geometry support, but TfC direction codes 0/1 have "
            "no published travel-direction legend. This graph is not a signal model."
        )

    categories = pd.DataFrame(
        [
            {
                "Category": "Measured input",
                "Status": "Available",
                "Evidence": "TfC directional 20-minute link counts",
            },
            {
                "Category": "Remote sensing",
                "Status": "Historic context",
                "Evidence": "Sentinel-2 L2A 2024; WorldCover 2021 historic reference",
            },
            {
                "Category": "Third-party map data",
                "Status": "Validated current context" if roads is not None else "Not available",
                "Evidence": (
                    "Current directed OSM drive graph and conservative geometry matches; "
                    "routing direction remains unresolved"
                    if roads is not None
                    else "No checksum-validated current OSM bundle"
                ),
            },
            {
                "Category": "Inferred parameter",
                "Status": "Analytical only",
                "Evidence": "250 m connected-component site groups",
            },
            {
                "Category": "Measured-demand SUMO variables",
                "Status": "Not created",
                "Evidence": "Routes, turns, phases, driver behavior and pedestrian demand unresolved",
            },
        ]
    )
    st.dataframe(
        categories,
        width="stretch",
        hide_index=True,
        alt="Evidence categories and availability for a measured-demand simulation",
    )
    st.caption(
        "Research prototype only. No traffic calibration, controller deployment, causal benefit, "
        "or calibrated Greater Cairo digital twin is claimed."
    )


def _abstract_simulation_section() -> None:
    st.header("Existing abstract SUMO experiment", icon=":material/model_training:")
    st.badge("Synthetic simulation", color="blue", icon=":material/science:")
    st.caption(
        "This is the existing abstract four-way SUMO benchmark. It is separate from measured "
        "Cairo traffic and is not calibrated to the TfC observations."
    )
    try:
        traffic = validate_traffic_artifacts(ROOT)
    except ArtifactIntegrityError as exc:
        traffic = None
        st.error(f"Traffic artifacts rejected: {exc}", icon=":material/error:")
    if traffic is None:
        st.warning(
            "No complete provenance-verified abstract benchmark. Run `urbantwin traffic-run --resume`.",
            icon=":material/play_arrow:",
        )
        return

    import altair as alt

    frame = traffic["metrics"].copy()
    source_label = traffic["provenance"]["data_kind"]
    st.caption(
        f"Source label: {source_label}. All traffic and CO₂ values below are simulator-only."
    )
    st.dataframe(
        frame,
        width="stretch",
        hide_index=True,
        alt="Per-seed metrics for the provenance-verified abstract SUMO benchmark",
    )
    time_loss = (
        alt.Chart(frame)
        .mark_boxplot()
        .encode(
            x=alt.X("scenario:N", title="Synthetic demand scenario"),
            y=alt.Y(
                "mean_time_loss_s_completed:Q",
                title="Completed-trip time loss (s)",
            ),
            color=alt.Color("method:N", title="Signal method"),
            tooltip=["scenario:N", "method:N"],
        )
        .properties(title="Paired held-out simulation seeds")
    )
    st.altair_chart(
        time_loss,
        width="stretch",
        alt="Completed-trip time loss by synthetic scenario and signal method",
    )
    horizon = (
        frame.groupby(["scenario", "method"], as_index=False)[
            ["unfinished_vehicles", "pending_vehicles"]
        ]
        .mean()
        .melt(
            id_vars=["scenario", "method"],
            var_name="horizon_status",
            value_name="vehicles",
        )
    )
    horizon_chart = (
        alt.Chart(horizon)
        .mark_bar()
        .encode(
            x=alt.X("scenario:N", title="Synthetic demand scenario"),
            y=alt.Y("vehicles:Q", title="Mean vehicles at horizon"),
            color=alt.Color("method:N", title="Signal method"),
            xOffset="method:N",
            column=alt.Column("horizon_status:N", title=None),
            tooltip=["scenario:N", "method:N", "horizon_status:N", "vehicles:Q"],
        )
        .properties(title="Unfinished and pending-to-insert vehicles")
    )
    st.altair_chart(
        horizon_chart,
        width="stretch",
        alt="Unfinished inserted and pending-to-insert vehicles in abstract SUMO runs",
    )
    if frame["simulated_co2_g_inserted_fleet"].notna().any():
        co2_chart = (
            alt.Chart(frame)
            .mark_boxplot()
            .encode(
                x=alt.X("scenario:N", title="Synthetic demand scenario"),
                y=alt.Y(
                    "simulated_co2_g_inserted_fleet:Q",
                    title="SUMO-modeled CO₂ (g)",
                ),
                color=alt.Color("method:N", title="Signal method"),
            )
            .properties(title="Simulator-only emissions-class model")
        )
        st.altair_chart(
            co2_chart,
            width="stretch",
            alt="SUMO-modeled CO2 by synthetic scenario and signal method",
        )
    st.caption(
        "Modeled CO₂ is not observed environmental data. Simulation outcomes do not establish "
        "real Greater Cairo controller performance."
    )


st.set_page_config(
    page_title="UrbanTwin-AI research dashboard",
    page_icon=":material/location_city:",
    layout="wide",
)
st.title("UrbanTwin-AI research evidence", icon=":material/location_city:")
st.caption(
    "Earth observation, measured directional traffic demand and explicitly separate "
    "counterfactual simulation evidence for Greater Cairo research."
)
st.info(
    "This is a research prototype, not a deployed traffic controller or a calibrated Cairo "
    "digital twin. Every section retains its own measured, remotely sensed, mapped, inferred, "
    "or simulated source label.",
    icon=":material/info:",
)

SECTIONS = [
    "Observed EO",
    "Observed TfC traffic",
    "Measured-demand simulation",
    "Abstract simulation",
]
section = st.segmented_control(
    "Research section",
    SECTIONS,
    default=SECTIONS[0],
    required=True,
    key="research_section",
    width="stretch",
    wrap=True,
)

if section == "Observed TfC traffic":
    _observed_traffic_section()
elif section == "Measured-demand simulation":
    _measured_simulation_section()
elif section == "Abstract simulation":
    _abstract_simulation_section()
else:
    _observed_eo_section()
