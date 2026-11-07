"""Run with: ``streamlit run src/urbantwin/visualization/dashboard.py``."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import plotly.express as px
import streamlit as st
from pyproj import Transformer

from urbantwin.common import require_real_provenance, sha256_file
from urbantwin.visualization.report_writer import (
    ArtifactIntegrityError,
    validate_geoai_artifacts,
    validate_traffic_artifacts,
)

ROOT = Path(__file__).resolve().parents[3]
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


def _load_raster(path: Path, expected_kind: str):
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


st.set_page_config(page_title="UrbanTwin-AI Research Dashboard", layout="wide")
st.title("UrbanTwin-AI | Geospatial and Mobility Research")
st.info(
    "The geospatial AOI is near Tahrir Square. The generated SUMO intersection is "
    "**synthetic and not calibrated to Cairo**. Satellite land cover and simulated traffic "
    "are not contemporaneous observations."
)
st.caption(
    "Experimental tables are displayed only when their source labels and recorded hashes agree."
)
col1, col2 = st.columns(2)
with col1:
    st.subheader("Earth observation / geospatial analysis")
    source = st.radio(
        "Raster source",
        ["Real (Cairo Sentinel-2/WorldCover)", "Synthetic test fixture"],
        horizontal=False,
    )
    expected_kind = "real" if source.startswith("Real") else "synthetic"
    path = ROOT / ("data/real" if expected_kind == "real" else "data/synthetic")
    try:
        raster = _load_raster(path, expected_kind)
    except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raster = None
        st.error(f"Raster artifact integrity error: {exc}")
    if raster is not None:
        bands, labels, provenance = raster
        st.write("Data source label:", provenance["data_kind"])
        if expected_kind == "synthetic":
            st.warning("Synthetic fixture: software testing only, not scientific evidence")
        rgb = np.zeros((*labels.shape, 3), dtype=np.uint8)
        for class_id, color in COLORS.items():
            rgb[labels == class_id] = color
        caption = (
            "ESA WorldCover reference label raster"
            if expected_kind == "real"
            else "Synthetic label fixture"
        )
        st.image(rgb, caption=caption, width="stretch")
        if expected_kind == "real":
            try:
                from urbantwin.data.spatial_features import summarize

                stats = summarize(
                    bands,
                    labels,
                    float(provenance["configuration"]["resolution_m"]),
                )
                st.json(stats)
            except Exception as exc:  # noqa: BLE001 - one panel must not crash the dashboard
                st.error(f"Spatial summary unavailable: {exc}")
            st.subheader("Georeferenced land-cover map")
            try:
                import pydeck as pdk
                from rasterio.transform import Affine

                transform = Affine(*provenance["transform"])
                to_ll = Transformer.from_crs(
                    provenance["crs"], "EPSG:4326", always_xy=True
                )
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
                roads_file = ROOT / "data/osm/roads.geojson"
                layers = [
                    pdk.Layer(
                        "ScatterplotLayer",
                        samples,
                        get_position="[lon, lat]",
                        get_fill_color="color",
                        get_radius=9,
                        pickable=True,
                    )
                ]
                if roads_file.exists():
                    layers.append(
                        pdk.Layer(
                            "GeoJsonLayer",
                            json.loads(roads_file.read_text(encoding="utf-8")),
                            get_line_color=[40, 40, 40, 220],
                            get_line_width=2,
                            line_width_min_pixels=1,
                        )
                    )
                deck = pdk.Deck(
                    map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
                    initial_view_state=pdk.ViewState(
                        latitude=float(np.mean([sample["lat"] for sample in samples])),
                        longitude=float(np.mean([sample["lon"] for sample in samples])),
                        zoom=13,
                    ),
                    layers=layers,
                    tooltip={"text": "WorldCover class code: {class_id}"},
                )
                st.pydeck_chart(deck, width="stretch")
                st.caption(
                    "© OpenStreetMap contributors where OSM roads/basemap are shown. "
                    "WorldCover colors are an approximate visualization of raster classes."
                )
            except Exception as exc:  # noqa: BLE001 - optional map failures are user-visible
                st.warning(f"Map rendering unavailable: {exc}")
    else:
        command = "urbantwin eo-fetch" if expected_kind == "real" else "urbantwin demo"
        st.warning(f"No complete {expected_kind} raster bundle. Run `{command}`.")

with col2:
    st.subheader("Provenance-verified experiment results")
    try:
        geo = validate_geoai_artifacts(ROOT)
    except ArtifactIntegrityError as exc:
        geo = None
        st.error(f"GeoAI artifacts rejected: {exc}")
    if geo is not None:
        frame = geo["metrics"]
        st.caption("Source label: real; evaluation reference: ESA WorldCover pseudo-labels")
        st.dataframe(frame, width="stretch")
        st.plotly_chart(
            px.bar(
                frame,
                x="method",
                y="macro_f1",
                color="features",
                barmode="group",
                title="Grouped spatial evaluation: macro F1 vs WorldCover labels",
            ),
            width="stretch",
        )
    else:
        st.caption("No provenance-verified real GeoAI benchmark. Run `urbantwin geo-train`.")

    try:
        traffic = validate_traffic_artifacts(ROOT)
    except ArtifactIntegrityError as exc:
        traffic = None
        st.error(f"Traffic artifacts rejected: {exc}")
    if traffic is not None:
        frame = traffic["metrics"]
        source_label = traffic["provenance"]["data_kind"]
        st.caption(f"Source label: {source_label}; all traffic and CO2 values are simulator-only")
        st.dataframe(frame, width="stretch")
        st.plotly_chart(
            px.box(
                frame,
                x="scenario",
                y="mean_time_loss_s_completed",
                color="method",
                title="SUMO completed-trip time loss (paired held-out seeds)",
            ),
            width="stretch",
        )
        horizon_counts = (
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
        st.plotly_chart(
            px.bar(
                horizon_counts,
                x="scenario",
                y="vehicles",
                color="method",
                facet_col="horizon_status",
                barmode="group",
                title="Unfinished inserted and pending-to-insert vehicles",
            ),
            width="stretch",
        )
        if frame["simulated_co2_g_inserted_fleet"].notna().any():
            st.plotly_chart(
                px.box(
                    frame,
                    x="scenario",
                    y="simulated_co2_g_inserted_fleet",
                    color="method",
                    title="Simulator-only SUMO CO2 estimate (recorded emissions classes)",
                ),
                width="stretch",
            )
    else:
        st.caption("SUMO benchmark NOT VERIFIED. Install SUMO, then run `urbantwin traffic-run`.")

    road = ROOT / "results/roads.json"
    if road.exists():
        try:
            road_data = json.loads(road.read_text(encoding="utf-8"))
            st.json(road_data)
        except (OSError, json.JSONDecodeError) as exc:
            st.error(f"Road summary unavailable: {exc}")

st.divider()
st.caption(
    "Research prototype only. WorldCover agreement is not independent land-cover accuracy; "
    "simulation outcomes do not establish real Cairo traffic or emissions performance."
)
