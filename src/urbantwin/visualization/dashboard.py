"""Run with: streamlit run src/urbantwin/visualization/dashboard.py"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import pydeck as pdk
from pyproj import Transformer

ROOT=Path(__file__).resolve().parents[3]
COLORS={10:[0,100,0],20:[255,187,34],30:[255,255,76],40:[240,150,255],
        50:[250,0,0],60:[180,180,180],70:[240,240,240],80:[0,100,200],
        90:[0,150,160],95:[0,207,117],100:[250,230,160]}

st.set_page_config(page_title="UrbanTwin-AI Research Dashboard",layout="wide")
st.title("UrbanTwin-AI | Geospatial and Mobility Research")
st.info("The geospatial AOI is near Tahrir Square. The default SUMO four-way intersection is **synthetic and not calibrated to Cairo**. Satellite land cover and simulated traffic are NOT contemporaneous observations.")
st.caption("Only result files produced by executable pipelines are displayed; no fabricated benchmark scores.")
col1,col2=st.columns(2)
with col1:
    st.subheader("Earth observation / geospatial analysis")
    source=st.radio("Raster source",["Real (Cairo Sentinel-2/WorldCover)","Synthetic test fixture"],horizontal=False)
    path=ROOT/("data/real" if source.startswith("Real") else "data/synthetic")
    if (path/"eo_stack.npz").exists():
        provenance=json.loads((path/"provenance.json").read_text(encoding="utf-8"))
        label=provenance.get("data_kind","unknown")
        st.write("Data provenance:",label)
        if label!="real":st.warning("Synthetic fixture: software-testing only, not scientific evidence")
        z=np.load(path/"eo_stack.npz");labels=z["labels"]
        # Colorized real raster without inventing geographic statistics.
        rgb=np.zeros((*labels.shape,3),dtype=np.uint8)
        for cls,c in COLORS.items():rgb[labels==cls]=c
        st.image(rgb,caption="ESA WorldCover reference label raster" if label=="real" else "Synthetic label fixture",use_container_width=True)
        if label=="real":
            try:
                from urbantwin.data.spatial_features import summarize
                stats=summarize(z["bands"],labels,float(provenance["configuration"]["resolution_m"]))
                st.json(stats)
            except Exception as exc:st.error(str(exc))
            st.subheader("Georeferenced land-cover map")
            try:
                from rasterio.transform import Affine
                T=Affine(*provenance["transform"])
                to_ll=Transformer.from_crs(provenance["crs"],"EPSG:4326",always_xy=True)
                skip=max(1,int(max(labels.shape)/70))
                samples=[]
                for r in range(0,labels.shape[0],skip):
                    for c in range(0,labels.shape[1],skip):
                        cls=int(labels[r,c])
                        if cls not in COLORS:continue
                        x,y=T*(c+.5,r+.5);lon,lat=to_ll.transform(x,y)
                        samples.append({"lon":lon,"lat":lat,"color":COLORS[cls],"class_id":str(cls)})
                roadsfile=ROOT/"data/osm/roads.geojson"
                layers=[pdk.Layer("ScatterplotLayer",samples,get_position="[lon, lat]",get_fill_color="color",get_radius=9,pickable=True)]
                if roadsfile.exists():
                    # Actual OSM lines; no claim that the simulation uses them.
                    layers.append(pdk.Layer("GeoJsonLayer",json.loads(roadsfile.read_text(encoding="utf-8")),
                                            get_line_color=[40,40,40,220],get_line_width=2,line_width_min_pixels=1))
                st.pydeck_chart(pdk.Deck(map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
                                         initial_view_state=pdk.ViewState(latitude=np.mean([s["lat"] for s in samples]),
                                                                           longitude=np.mean([s["lon"] for s in samples]),zoom=13),
                                         layers=layers,tooltip={"text":"WorldCover class code: {class_id}"}))
                st.caption("© OpenStreetMap contributors where OSM roads/basemap are shown. Categorical WorldCover colors are an approximate visualization of source raster classes.")
            except Exception as exc:st.warning(f"Map rendering issue: {exc}")
    else:st.warning("No imagery has been fetched. Run `urbantwin eo-fetch` (real) or `urbantwin demo` (synthetic test).")

with col2:
    st.subheader("Measured experiment results")
    geo=ROOT/"results/geoai/geoai_metrics.csv"
    if geo.exists():
        df=pd.read_csv(geo)
        st.dataframe(df,use_container_width=True)
        st.plotly_chart(px.bar(df,x="method",y="macro_f1",color="features",barmode="group",
                               title="Spatial holdout macro F1 vs WorldCover reference labels"),use_container_width=True)
    else:st.caption("No real GeoAI benchmark yet. Run `urbantwin geo-train`.")
    traffic=ROOT/"results/traffic/traffic_metrics.csv"
    if traffic.exists():
        df=pd.read_csv(traffic)
        st.dataframe(df,use_container_width=True)
        st.plotly_chart(px.box(df,x="scenario",y="mean_time_loss_s_completed",color="method",
                               title="SUMO completed-trip time loss (held-out seeds)"),use_container_width=True)
        st.plotly_chart(px.bar(df.groupby(["scenario","method"],as_index=False)["unfinished_vehicles"].mean(),
                               x="scenario",y="unfinished_vehicles",color="method",barmode="group",
                               title="Vehicles unfinished at simulation horizon"),use_container_width=True)
    else:st.caption("No SUMO benchmark yet. Run `urbantwin traffic-run` after installing SUMO.")
    road=ROOT/"results/roads.json"
    if road.exists():st.json(json.loads(road.read_text(encoding="utf-8")))

st.divider()
st.caption("Research prototype only. Model agreement with WorldCover is not independent land-cover accuracy; SUMO outcomes are synthetic unless a compatible calibrated network and demand are supplied.")
