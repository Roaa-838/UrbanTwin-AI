"""Create a short report. If no real results are present, say so prominently."""
from __future__ import annotations
from pathlib import Path
import json
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak
from reportlab.lib.units import inch

ROOT=Path(__file__).resolve().parents[3]

def make_report(output: str|Path, include_draft: bool=False) -> str:
    out=Path(output);out.parent.mkdir(parents=True,exist_ok=True)
    real_geo=ROOT/"results/geoai/geoai_metrics.csv"
    real_prov=ROOT/"data/real/provenance.json"
    traffic=ROOT/"results/traffic/traffic_summary.csv"
    if (not real_geo.exists() or not real_prov.exists()) and not include_draft:
        raise RuntimeError("No real GeoAI output. Use --draft to create a methodology-only report, or run real experiments first")
    sheet=getSampleStyleSheet();sheet.add(ParagraphStyle(name="MainTitle",parent=sheet["Title"],fontSize=18,leading=23,alignment=TA_CENTER,spaceAfter=18))
    story=[Paragraph("UrbanTwin-AI",sheet["MainTitle"]),
           Paragraph("Geospatial Intelligence and Traffic Optimization for Sustainable Cities",sheet["Heading2"]),
           Paragraph("Independent research prototype | Cairo area of interest",sheet["Normal"]),Spacer(1,14),
           Paragraph("Research question",sheet["Heading2"]),
           Paragraph("Can spatial indicators derived from Earth observation be analyzed alongside controlled traffic-signal simulations to support reproducible urban research?",sheet["BodyText"]),Spacer(1,8),
           Paragraph("Scope and integrity",sheet["Heading2"]),
           Paragraph("The satellite and ESA WorldCover measurements refer to a small Cairo study area. "
                     "The repository's default SUMO intersection is synthetic, not a reconstruction of Cairo's "
                     "street geometry, measured traffic flows, or emissions. Spatial and simulation findings are presented separately.",sheet["BodyText"]),
           Spacer(1,8),Paragraph("Methods",sheet["Heading2"]),
           Paragraph("Sentinel-2 L2A imagery is SCL-screened and aligned in a projected CRS, then compared with "
                     "ESA WorldCover 2021 reference labels. Random Forest and histogram gradient boosting use "
                     "blocked spatial holdouts. GA and PSO tune two SUMO signal green durations using training seeds; "
                     "held-out seeds evaluate performance against a fixed-time baseline.",sheet["BodyText"]),Spacer(1,8)]
    if real_geo.exists() and real_prov.exists():
        prov=json.loads(real_prov.read_text(encoding="utf-8"))
        if prov.get("data_kind")=="real":
            data=pd.read_csv(real_geo)
            story.append(Paragraph("Observed GeoAI results",sheet["Heading2"]))
            cells=[["Model","Ablation","Macro F1", "Test pixels"]]
            for _,r in data.iterrows():cells.append([str(r["method"]),str(r["features"]),f'{r["macro_f1"]:.3f}',str(r["test_n"])])
            t=Table(cells,repeatRows=1,hAlign="LEFT",colWidths=[136,140,75,75])
            t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#e5e7eb")),("GRID",(0,0),(-1,-1),.35,colors.grey),("FONTSIZE",(0,0),(-1,-1),8),("VALIGN",(0,0),(-1,-1),"MIDDLE")]))
            story.extend([t,Spacer(1,8),Paragraph("These metrics assess agreement with WorldCover pseudo-labels, not independently verified classification accuracy.",sheet["BodyText"])])
    else:
        story.extend([Paragraph("Experimental status",sheet["Heading2"]),
                      Paragraph("METHODOLOGY-ONLY DRAFT: no real satellite-model results have been generated in this repository. Do not cite this draft as an experimental finding.",sheet["BodyText"])])
    story.append(Spacer(1,8))
    if traffic.exists():
        data=pd.read_csv(traffic)
        story.append(Paragraph("Observed SUMO simulation results",sheet["Heading2"]))
        cells=[["Scenario","Method","Time loss (s)","Unfinished"]]
        for _,r in data.iterrows():cells.append([str(r["scenario"]),str(r["method"]),f'{r["mean_time_loss_s"]:.2f}',f'{r["mean_unfinished"]:.1f}'])
        t=Table(cells,repeatRows=1,hAlign="LEFT",colWidths=[110,95,105,92]);t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#e5e7eb")),("GRID",(0,0),(-1,-1),.35,colors.grey),("FONTSIZE",(0,0),(-1,-1),8)]))
        story.extend([t,Spacer(1,8),Paragraph("All results above are simulated; time-loss averages include only completed trips. Unfinished counts are shown separately.",sheet["BodyText"])])
    else:story.extend([Paragraph("SUMO status",sheet["Heading2"]),Paragraph("No SUMO benchmark results recorded yet.",sheet["BodyText"])])
    story.extend([Spacer(1,8),Paragraph("Limitations and future work",sheet["Heading2"]),
                  Paragraph("Single AOI, scene-cloud filtering, WorldCover pseudo-labels, spatial correlation between adjacent blocks, limited simulation seeds, and abstract traffic demand constrain generalization. Future work can integrate a validated real-road network and observed travel-demand counts.",sheet["BodyText"]),Spacer(1,8),
                  Paragraph("Data attribution",sheet["Heading2"]),
                  Paragraph("Contains Copernicus Sentinel data; ESA WorldCover © ESA; road data © OpenStreetMap contributors (ODbL) when used. See repository documentation for data sources and licensing.",sheet["BodyText"])])
    pdf=SimpleDocTemplate(str(out),pagesize=(595,842),rightMargin=48,leftMargin=48,topMargin=48,bottomMargin=48)
    pdf.build(story)
    return str(out)
