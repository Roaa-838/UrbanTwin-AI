"""Command-line interface for UrbanTwin-AI."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from urbantwin.common import ROOT,load_yaml,save_json

STUDY=ROOT/"configs/study_area.yaml"
EXP=ROOT/"configs/experiments.yaml"


def main(argv=None):
    parser=argparse.ArgumentParser(description="UrbanTwin-AI reproducible research CLI")
    sub=parser.add_subparsers(dest="cmd",required=True)
    demo=sub.add_parser("demo",help="Synthetic offline software smoke test; not research evidence")
    real=sub.add_parser("eo-fetch",help="Fetch real Sentinel-2 and WorldCover for Cairo AOI")
    real.add_argument("--config",default=str(STUDY))
    road=sub.add_parser("roads",help="Download and characterize actual OSM roads")
    road.add_argument("--config",default=str(STUDY))
    geo=sub.add_parser("geo-train",help="Spatial blocked real WorldCover agreement benchmark")
    geo.add_argument("--data-dir",default=str(ROOT/"data/real"))
    geo.add_argument("--config",default=str(EXP))
    traffic=sub.add_parser("traffic-run",help="Run true SUMO simulations on an abstract synthetic junction")
    traffic.add_argument("--config",default=str(EXP))
    traffic.add_argument("--quick",action="store_true",help="Very small GA/PSO budgets for smoke testing only")
    traffic.add_argument("--network",default=None,help="Optional external SUMO .net.xml file")
    traffic.add_argument("--routes-template",default=None,help="Compatible routes path with {scenario} and {seed}")
    traffic.add_argument("--tls-id",default=None)
    report=sub.add_parser("report",help="Generate PDF summarizing recorded experiments")
    report.add_argument("--draft",action="store_true",help="Allow methodology-only PDF if no real outputs exist")
    status=sub.add_parser("status",help="Show which result files are currently available")
    args=parser.parse_args(argv)
    if args.cmd=="demo":
        from urbantwin.data.demo import write_demo
        from urbantwin.geoai.train import run_classification
        write_demo(ROOT/"data/synthetic")
        table=run_classification(ROOT/"data/synthetic",ROOT/"results/demo",EXP,allow_synthetic=True)
        print("SYNTHETIC TEST DATA — do not report these results as real research")
        print(table.to_string(index=False))
    elif args.cmd=="eo-fetch":
        from urbantwin.data.earth_observation import fetch_real
        meta=fetch_real(load_yaml(args.config),ROOT/"data/real")
        print("Downloaded REAL satellite and WorldCover data; STAC item IDs:")
        print(json.dumps({"sentinel":meta["sentinel_stac_item_ids"],"worldcover":meta["worldcover_stac_item_ids"]},indent=2))
    elif args.cmd=="roads":
        from urbantwin.data.spatial_features import road_network_stats
        st,edges=road_network_stats(load_yaml(args.config))
        save_json(st,ROOT/"results/roads.json")
        out=ROOT/"data/osm";out.mkdir(parents=True,exist_ok=True)
        edges[["geometry"]].reset_index(drop=True).to_file(out/"roads.geojson",driver="GeoJSON")
        print(json.dumps(st,indent=2))
    elif args.cmd=="geo-train":
        from urbantwin.geoai.train import run_classification
        from urbantwin.geoai.evaluate import make_figures
        out=ROOT/"results/geoai"
        table=run_classification(args.data_dir,out,args.config)
        paths=make_figures(out)
        print(table.to_string(index=False))
        print("Figures:",*[str(x) for x in paths],sep="\n  ")
    elif args.cmd=="traffic-run":
        from urbantwin.traffic.benchmark import run_benchmark
        from urbantwin.visualization.plots import traffic_figures
        config=args.config
        if args.quick:
            cfg=load_yaml(config)
            cfg["traffic"].update({"optimizer_training_seeds":[17],"evaluation_seeds":[101],
                                   "ga_population":4,"ga_generations":1,"pso_particles":3,"pso_iterations":1,
                                   "simulation_seconds":240,"clearance_seconds":60,
                                   "vehicle_rates_per_approach_per_hour":{"medium":200}})
            config=ROOT/"results/quick_experiments.yaml"
            import yaml
            config.parent.mkdir(parents=True,exist_ok=True)
            config.write_text(yaml.safe_dump(cfg,sort_keys=False),encoding="utf-8")
            print("QUICK TEST — optimizer budget too small for scientific conclusions")
        target=ROOT/("results/traffic_quick" if args.quick else "results/traffic")
        frame=run_benchmark(config,target,args.network,args.routes_template,args.tls_id)
        paths=traffic_figures(target)
        print(frame[["scenario","method","seed","mean_time_loss_s_completed","unfinished_vehicles"]].to_string(index=False))
        print("Figures:",*[str(x) for x in paths],sep="\n  ")
    elif args.cmd=="report":
        from urbantwin.visualization.report_writer import make_report
        output=ROOT/"reports"/("methodology_draft.pdf" if args.draft else "generated_research_report.pdf")
        print(make_report(output,include_draft=args.draft))
    elif args.cmd=="status":
        files={"real EO":ROOT/"data/real/eo_stack.npz", "OSM roads":ROOT/"results/roads.json",
               "real GeoAI results":ROOT/"results/geoai/geoai_metrics.csv",
               "SUMO results":ROOT/"results/traffic/traffic_metrics.csv",
               "synthetic test":ROOT/"results/demo/geoai_metrics.csv"}
        for name,file in files.items():print(f'{"READY" if file.exists() else "NOT RUN"}: {name} ({file.relative_to(ROOT)})')
    return 0


if __name__=="__main__":
    raise SystemExit(main())
