import json

import yaml

from urbantwin.data.demo import write_demo
from urbantwin.geoai.evaluate import make_figures
from urbantwin.geoai.train import run_classification


def test_demo_training(tmp_path):
    write_demo(tmp_path/"synthetic")
    cfg={"geoai":{"spatial_block_pixels":20,"test_fraction":.25,
        "spatial_split_seeds":[42,142],
        "max_train_per_class":100,"min_train_per_class":10,"min_test_per_class":5,
        "classifiers":["random_forest"],"ablation":["bands_only","bands_plus_indices"]}}
    file=tmp_path/"config.yaml";file.write_text(yaml.safe_dump(cfg))
    result=run_classification(tmp_path/"synthetic",tmp_path/"results",file,allow_synthetic=True)
    assert len(result)==2
    assert all(result.macro_f1.between(0,1))
    assert set(result.split_count)=={2}
    details=json.loads((tmp_path/"results/geoai_details.json").read_text())
    assert details["class_selection_protocol"]=="training_partition_only"
    assert details["evaluated_split_count"]==2
    assert all("unsupported_test_class_support" in split
               for split in details["split_diagnostics"])
    experiment=details["experiments"]["random_forest__bands_only"]
    assert len(experiment["runs"])==2
    assert all(run["confusion_matrix"] for run in experiment["runs"])
    assert len(details["metrics_sha256"])==64
    assert len(details["split_metrics_sha256"])==64
    assert all(path.is_file() for path in make_figures(tmp_path/"results"))
