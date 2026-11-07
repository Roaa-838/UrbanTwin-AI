from urbantwin.data.demo import write_demo
from urbantwin.geoai.train import run_classification
from pathlib import Path
import yaml


def test_demo_training(tmp_path):
    write_demo(tmp_path/"synthetic")
    cfg={"geoai":{"spatial_block_pixels":20,"test_fraction":.25,
        "max_train_per_class":100,"min_train_per_class":10,"min_test_per_class":5,
        "classifiers":["random_forest"],"ablation":["bands_only","bands_plus_indices"]}}
    file=tmp_path/"config.yaml";file.write_text(yaml.safe_dump(cfg))
    result=run_classification(tmp_path/"synthetic",tmp_path/"results",file,allow_synthetic=True)
    assert len(result)==2
    assert all(result.macro_f1.between(0,1))
    assert (tmp_path/"results/geoai_details.json").exists()
