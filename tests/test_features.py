import numpy as np
import pytest
from urbantwin.data.earth_observation import compute_indices, grid_for_aoi
from urbantwin.data.spatial_features import landcover_percentages
from urbantwin.geoai.train import construct_samples, split_groups
from urbantwin.data.demo import write_demo
from urbantwin.common import require_real_provenance
import json


def test_index_math():
    a=np.ones((5,3,3),dtype="float32")
    a[3]=.8;a[2]=.2;a[4]=.4
    ndvi,ndbi=compute_indices(a)
    assert np.allclose(ndvi,.6)
    assert np.allclose(ndbi,-1/3)


def test_grid_projected():
    g=grid_for_aoi({"bbox":[31.225,30.038,31.247,30.053],"crs":"EPSG:32636","resolution_m":10})
    assert 50<g[1]<400 and 50<g[2]<400


def test_landcover_distribution():
    x=np.array([[10,10],[50,0]],dtype=np.uint8)
    shares=landcover_percentages(x)
    assert shares["Tree cover"]==pytest.approx(200/3, abs=.001)
    assert shares["Built-up"]==pytest.approx(100/3, abs=.001)


def test_group_disjoint(tmp_path):
    write_demo(tmp_path,n=100)
    data=np.load(tmp_path/"eo_stack.npz")
    X,y,g,_=construct_samples(data["bands"],data["labels"],20)
    a,b=split_groups(X,y,g,0.3,42)
    assert not set(g[a])&set(g[b])
    assert len(a)>0 and len(b)>0


def test_fake_data_rejected(tmp_path):
    write_demo(tmp_path)
    meta=json.loads((tmp_path/"provenance.json").read_text())
    with pytest.raises(ValueError,match="Synthetic"):
        require_real_provenance(meta)
