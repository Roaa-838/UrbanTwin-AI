import json

import numpy as np
import pytest

from urbantwin.common import require_real_provenance
from urbantwin.data.demo import write_demo
from urbantwin.data.earth_observation import (
    _composite_scenes,
    _reproject_asset,
    compute_indices,
    fetch_real,
    grid_for_aoi,
)
from urbantwin.data.spatial_features import landcover_percentages, summarize
from urbantwin.geoai.train import construct_samples, split_groups


def test_index_math():
    a=np.ones((5,3,3),dtype="float32")
    a[3]=.8;a[2]=.2;a[4]=.4
    ndvi,ndbi=compute_indices(a)
    assert np.allclose(ndvi,.6)
    assert np.allclose(ndbi,-1/3)


def test_index_zero_denominator_and_nonfinite_are_nan():
    bands=np.ones((5,2,2),dtype="float32")
    bands[2]=np.array([[-1,0],[1,np.nan]],dtype="float32")
    bands[3]=np.array([[1,0],[3,1]],dtype="float32")
    bands[4]=np.array([[-1,0],[1,2]],dtype="float32")
    before=bands.copy()
    ndvi,ndbi=compute_indices(bands)
    assert np.isnan(ndvi[0,0]) and np.isnan(ndvi[0,1]) and np.isnan(ndvi[1,1])
    assert ndvi[1,0]==pytest.approx(.5)
    assert np.isnan(ndbi[0,0]) and np.isnan(ndbi[0,1])
    assert np.array_equal(bands,before,equal_nan=True)


def test_grid_projected():
    from pyproj import Transformer

    g=grid_for_aoi({"bbox":[31.225,30.038,31.247,30.053],"crs":"EPSG:32636","resolution_m":10})
    assert 50<g[1]<400 and 50<g[2]<400
    transform,height,width=g
    assert transform.a==10 and transform.e==-10
    assert width*height<=400_000
    projected=Transformer.from_crs("EPSG:4326","EPSG:32636",always_xy=True).transform_bounds(
        31.225,30.038,31.247,30.053,densify_pts=21)
    grid_bounds=(transform.c,transform.f+height*transform.e,
                 transform.c+width*transform.a,transform.f)
    assert grid_bounds[0]<=projected[0] and grid_bounds[1]<=projected[1]
    assert grid_bounds[2]>=projected[2] and grid_bounds[3]>=projected[3]


def test_grid_rejects_geographic_crs():
    with pytest.raises(ValueError,match="projected"):
        grid_for_aoi({"bbox":[31.225,30.038,31.247,30.053],"crs":"EPSG:4326","resolution_m":10})


def test_landcover_distribution():
    x=np.array([[10,10],[50,0]],dtype=np.uint8)
    shares=landcover_percentages(x)
    assert shares["Tree cover"]==pytest.approx(200/3, abs=.001)
    assert shares["Built-up"]==pytest.approx(100/3, abs=.001)


def test_landcover_distribution_excludes_nodata_unknown_and_masked():
    x=np.array([[10,10],[50,255]],dtype=np.uint8)
    mask=np.array([[True,False],[True,True]])
    shares=landcover_percentages(x,mask)
    assert shares=={"Tree cover":50.0,"Built-up":50.0}


def test_spatial_summary_uses_only_supported_aligned_overlap():
    bands=np.ones((5,2,2),dtype="float32")
    bands[2]=.2
    bands[3]=.8
    bands[4]=.4
    labels=np.array([[10,50],[0,255]],dtype="uint8")

    result=summarize(bands,labels,resolution_m=10)

    assert result["valid_overlap_pixels"]==2
    assert result["represented_area_km2"]==pytest.approx(0.0002)
    assert result["landcover_percentages"]=={"Tree cover":50.0,"Built-up":50.0}
    assert result["ndvi"]["mean"]==pytest.approx(.6)


def test_scene_mask_is_shared_by_all_bands():
    scenes=np.stack([
        np.ones((5,2,2),dtype="float32"),
        np.full((5,2,2),3,dtype="float32"),
    ])
    masks=np.ones((2,2,2),dtype=bool)
    masks[1,0,0]=False
    scenes[0,4,1,1]=np.nan
    composite,valid,count=_composite_scenes(scenes,masks)
    assert count.tolist()==[[1,2],[2,1]]
    assert composite[0,0,0]==1
    assert composite[0,0,1]==2
    assert composite[0,1,1]==3
    assert valid.all()


def test_aligned_categorical_raster_preserves_cells(tmp_path):
    rasterio=pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    source=tmp_path/"classes.tif"
    transform=from_origin(500000,3320000,10,10)
    expected=np.array([[10,50],[80,60]],dtype=np.uint8)
    with rasterio.open(source,"w",driver="GTiff",height=2,width=2,count=1,
                       dtype="uint8",crs="EPSG:32636",transform=transform,nodata=0) as dst:
        dst.write(expected,1)
    target_transform=from_origin(500000,3320000,5,5)
    aligned,_,_=_reproject_asset(str(source),target_transform,4,4,"EPSG:32636",categorical=True)
    assert aligned.shape==(4,4)
    assert np.array_equal(aligned,np.repeat(np.repeat(expected,2,axis=0),2,axis=1))


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


def test_demo_is_byte_reproducible_and_explicitly_synthetic(tmp_path):
    first=write_demo(tmp_path/"a",n=80,seed=7)
    second=write_demo(tmp_path/"b",n=80,seed=7)
    assert first==second
    assert first["data_kind"]=="synthetic"
    assert first["eo_stack_sha256"]==second["eo_stack_sha256"]
    assert first["signed_asset_urls_recorded"] is False
    assert (tmp_path/"a/eo_stack.npz").read_bytes()==(tmp_path/"b/eo_stack.npz").read_bytes()
    assert (tmp_path/"a/provenance.json").read_bytes()==(tmp_path/"b/provenance.json").read_bytes()
    a=np.load(tmp_path/"a/eo_stack.npz")
    b=np.load(tmp_path/"b/eo_stack.npz")
    assert set(a.files)=={"bands","labels","valid_mask","observation_count"}
    for key in a.files:
        assert np.array_equal(a[key],b[key],equal_nan=True)


def test_fetch_real_never_accepts_synthetic_configuration(tmp_path):
    with pytest.raises(ValueError,match="synthetic substitution"):
        fetch_real({"mode":"synthetic"},tmp_path)
