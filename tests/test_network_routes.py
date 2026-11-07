from pathlib import Path
from xml.etree import ElementTree as ET
import pytest
from urbantwin.traffic.network import create_routes


def test_route_generation_is_seed_deterministic(tmp_path):
    a=create_routes(tmp_path/'a.xml',200,300,55)
    b=create_routes(tmp_path/'b.xml',200,300,55)
    assert a.read_bytes()==b.read_bytes()
    root=ET.parse(a).getroot()
    assert len(root.findall('route'))==4
    assert len(root.findall('vehicle'))>0


def test_reject_zero_demand(tmp_path):
    with pytest.raises(ValueError):
        create_routes(tmp_path/'a.xml',0,100,42)
