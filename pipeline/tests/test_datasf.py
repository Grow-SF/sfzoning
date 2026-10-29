from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import LineString, Polygon

from sfzmaps.sources import datasf

FIX = Path(__file__).parent / "fixtures" / "parcels_sample.geojson"


@pytest.fixture(scope="module")
def parcels():
    return gpd.read_file(FIX).to_crs("EPSG:2227")


@pytest.fixture(scope="module")
def blocks(parcels):
    return datasf.build_blocks(parcels)


def test_build_blocks_dissolves_by_block(blocks):
    assert len(blocks) == 3
    assert set(blocks.block_num) == {"0045", "0046", "0047"}
    assert blocks.crs.to_epsg() == 2227


def test_block_corners_rectangular_block_has_4(blocks):
    corners = datasf.block_corners(blocks)
    assert len(corners[corners.block_num == "0045"]) == 4


def test_block_num_suffix_dropped_and_inactive_ignored():
    sq = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
    sq2 = Polygon([(100, 0), (200, 0), (200, 100), (100, 100)])
    far = Polygon([(500, 500), (600, 500), (600, 600), (500, 600)])
    g = gpd.GeoDataFrame(
        {"block_num": ["12", "12A", "9"], "active": [True, "true", False]},
        geometry=[sq, sq2, far], crs="EPSG:2227")
    b = datasf.build_blocks(g)
    assert list(b.block_num) == ["0012"]
    assert b.geometry.iloc[0].area == pytest.approx(20000)


def test_intersections():
    s = gpd.GeoDataFrame(
        {"street": ["LOMBARD", "HYDE", "HYDE"],
         "f_node_cnn": ["1.0", "1.0", "2.0"], "t_node_cnn": ["3.0", "2.0", "4.0"],
         "active": [True, True, True]},
        geometry=[LineString([(0, 0), (100, 0)]), LineString([(0, 0), (0, 100)]),
                  LineString([(0, 100), (0, 200)])], crs="EPSG:2227")
    x = datasf.intersections(s)
    assert list(x.node_cnn) == ["1"]
    assert x.streets.iloc[0] == ["HYDE", "LOMBARD"]


@pytest.mark.parametrize("raw,want", [
    ("Lombard St", "LOMBARD"), ("First Street", "1ST"), ("Van Ness Ave", "VAN NESS"),
    ("Twenty-Fifth Ave", "25TH"), ("01ST ST", "1ST"), ("O'Farrell St", "O FARRELL"),
])
def test_normalize_street(raw, want):
    assert datasf.normalize_street(raw) == want
