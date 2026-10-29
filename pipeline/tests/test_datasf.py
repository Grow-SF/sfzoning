from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import LineString, Point, Polygon

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
    ("South Park", "SOUTH PARK"), ("ST FRANCIS CIR", "ST FRANCIS"),
    ("Forty Fifth Ave", "45TH"), ("Fortieth Ave", "40TH"), ("Forty-Eighth Ave", "48TH"),
    ("Twenty Second St", "22ND"),
])
def test_normalize_street(raw, want):
    assert datasf.normalize_street(raw) == want


def test_intersections_locates_node_despite_reversed_segment():
    # Node 1 is at (0,0). Segment 3 is digitized backwards (f_node end is far away).
    s = gpd.GeoDataFrame(
        {"street": ["A", "B", "C"], "f_node_cnn": ["1.0", "1.0", "1.0"],
         "t_node_cnn": ["2.0", "3.0", "4.0"], "active": [True] * 3},
        geometry=[LineString([(0, 0), (100, 0)]), LineString([(0, 0), (0, 100)]),
                  LineString([(300, 300), (0, 0)])], crs="EPSG:2227")
    x = datasf.intersections(s)
    n1 = x[x.node_cnn == "1"].geometry.iloc[0]
    assert n1.distance(Point(0, 0)) < 1


def test_intersections_prefers_street_field_unstripped():
    s = gpd.GeoDataFrame(
        {"street": ["SOUTH PARK", "BRYANT"], "streetname": ["SOUTH PARK", "BRYANT ST"],
         "f_node_cnn": ["1.0", "1.0"], "t_node_cnn": ["2.0", "3.0"], "active": [True] * 2},
        geometry=[LineString([(0, 0), (100, 0)]), LineString([(0, 0), (0, 100)])],
        crs="EPSG:2227")
    assert datasf.intersections(s).streets.iloc[0] == ["BRYANT", "SOUTH PARK"]


def _page(n):
    return gpd.GeoDataFrame({"a": range(n)}, geometry=[Point(0, 0)] * n, crs="EPSG:4326")


def test_page_all_stops_on_short_page():
    calls = []

    def fp(offset, limit):
        calls.append(offset)
        return _page(limit if offset < 4 else 1)

    frames = datasf._page_all(fp, page_size=2)
    assert calls == [0, 2, 4] and sum(map(len, frames)) == 5


def test_page_all_stops_on_empty_page_after_full():
    frames = datasf._page_all(lambda o, l: _page(l if o == 0 else 0), page_size=2)
    assert len(frames) == 1


def test_page_all_first_page_empty_raises():
    with pytest.raises(RuntimeError, match="empty"):
        datasf._page_all(lambda o, l: _page(0), page_size=2)


def test_page_all_max_pages_guard():
    with pytest.raises(RuntimeError, match="exceeded"):
        datasf._page_all(lambda o, l: _page(l), page_size=2, max_pages=3)
