import gzip
import json
import shutil
import subprocess

import geopandas as gpd
import numpy as np
import pyogrio
import pytest
from shapely.geometry import MultiPolygon, box as _box

from sfzmaps import config
from sfzmaps.districts import shared

X0, Y0 = 6006000, 2110000  # San Francisco, EPSG:2227


def box(a, b, c, d):
    return _box(a + X0, b + Y0, c + X0, d + Y0)


GEOM = {"1000001": box(0, 0, 100, 100), "1000002": box(100, 0, 200, 100), "1001001": box(0, 200, 50, 300)}


def write_series(series, lots, geom=GEOM, partial=False, districts=None):
    d = config.OUT_DIR / series / ("districts-partial" if partial else "districts")
    d.mkdir(parents=True)
    g = gpd.GeoDataFrame(
        {"blklot": list(lots), "code": [lots[b][0] for b in lots], "height_code": [lots[b][1] for b in lots],
         "category": [lots[b][2] for b in lots], "confidence": [0.9] * len(lots), "source_sheet": ["1"] * len(lots)},
        geometry=[geom[b] for b in lots], crs="EPSG:2227")
    g.to_file(d / "parcels.fgb", driver="FlatGeobuf")
    lines = districts if districts is not None else g.dissolve("category").reset_index()[["category", "geometry"]]
    lines.to_file(d / "districts.fgb", driver="FlatGeobuf")


def polygons(*shapes):
    return gpd.GeoDataFrame({"category": ["x"] * len(shapes)}, geometry=list(shapes), crs="EPSG:2227")


@pytest.fixture
def out(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OUT_DIR", tmp_path / "out")
    write_series("1948", {"1000001": ("R-1", None, "single_family"), "1000002": ("C-2", None, "commercial_light")})
    write_series("1990-1996-zoning", {"1000001": ("RH-2", "40-X", "two_unit"), "1001001": ("M-1", None, "industrial")})
    return config.OUT_DIR


def test_merge_holds_each_lot_once_with_every_eras_values(out):
    by = shared.merge(["1948", "1990-1996-zoning"]).set_index("blklot")
    assert len(by) == 3
    assert by.loc["1000001", "c_1948"] == "single_family"
    assert by.loc["1000001", "c_1990_1996_zoning"] == "two_unit"
    assert by.loc["1000001", "h_1990_1996_zoning"] == "40-X"


def test_a_lot_an_incomplete_map_leaves_out_has_no_colour_that_year(out):
    series = ["1948", "1990-1996-zoning"]
    parcels = shared.merge(series)
    lines, sets = shared.district_edges(series)
    table = np.frombuffer(shared.colour_table(parcels, series, sets), dtype=np.uint8)
    colours = table[:2 * len(parcels)].reshape(2, len(parcels))
    number = {b: i for i, b in enumerate(parcels["blklot"])}
    assert colours[0, number["1001001"]] == shared.NO_VALUE
    assert colours[1, number["1000002"]] == shared.NO_VALUE
    assert colours[0, number["1000001"]] == shared.CATEGORIES.index("single_family")
    assert colours[1, number["1000001"]] == shared.CATEGORIES.index("two_unit")


def test_smaller_lots_get_lower_numbers_so_they_draw_over_larger_ones(out):
    big = dict(GEOM, **{"9900001": box(0, 0, 500, 500)})
    write_series("1960", {"9900001": ("P", None, "public"), "1000001": ("R-1", None, "single_family")}, geom=big)
    parcels = shared.merge(["1960"])
    assert list(parcels["blklot"]) == ["1000001", "9900001"]


def test_lots_stacked_on_one_footprint_with_the_same_values_draw_once(out):
    condo = dict(GEOM, **{"1000003": GEOM["1000001"]})
    write_series("1960", {"1000001": ("R-1", None, "single_family"), "1000003": ("R-1", None, "single_family"),
                          "1000002": ("C-2", None, "commercial_light")}, geom=condo)
    assert sorted(shared.merge(["1960"])["blklot"]) == ["1000001", "1000002"]


def test_a_stack_that_differs_in_any_era_keeps_every_lot(out):
    condo = dict(GEOM, **{"1000003": GEOM["1000001"]})
    write_series("1960", {"1000001": ("R-1", None, "single_family"), "1000003": ("R-1", None, "single_family")}, geom=condo)
    write_series("1964", {"1000001": ("R-1", None, "single_family"), "1000003": ("R-2", None, "two_unit")}, geom=condo)
    assert sorted(shared.merge(["1960", "1964"])["blklot"]) == ["1000001", "1000003"]


def test_conflicting_geometry_refuses_to_merge(out):
    moved = dict(GEOM, **{"1000001": box(5, 0, 105, 100)})
    write_series("1960", {"1000001": ("R-1", None, "single_family")}, geom=moved)
    with pytest.raises(ValueError, match="1000001"):
        shared.merge(["1948", "1960"])


def test_a_category_without_a_colour_refuses_to_build(out):
    write_series("1960", {"1000001": ("X", None, "brand_new_use")})
    parcels = shared.merge(["1960"])
    with pytest.raises(ValueError, match="brand_new_use"):
        shared.colour_table(parcels, ["1960"], [])


def lines_in(series):
    lines, sets = shared.district_edges(series)
    return {tuple(sets[s]): g for s, g in lines.groupby("s")}


def test_an_edge_two_years_share_is_stored_once_and_drawn_in_both(out):
    # Both years have the left block; only the later one has the right block. Their shared wall
    # at x=100 belongs to both.
    left, right = box(0, 0, 100, 100), box(100, 0, 200, 100)
    write_series("1960", {"1000001": ("R", None, "single_family")}, districts=polygons(left))
    write_series("1964", {"1000001": ("R", None, "single_family")}, districts=polygons(left, right))
    by = lines_in(["1960", "1964"])
    assert set(by) == {(0, 1), (1,)}
    assert by[(0, 1)].length.sum() == pytest.approx(400)
    assert by[(1,)].length.sum() == pytest.approx(300)


def test_an_edge_matches_even_when_neighbours_walk_it_the_other_way(out):
    # Each year has one of the two blocks; their rings run opposite ways along the wall.
    write_series("1960", {"1000001": ("R", None, "single_family")}, districts=polygons(box(0, 0, 100, 100)))
    write_series("1964", {"1000001": ("R", None, "single_family")}, districts=polygons(box(100, 0, 200, 100)))
    by = lines_in(["1960", "1964"])
    assert by[(0, 1)].length.sum() == pytest.approx(100)


def test_every_part_of_a_district_made_of_separate_areas_is_drawn(out):
    apart = MultiPolygon([box(0, 0, 100, 100), box(500, 0, 600, 100)])
    write_series("1960", {"1000001": ("R", None, "single_family")}, districts=polygons(apart))
    assert lines_in(["1960"])[(0,)].length.sum() == pytest.approx(800)


def test_connected_edges_become_one_line(out):
    write_series("1960", {"1000001": ("R", None, "single_family")}, districts=polygons(box(0, 0, 100, 100)))
    lines, _ = shared.district_edges(["1960"])
    assert len(lines) == 1


def test_lot_cards_hold_each_years_values_by_lot_number(out):
    series = ["1948", "1990-1996-zoning"]
    parcels = shared.merge(series)
    cards = shared.lot_cards(parcels, series)
    i = cards["blklot"].index("1000001")
    assert cards["strings"][cards["eras"]["1990-1996-zoning"]["code"][i]] == "RH-2"
    assert cards["strings"][cards["eras"]["1990-1996-zoning"]["height"][i]] == "40-X"
    assert cards["eras"]["1948"]["height"][i] is None
    assert cards["eras"]["1948"]["confidence"][cards["blklot"].index("1001001")] is None
    json.dumps(cards)  # numpy values would break the file


def test_only_full_series_are_included(out):
    write_series("1929", {"1000001": ("R", None, "single_family")}, partial=True)
    assert shared.published_series(["1948", "1929", "1990-1996-zoning", "1935"]) == ["1948", "1990-1996-zoning"]


def test_the_archive_name_follows_the_data(out):
    a = shared.source_hashes(["1948", "1990-1996-zoning"])
    shutil.rmtree(out / "1948")
    write_series("1948", {"1000001": ("R-2", None, "two_unit"), "1000002": ("C-2", None, "commercial_light")})
    assert shared.archive_name(shared.source_hashes(["1948", "1990-1996-zoning"])) != shared.archive_name(a)


def test_stale_reports_a_republished_series(out):
    shared.write_manifest(["1948", "1990-1996-zoning"], "districts-x", 3, 2)
    assert shared.stale_series() == []
    shutil.rmtree(out / "1948")
    write_series("1948", {"1000001": ("R-2", None, "two_unit")})
    assert shared.stale_series() == ["1948"]


def _has_pmtiles_driver() -> bool:
    if not shutil.which("ogr2ogr"):
        return False
    return "PMTiles" in subprocess.run(["ogr2ogr", "--formats"], capture_output=True, text=True).stdout


@pytest.mark.skipif(not _has_pmtiles_driver(), reason="ogr2ogr lacks the PMTiles driver")
def test_build_writes_files_the_pointer_describes(out):
    series = ["1948", "1990-1996-zoning"]
    path = shared.build(series)
    m = json.loads((out / "shared" / "districts.json").read_text())
    assert m["archive"] == f"shared/{path.name}"
    assert m["eras"] == series and m["lots"] == 3
    table = gzip.decompress((out / m["table"]).read_bytes())
    assert len(table) == 2 * (m["lots"] + m["lineSets"])
    assert len(json.loads(gzip.decompress((out / m["cards"]).read_bytes()))["blklot"]) == 3
    assert {row[0] for row in pyogrio.list_layers(path)} == {"parcels", "districts"}
    # Nothing to do on an unchanged rerun: the same archive comes back.
    before = path.stat().st_mtime_ns
    assert shared.build(series).stat().st_mtime_ns == before
