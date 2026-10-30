import geopandas as gpd
from shapely.geometry import box

from sfzmaps.districts import today


def _z(rows):
    return gpd.GeoDataFrame(
        {"zoning": [r[0] for r in rows], "zoning_sim": [r[1] for r in rows]},
        geometry=[r[2] for r in rows], crs="EPSG:2227")


def test_majority_overlap():
    parcels = gpd.GeoDataFrame({"blklot": ["a"]}, geometry=[box(0, 0, 100, 100)], crs="EPSG:2227")
    zoning = _z([("RH-1", "RH-1", box(-10, -10, 70, 110)), ("NCT-3", "NCT-3", box(70, -10, 200, 110))])
    out = today.assign_today(parcels, zoning)
    assert out.loc[0, "code"] == "RH-1" and out.loc[0, "confidence"] == 1.0


def test_fallback_to_zoning_sim():
    parcels = gpd.GeoDataFrame({"blklot": ["a"]}, geometry=[box(0, 0, 10, 10)], crs="EPSG:2227")
    out = today.assign_today(parcels, _z([(None, "RM-2", box(-5, -5, 50, 50))]))
    assert out.loc[0, "code"] == "RM-2"


def test_today_categories():
    parcels = gpd.GeoDataFrame({"blklot": ["a", "b", "c"]},
                               geometry=[box(0, 0, 10, 10), box(20, 0, 30, 10), box(40, 0, 50, 10)],
                               crs="EPSG:2227")
    zoning = _z([("RH-1", "RH-1", box(0, 0, 15, 10)), ("RM-2", "RM-2", box(15, 0, 35, 10)),
                 ("C-3-O", "C-3-O", box(35, 0, 60, 10))])
    out = today.assign_today(parcels, zoning).set_index("blklot")
    assert out.loc["a", "category"] == "single_family"
    assert out.loc["b", "category"] == "multi_family"
    assert out.loc["c", "category"] == "commercial"
