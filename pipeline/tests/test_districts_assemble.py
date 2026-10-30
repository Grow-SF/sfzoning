import json
import shutil
import subprocess

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box as _box

from sfzmaps import config
from sfzmaps.districts import assemble

X0, Y0 = 6006000, 2110000  # San Francisco, EPSG:2227


def box(a, b, c, d):
    return _box(a + X0, b + Y0, c + X0, d + Y0)


@pytest.fixture
def data(tmp_path, monkeypatch):
    for name in ("DATA_DIR", "OUT_DIR", "WORK_DIR", "REF_DIR"):
        monkeypatch.setattr(config, name, tmp_path / name.lower())
    config.DATA_DIR.mkdir()
    parcels = gpd.GeoDataFrame(
        {"blklot": ["1000001", "1000002", "1001001", "1001002", "1002001"],
         "block_num": ["1000", "1000", "1001", "1001", "1002"],
         "active": True},
        geometry=[box(0, 0, 100, 100), box(100, 0, 200, 100), box(0, 200, 50, 300),
                  box(50, 200, 100, 300), box(300, 0, 400, 50)], crs="EPSG:2227")
    monkeypatch.setattr(assemble, "load_parcels", lambda: parcels.copy())
    return tmp_path


def _raw(codes, height=None, conf=0.9, sheet="1"):
    n = len(codes)
    return pd.DataFrame({"blklot": ["1000001", "1000002", "1001001", "1001002", "1002001"][:n],
                         "code": pd.Series(codes, dtype=object),
                         "height_code": pd.Series(height or [None] * n, dtype=object),
                         "confidence": [conf] * n, "source_sheet": [sheet] * n})


def _overrides(text):
    d = config.DATA_DIR / "overrides"
    d.mkdir(exist_ok=True)
    (d / "1960.csv").write_text(text)


def test_override_applied_before_join(data, monkeypatch):
    _overrides("blklot,block,code\n1000001,,C-2\n,1001,R-1\n")
    monkeypatch.setattr(assemble, "read_raw", lambda s, **k: _raw(["R-2", "R-2", "R-4", "R-4", "M-1"]))
    out = assemble.run_series("1960")
    g = gpd.read_file(out / "parcels.fgb").set_index("blklot")
    assert g.loc["1000001", "code"] == "C-2" and g.loc["1000001", "category"] == "commercial"
    assert g.loc["1000001", "confidence"] == 1.0
    assert (g.loc[["1001001", "1001002"], "category"] == "single_family").all()
    assert g.loc["1000002", "category"] == "two_unit"


def test_override_missing_blklot_warns(data):
    _overrides("blklot,code\n9999999,C-2\n1000002,R-1\n")
    with pytest.warns(UserWarning, match="9999999"):
        df = assemble.apply_overrides(_raw(["R-2", "R-2"]), "1960")
    assert list(df["code"]) == ["R-2", "R-1"]


def test_no_overrides_is_identity(data):
    df = _raw(["R-2"])
    assert assemble.apply_overrides(df, "1960").equals(df)


def test_height_join_by_blklot(data, monkeypatch):
    raws = {"2000-digital-zoning": _raw(["RH-1", "RH-2", "RM-1", "NC-1", "M-1"]),
            "2000-digital-height": _raw([None, None, None], height=["40-X", "40-X", "65-A"], conf=0.7)}
    monkeypatch.setattr(assemble, "read_raw", lambda s, **k: raws[s].copy())
    hz = assemble.run_series("2000-digital-height")
    assert hz == config.OUT_DIR / "2000-digital-zoning" / "districts"
    assert (hz / "height.parquet").exists()
    assert not (config.OUT_DIR / "2000-digital-height").exists()
    out = assemble.run_series("2000-digital-zoning")
    g = gpd.read_file(out / "parcels.fgb").set_index("blklot")
    assert g.loc["1000001", "height_code"] == "40-X" and g.loc["1001001", "height_code"] == "65-A"
    assert pd.isna(g.loc["1001002", "height_code"])
    assert g.loc["1000001", "code"] == "RH-1" and g.loc["1000001", "confidence"] == 0.9
    # a height rerun after the zoning run republishes the zoning output
    raws["2000-digital-height"] = _raw([None], height=["80-B"])
    assemble.run_series("2000-digital-height")
    g = gpd.read_file(out / "parcels.fgb").set_index("blklot")
    assert g.loc["1000001", "height_code"] == "80-B"


def test_stats_sum(data, monkeypatch):
    monkeypatch.setattr(assemble, "read_raw", lambda s, **k: _raw(["R-1", "R-1", "R-4", None, "ZZ-9"]))
    out = assemble.run_series("1960")
    s = json.loads((out / "stats.json").read_text())
    cats = {k: v for k, v in s.items() if isinstance(v, dict) and "parcels" in v}
    assert set(cats) == set(assemble.CATEGORIES)
    assert sum(v["parcels"] for v in cats.values()) == 5
    assert sum(v["area_sqft"] for v in cats.values()) == pytest.approx(s["mapped_area_sqft"])
    assert s["mapped_area_sqft"] == pytest.approx(10000 * 2 + 5000 * 2 + 5000)
    assert cats["single_family"] == {"parcels": 2, "area_sqft": pytest.approx(20000)}
    assert s["unknown_share"] == pytest.approx(0.4)
    d = gpd.read_file(out / "districts.fgb")
    assert sorted(d["code"]) == ["R-1", "R-4", "ZZ-9"]
    assert set(d.columns) >= {"code", "category"}
    assert json.loads((out / "unknown_codes.json").read_text()) == {"ZZ-9": 1}


def test_needs_review_sheet_caps_confidence():
    df = _raw(["R-1", "R-1"], conf=0.95)
    df.loc[1, "source_sheet"] = "2"
    out = assemble.cap_confidence(df, {"1": "passed", "2": "needs_review"})
    assert list(out["confidence"]) == [0.95, assemble.MEDIUM]


def test_today_published_same_way(data, monkeypatch):
    zoning = gpd.GeoDataFrame({"zoning": ["RH-1", "RM-2"], "zoning_sim": ["RH-1", "RM-2"]},
                              geometry=[box(-10, -10, 150, 310), box(150, -10, 500, 310)],
                              crs="EPSG:2227")
    config.REF_DIR.mkdir()
    zoning.to_parquet(config.REF_DIR / "zoning_current.parquet")
    out = assemble.run_series("today")
    assert out == config.OUT_DIR / "today" / "districts"
    g = gpd.read_file(out / "parcels.fgb").set_index("blklot")
    assert g.loc["1000001", "category"] == "single_family"
    assert g.loc["1002001", "category"] == "multi_family"
    assert (out / "stats.json").exists()


def _has_pmtiles_driver() -> bool:
    if not shutil.which("ogr2ogr"):
        return False
    r = subprocess.run(["ogr2ogr", "--formats"], capture_output=True, text=True)
    return "PMTiles" in r.stdout


@pytest.mark.skipif(not _has_pmtiles_driver(), reason="ogr2ogr lacks the PMTiles driver")
def test_pmtiles_written(data, monkeypatch):
    monkeypatch.setattr(assemble, "read_raw", lambda s, **k: _raw(["R-1", "R-2", "R-4", "C-2", "M-1"]))
    out = assemble.run_series("1960")
    p = out / "1960-districts.pmtiles"
    assert p.exists() and p.stat().st_size > 0
    assert p.read_bytes()[:7] == b"PMTiles"
    if shutil.which("pmtiles"):
        meta = json.loads(subprocess.run(["pmtiles", "show", "--metadata", str(p)],
                                         capture_output=True, text=True).stdout)
        layers = {l["id"]: l for l in meta["vector_layers"]}
        assert set(layers) == {"parcels", "districts"}
        assert set(layers["parcels"]["fields"]) == {"blklot", "code", "height_code", "category", "confidence"}
        assert set(layers["districts"]["fields"]) == {"code", "category"}


def test_parcel_without_geometry_dropped(data, monkeypatch):
    parcels = assemble.load_parcels()
    parcels.loc[4, "geometry"] = None
    monkeypatch.setattr(assemble, "load_parcels", lambda: parcels.copy())
    monkeypatch.setattr(assemble, "read_raw", lambda s, **k: _raw(["R-1", "R-1", "R-4", "R-4", "M-1"]))
    out = assemble.run_series("1960")
    assert len(gpd.read_file(out / "parcels.fgb")) == 4
    assert sorted(gpd.read_file(out / "districts.fgb")["code"]) == ["R-1", "R-4"]


def test_merge_use_height_reads():
    use = pd.DataFrame({"blklot": ["a", "b"], "code": ["R-1", "C-2"], "confidence": [0.9, 0.8],
                        "source_sheet": ["1", "1"]})
    height = pd.DataFrame({"blklot": ["b", "c"], "height_code": ["40-X", "65-A"],
                           "confidence": [0.7, 0.6], "source_sheet": ["1H", "1H"]})
    m = assemble.merge_use_height(use, height, role="use").set_index("blklot")
    assert list(m.index) == ["a", "b", "c"]
    assert m.loc["b", "code"] == "C-2" and m.loc["b", "height_code"] == "40-X"
    assert m.loc["b", "confidence"] == 0.8 and m.loc["b", "source_sheet"] == "1"
    assert m.loc["c", "code"] is None and m.loc["c", "confidence"] == 0.0
    h = assemble.merge_use_height(use, height, role="height").set_index("blklot")
    assert h.loc["c", "confidence"] == 0.6 and h.loc["c", "source_sheet"] == "1H"
    assert h.loc["a", "height_code"] is None and h.loc["a", "confidence"] == 0.0
