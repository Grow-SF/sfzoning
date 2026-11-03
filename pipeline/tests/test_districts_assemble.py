import json
import shutil
import subprocess

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box as _box

from sfzmaps import inputs, config
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


def _overrides(text, series="1960"):
    p = inputs.overrides_path(series)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


def test_override_applied_before_join(data, monkeypatch):
    _overrides("blklot,block,code\n1000001,,C-2\n,1001,R-1\n")
    monkeypatch.setattr(assemble, "read_raw", lambda s, **k: _raw(["R-2", "R-2", "R-4", "R-4", "M-1"]))
    out = assemble.run_series("1960")
    g = gpd.read_file(out / "parcels.fgb").set_index("blklot")
    assert g.loc["1000001", "code"] == "C-2" and g.loc["1000001", "category"] == "commercial_light"
    assert g.loc["1000001", "confidence"] == 1.0
    assert (g.loc[["1001001", "1001002"], "category"] == "single_family").all()
    assert g.loc["1000002", "category"] == "two_unit"


def test_listed_park_lots_are_public_and_keep_their_code(data, monkeypatch):
    # 1000001 is read as housing, 1002001 isn't on the map at all; 1000002 isn't a listed park
    inputs.parks_path().parent.mkdir(parents=True, exist_ok=True)
    inputs.parks_path().write_text("name,blklot,since,source\nA park,1000001,1900,\nA park,1002001,1900,\n")
    monkeypatch.setattr(assemble, "read_raw", lambda s, **k: _raw(["R-1", "R-1", "R-4", "R-4"]))
    g = gpd.read_file(assemble.run_series("1960") / "parcels.fgb").set_index("blklot")
    assert g.loc["1000001", ["code", "category"]].tolist() == ["R-1", "public"]
    assert g.loc["1000001", "note"] == "Park since 1900; this map zones it R-1"
    assert g.loc["1002001", "category"] == "public"
    assert g.loc["1000002", "category"] == "single_family" and pd.isna(g.loc["1000002", "note"])


def test_a_park_is_public_from_its_year_on(data, monkeypatch):
    # since=1950: the 1948 map keeps its zoning, the 1960 map shows the park
    inputs.parks_path().parent.mkdir(parents=True, exist_ok=True)
    inputs.parks_path().write_text("name,blklot,since,source\nNew park,1000001,1950,\nOld park,1000002,1900,\n")
    monkeypatch.setattr(assemble, "read_raw", lambda s, **k: _raw(["R-1", "R-1"]))
    for series, category, count in (("1948", "single_family", 1), ("1960", "public", 2)):
        d = assemble.run_series(series)
        assert gpd.read_file(d / "parcels.fgb").set_index("blklot").loc["1000001", "category"] == category
        # 100x100 ft lots: 10,000 sq ft each
        assert json.loads((d / "stats.json").read_text())["parks"] == {"count": count, "acres": round(count * 1e4 / 43560, 1)}


def test_public_land_is_public_before_it_is_counted_as_a_park(data, monkeypatch):
    # Presidio-like: public land from 1776, a park from 1994
    inputs.parks_path().parent.mkdir(parents=True, exist_ok=True)
    inputs.parks_path().write_text("name,blklot,since,source,public_since\nPresidio,1000001,1994,,1776\n")
    monkeypatch.setattr(assemble, "read_raw", lambda s, **k: _raw(["R-1", "R-1"]))
    for series, count, note in (("1960", 0, "Public land; this map zones it R-1"),
                                ("2000-digital-zoning", 1, "Park since 1994; this map zones it R-1")):
        d = assemble.run_series(series)
        g = gpd.read_file(d / "parcels.fgb").set_index("blklot")
        assert g.loc["1000001", "category"] == "public" and g.loc["1000001", "note"] == note
        assert g.loc["1000002", "category"] == "single_family"
        assert json.loads((d / "stats.json").read_text())["parks"]["count"] == count


def _first_maps_from(monkeypatch, *reads):
    """Publish each (series, codes) read in turn, then write inputs/park_maps.csv from them for
    the park lots 1000001 and 1000002 (no dates in parks.csv)."""
    from sfzmaps import parks

    for series, codes in reads:
        monkeypatch.setattr(assemble, "read_raw", lambda s, codes=codes, **k: _raw(codes))
        assemble.run_series(series)
    parks.write_first_maps(pd.DataFrame({"name": ["A park", "B park"], "blklot": ["1000001", "1000002"]}))


def test_a_park_a_map_shows_is_public_on_every_later_map(data, monkeypatch):
    # P is a public code from 1960 on (the research table has no public code before it)
    _first_maps_from(monkeypatch, ("1960", ["P", "R-1"]), ("1964", ["R-1", "R-1"]))
    g = gpd.read_file(assemble.run_series("1964") / "parcels.fgb").set_index("blklot")
    assert g.loc["1000001", ["code", "category"]].tolist() == ["R-1", "public"]
    assert g.loc["1000001", "note"] == "Park since the 1960 map; this map zones it R-1"
    assert json.loads((assemble.out_dir("1964") / "stats.json").read_text())["parks"]["count"] == 1


def test_a_park_no_map_shows_keeps_its_maps_zoning(data, monkeypatch):
    _first_maps_from(monkeypatch, ("1960", ["P", "R-1"]), ("1964", ["R-1", "R-1"]))
    g = gpd.read_file(assemble.run_series("1964") / "parcels.fgb").set_index("blklot")
    assert g.loc["1000002", "category"] == "single_family" and pd.isna(g.loc["1000002", "note"])
    assert inputs.park_maps_path().read_text().splitlines()[1:] == ["A park,1000001,1960"]


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
    # a height rerun after the zoning run republishes the zoning output (forced: the stub's
    # new answer comes from no input the fingerprint sees)
    raws["2000-digital-height"] = _raw([None], height=["80-B"])
    assemble.run_series("2000-digital-height", force=True)
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
    assert g.loc["1002001", "category"] == "homes_low"
    assert (out / "stats.json").exists()


def _has_pmtiles_driver() -> bool:
    if not shutil.which("ogr2ogr"):
        return False
    r = subprocess.run(["ogr2ogr", "--formats"], capture_output=True, text=True)
    return "PMTiles" in r.stdout


@pytest.mark.skipif(not _has_pmtiles_driver(), reason="ogr2ogr lacks the PMTiles driver")
def test_review_tiles_on_demand_only(data, monkeypatch):
    monkeypatch.setattr(assemble, "read_raw", lambda s, **k: _raw(["R-1", "R-2", "R-4", "C-2", "M-1"]))
    out = assemble.run_series("1960")
    p = out / "1960-districts.pmtiles"
    assert not p.exists()                     # publishing doesn't make them
    assert assemble.review_tiles("1960") == p
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


def test_unzoned_harbor_blocks_by_era(monkeypatch):
    monkeypatch.setattr(assemble, "UNZONED_UNTIL", {"1002": 1969})
    raw = _raw(["R-1", "R-1", "R-4", "R-4", "2R"])
    early = assemble.mask_unzoned(raw, 1948).set_index("blklot")
    assert early.loc["1002001", "code"] is None and early.loc["1002001", "confidence"] == 0.0
    assert early.loc["1000001", "code"] == "R-1"
    late = assemble.mask_unzoned(raw, 1972).set_index("blklot")
    assert late.loc["1002001", "code"] == "2R"


def test_default_unzoned_blocks_are_the_port_piers():
    assert assemble.UNZONED_UNTIL == {"9900": 1969}


# --- provenance, partial reads, status --------------------------------------------------

def test_provenance_written_and_copied_into_stats(data, monkeypatch):
    from sfzmaps.districts import regionread

    monkeypatch.setattr(regionread, "ENABLED", True)
    monkeypatch.setattr(assemble, "git_rev", lambda: "abc1234-dirty")
    raws = {"2000-digital-zoning": _raw(["RH-1", "RH-2", "RM-1", "NC-1", "M-1"]),
            "2000-digital-height": _raw([None, None], height=["40-X", "65-A"])}
    monkeypatch.setattr(assemble, "read_raw", lambda s, **k: raws[s].copy())
    assemble.run_series("2000-digital-height")
    hp = json.loads((config.OUT_DIR / "2000-digital-zoning" / "districts" / "height.json").read_text())
    assert hp["reader"] == "B" and hp["region_read"] is True and hp["git"] == "abc1234-dirty"
    out = assemble.run_series("2000-digital-zoning")
    up = json.loads((config.WORK_DIR / "2000-digital-zoning" / "districts_raw.json").read_text())
    assert set(up) >= {"reader", "region_read", "git", "read_at"}
    st = json.loads((out / "stats.json").read_text())
    assert st["provenance"] == {"use": up, "height": hp}
    rows = {r["series"]: r for r in assemble.status_rows()}
    assert rows["2000-digital-zoning"]["needs_rerun"] is False
    assert rows["1960"]["needs_rerun"] is True and rows["1960"]["use"] is None
    assert "2000-digital-zoning" in assemble.format_status(list(rows.values()))


def test_needs_rerun_rules():
    assert assemble.needs_rerun(None, "B")
    assert assemble.needs_rerun({"region_read": False}, "B")
    assert not assemble.needs_rerun({"region_read": True}, "B")
    assert assemble.needs_rerun({"region_read": True, "cache_only": True, "region_misses": 3}, "B")
    assert not assemble.needs_rerun({"region_read": False}, "A")


def test_cache_only_run_records_misses(data, monkeypatch):
    from sfzmaps import llm
    from sfzmaps.districts import regionread

    def read(s, **k):
        assert llm.CACHE_ONLY
        regionread.Stats.add(4, 3)
        return _raw(["R-1"])

    monkeypatch.setattr(assemble, "read_raw", read)
    assemble.run_series("1960", cache_only=True)
    assert not llm.CACHE_ONLY
    p = json.loads((config.WORK_DIR / "1960" / "districts_raw.json").read_text())
    assert p["cache_only"] is True and p["region_misses"] == 3
    assert assemble.needs_rerun(p, "B")


def test_sheet_run_is_partial_and_leaves_full_outputs(data, monkeypatch):
    monkeypatch.setattr(assemble, "read_raw", lambda s, **k: _raw(["R-1", "R-1", "R-4", "R-4", "M-1"]))
    full = assemble.run_series("1960")
    before = (full / "parcels.fgb").read_bytes(), (config.WORK_DIR / "1960" / "districts_raw.parquet").read_bytes()
    monkeypatch.setattr(assemble, "read_raw", lambda s, **k: _raw(["C-2"]))
    part = assemble.run_series("1960", sheets=["7"])
    assert part == config.OUT_DIR / "1960" / "districts-partial"
    assert ((full / "parcels.fgb").read_bytes(),
            (config.WORK_DIR / "1960" / "districts_raw.parquet").read_bytes()) == before
    st = json.loads((part / "stats.json").read_text())
    assert st["partial"] is True and st["provenance"]["use"]["sheets"] == ["7"]
    assert (config.WORK_DIR / "1960" / "districts_raw.partial.parquet").exists()
    # a partial height read does not republish the zoning series either
    monkeypatch.setattr(assemble, "read_raw", lambda s, **k: _raw([None], height=["40-X"]))
    zfull = assemble.out_dir("1980-1990-zoning")
    hz = assemble.run_series("1980-1990-height", sheets=["1H"])
    assert hz == config.OUT_DIR / "1980-1990-zoning" / "districts-partial"
    assert not (zfull / "height.parquet").exists() and not (zfull / "parcels.fgb").exists()


def test_unmapped_parcels_counted(data, monkeypatch):
    monkeypatch.setattr(assemble, "read_raw", lambda s, **k: _raw(["R-1", "R-1", "R-4"]))
    out = assemble.run_series("1960")
    assert json.loads((out / "stats.json").read_text())["unmapped_parcels"] == 2


def test_today_stats_note_rh1(data):
    zoning = gpd.GeoDataFrame({"zoning": ["RH-1"], "zoning_sim": ["RH-1"]},
                              geometry=[box(-10, -10, 500, 310)], crs="EPSG:2227")
    config.REF_DIR.mkdir()
    zoning.to_parquet(config.REF_DIR / "zoning_current.parquet")
    st = json.loads((assemble.run_series("today") / "stats.json").read_text())
    assert "RH-1" in st["notes"] and "fourplex" in st["notes"] and "by-right" in st["notes"]
    assert st["provenance"]["use"]["reader"] == "datasf"


def test_republish_rereads_nothing(data, monkeypatch):
    monkeypatch.setattr(assemble, "read_raw", lambda s, **k: _raw(["R-1", "R-1", "R-4", "R-4", "M-1"]))
    assemble.run_series("1960")
    monkeypatch.setattr(assemble, "read_raw", lambda *a, **k: 1 / 0)
    _overrides("blklot,code\n1000001,C-2\n")
    g = gpd.read_file(assemble.republish("1960") / "parcels.fgb").set_index("blklot")
    assert g.loc["1000001", "code"] == "C-2"


# --- freshness ----------------------------------------------------------------------------

@pytest.fixture
def counted(data, monkeypatch):
    """A 1960 series with one sheet file, and a read stub that counts its calls."""
    sheet = config.WORK_DIR / "1960" / "1" / "labels.json"
    sheet.parent.mkdir(parents=True)
    sheet.write_text('{"labels": []}')
    calls = []

    def read(s, **k):
        calls.append(s)
        return _raw(["R-1", "R-1", "R-4", "R-4", "M-1"])

    monkeypatch.setattr(assemble, "read_raw", read)
    return sheet, calls


def test_unchanged_inputs_skip_read_and_publish(counted):
    _, calls = counted
    out = assemble.run_series("1960")
    files = [assemble._raw_path("1960"), *(out / n for n in assemble.PUBLISHED)]
    before = [(p.read_bytes(), p.stat().st_mtime_ns) for p in files]
    assert assemble.run_series("1960") == out
    assert assemble.republish("1960") == out
    assert calls == ["1960"]
    assert [(p.read_bytes(), p.stat().st_mtime_ns) for p in files] == before


def test_changed_inputs_rerun_what_they_affect(counted):
    sheet, calls = counted
    out = assemble.run_series("1960")
    # an override changes only the publish: no new read
    _overrides("blklot,code\n1000001,C-2\n")
    g = gpd.read_file(assemble.run_series("1960") / "parcels.fgb").set_index("blklot")
    assert g.loc["1000001", "code"] == "C-2" and calls == ["1960"]
    # a sheet file the read uses changes: read again
    sheet.write_text('{"labels": [1]}')
    assemble.run_series("1960")
    assert calls == ["1960", "1960"]
    # a read that fell back on uncached regions is read again, even with nothing changed
    prov = json.loads(assemble._prov_path("1960").read_text())
    assemble._prov_path("1960").write_text(json.dumps(prov | {"cache_only": True, "region_misses": 2}))
    assemble.run_series("1960")
    assert calls == ["1960"] * 3
    assert (out / "stats.json").exists()


def test_read_code_inputs_are_the_readers_not_publishing_or_sheet_producers():
    names = {p.relative_to(p.parents[1]).as_posix() if p.parent.name == "districts" else p.name
             for p in assemble.read_inputs("1960")}
    assert {"districts/read.py", "districts/labels.py", "districts/pattern.py",
            "districts/regionread.py", "llm.py", "stitch.py", "transform.py"} <= names
    assert not {"districts/assemble.py", "districts/categories.py", "warp.py", "refine.py",
                "runner.py"} & names


# --- overrides by block_num ---------------------------------------------------------------

@pytest.fixture
def suffixed(data, monkeypatch):
    parcels = gpd.GeoDataFrame(
        {"blklot": ["2330006A", "2330007", "0792T021S", "0792T022", "0792001"],
         "block_num": ["2330", "2330", "0792T", "0792T", "0792"], "active": True},
        geometry=[box(0, 0, 100, 100), box(100, 0, 200, 100), box(0, 200, 50, 300),
                  box(50, 200, 100, 300), box(300, 0, 400, 50)], crs="EPSG:2227")
    monkeypatch.setattr(assemble, "load_parcels", lambda: parcels.copy())
    raw = pd.DataFrame({"blklot": list(parcels["blklot"]), "code": ["R-2"] * 5,
                        "height_code": pd.Series([None] * 5, dtype=object), "confidence": 0.9,
                        "source_sheet": "1"})
    return raw


def test_block_overrides_use_block_num(suffixed):
    _overrides("blklot,block,code\n,2330,C-2\n,0792T,M-1\n0792T021S,,R-1\n")
    out = assemble.apply_overrides(suffixed, "1960").set_index("blklot")["code"]
    assert out["2330006A"] == "C-2" and out["2330007"] == "C-2"
    assert out["0792T022"] == "M-1" and out["0792T021S"] == "R-1"   # blklot row wins
    assert out["0792001"] == "R-2"                                 # block 0792 is not 0792T


def test_blocks_of_and_unzoned_by_block_num(suffixed, monkeypatch):
    b = assemble.blocks_of(suffixed["blklot"])
    assert list(b) == ["2330", "2330", "0792T", "0792T", "0792"]
    monkeypatch.setattr(assemble, "UNZONED_UNTIL", {"0792T": 1969})
    out = assemble.mask_unzoned(suffixed, 1948).set_index("blklot")["code"]
    assert out["0792T021S"] is None and out["0792001"] == "R-2"
    # without a parcel table the fallback still reads the lettered block
    monkeypatch.setattr(assemble, "load_parcels", lambda: 1 / 0)
    assert list(assemble.blocks_of(pd.Series(["0792T021S", "2330006A"]))) == ["0792T", "2330"]


def test_height_series_overrides_set_height_code(data, monkeypatch):
    _overrides("blklot,block,height_code\n1000001,,105-E\n,1001,OS\n", "2000-digital-height")
    raws = {"2000-digital-zoning": _raw(["RH-1", "RH-2", "RM-1", "NC-1", "M-1"]),
            "2000-digital-height": _raw([None, None], height=["40-X", "40-X"])}
    monkeypatch.setattr(assemble, "read_raw", lambda s, **k: raws[s].copy())
    assemble.run_series("2000-digital-height")
    g = gpd.read_file(assemble.run_series("2000-digital-zoning") / "parcels.fgb").set_index("blklot")
    assert g.loc["1000001", "height_code"] == "105-E" and g.loc["1000002", "height_code"] == "40-X"
    assert (g.loc[["1001001", "1001002"], "height_code"] == "OS").all()
    assert g.loc["1000001", "code"] == "RH-1" and g.loc["1000001", "confidence"] == 0.9


def test_backfill_provenance_from_mtimes(data, monkeypatch):
    import os

    monkeypatch.setattr(assemble, "_git_at", lambda ts: "ffde7ea")
    for s in ("1960", "1972"):
        p = config.WORK_DIR / s / "districts_raw.parquet"
        p.parent.mkdir(parents=True, exist_ok=True)
        _raw(["R-1"]).to_parquet(p)
        os.utime(p, (1_800_000_000, 1_800_000_000))
    h = assemble.out_dir("2000-digital-zoning") / "height.parquet"
    h.parent.mkdir(parents=True)
    _raw([None], height=["40-X"]).to_parquet(h)
    (config.WORK_DIR / "1972" / "districts_raw.json").write_text('{"keep": true}')
    got = assemble.backfill_provenance({"1960": True, "2000-digital-height": True}, {"1960": "n"})
    assert sorted(p.name for p in got) == ["districts_raw.json", "height.json"]
    p = json.loads((config.WORK_DIR / "1960" / "districts_raw.json").read_text())
    assert p["region_read"] and p["backfilled"] and p["git"] == "ffde7ea" and p["note"] == "n"
    from datetime import datetime
    assert datetime.fromisoformat(p["read_at"]).timestamp() == 1_800_000_000
    assert json.loads((config.WORK_DIR / "1972" / "districts_raw.json").read_text()) == {"keep": True}
    assert json.loads((h.parent / "height.json").read_text())["region_read"] is True


def test_series_read_names_every_answer_it_used_the_same_on_every_run(data, monkeypatch):
    """districts_raw.parquet's sidecar = the series' own reads plus the answers of the legend /
    district-label artifacts it consumed, whether those were read now or reused, so it is
    complete and identical between runs (no dvc.lock churn)."""
    from sfzmaps import answers, llm
    legend = config.WORK_DIR / "1960" / "s1" / "district_legend.json"
    answers.write(legend, ["llm/" + "1" * 64 + ".json"])
    own = "llm/tiles/" + "2" * 64 + ".json"

    def read_raw(s, **k):
        answers.consume(legend)          # a fresh legend, reused
        llm._record(own)                 # the series' own pattern read
        return _raw(["R-2", "R-2", "R-4", "R-4", "M-1"])

    monkeypatch.setattr(assemble, "read_raw", read_raw)
    assemble.run_series("1960")
    first = answers.read(assemble._raw_path("1960"))
    assemble.run_series("1960")
    assert first == answers.read(assemble._raw_path("1960")) == sorted(["llm/" + "1" * 64 + ".json", own])
