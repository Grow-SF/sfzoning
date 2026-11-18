from pathlib import Path

import pytest

from sfzmaps import config, downloads
from sfzmaps.downloads.readme import readme


def _touch(p, data=b"x"):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)


@pytest.fixture
def tmp_data(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "WORK_DIR", tmp_path / "work")
    monkeypatch.setattr(config, "OUT_DIR", tmp_path / "out")
    d = tmp_path / "out" / "1948" / "districts"
    _touch(d / "parcels.fgb")
    _touch(d / "districts.fgb")
    _touch(tmp_path / "work" / "1948" / "1" / "warped_paper.tif")
    _touch(tmp_path / "work" / "1948" / "1" / "warped_paper.footprint.geojson")
    return tmp_path


@pytest.fixture
def tmp_data_su(tmp_data):
    _touch(tmp_data / "work" / "1980-1990-su" / "1" / "warped_paper.tif")
    _touch(tmp_data / "out" / "1980-1990-su" / "districts" / "parcels.fgb")
    return tmp_data


def test_catalog_lists_three_formats_per_vector_layer(tmp_data):
    keys = {i.key for i in downloads.catalog()}
    assert {"1948/lots.geojson", "1948/lots.gpkg", "1948/lots.shp",
            "1948/districts.geojson", "1948/districts.gpkg", "1948/districts.shp",
            "1948/citywide.cog"} <= keys
    by = {i.key: i for i in downloads.catalog()}
    assert by["1948/lots.shp"].filename == "1948-lots.shp.zip"
    assert by["1948/citywide.cog"].filename == "1948-citywide.tif"
    assert by["1948/lots.gpkg"].sources == [tmp_data / "out/1948/districts/parcels.fgb"]
    assert by["1948/districts.gpkg"].sources == [tmp_data / "out/1948/districts/districts.fgb"]
    names = {p.name for p in by["1948/citywide.cog"].sources}
    assert names == {"warped_paper.tif", "warped_paper.footprint.geojson", "stitch.py"}


def test_today_gets_vectors_when_built(tmp_data):
    _touch(tmp_data / "out/today/districts/parcels.fgb")
    _touch(tmp_data / "out/today/districts/districts.fgb")
    keys = {i.key for i in downloads.catalog()}
    assert "today/lots.shp" in keys and "today/citywide.cog" not in keys


def test_su_series_gets_citywide_only(tmp_data_su):
    keys = {i.key for i in downloads.catalog() if i.series == "1980-1990-su"}
    assert keys == {"1980-1990-su/citywide.cog"}


def test_fingerprint_changes_with_source_bytes_and_version(tmp_data):
    fpj = tmp_data / "work/1948/1/warped_paper.footprint.geojson"
    cw = {i.key: i for i in downloads.catalog()}["1948/citywide.cog"]
    before = downloads.fingerprint(cw, "1")
    _touch(fpj, b"moved")
    assert downloads.fingerprint(cw, "1") != before
    for key in ("1948/lots.gpkg", "1948/citywide.cog"):
        item = {i.key: i for i in downloads.catalog()}[key]
        fp = downloads.fingerprint(item, "1")
        assert fp == downloads.fingerprint(item, "1")
        assert fp != downloads.fingerprint(item, "2")
        _touch(item.sources[0], b"changed")
        assert fp != downloads.fingerprint(item, "1")


def test_readme_has_attribution_license_and_rumsey_only_for_1948():
    r = " ".join(readme("1948", "citywide", "2026-11-04", "abc1234", "cog").split())
    assert "Based on sfzoning.com by Steven Bacio for GrowSF (https://growsf.org)" in r
    assert "CC BY-NC-SA 4.0" in r and config.RUMSEY_CREDIT in r and "2026-11-04" in r
    assert config.RUMSEY_CREDIT not in readme("1960", "lots", "2026-11-04", "abc1234", "geojson")
    assert "not an official record" in readme("1960", "lots", "2026-11-04", "abc1234", "geojson")
    assert "DRAFT COPY" not in r and "{{" not in r


def test_readme_field_table_only_for_vector_layers():
    r = readme("1960", "lots", "d", "c", "geojson")
    row = next(l for l in r.splitlines() if l.startswith("height_code"))
    assert row.split()[1] == "height" and "height_cd" not in r
    assert "FIELDS" not in readme("1960", "citywide", "d", "c", "cog")


def _flat(text):
    return " ".join(text.split())


def test_readme_crs_text_per_layer():
    lots = _flat(readme("1960", "lots", "d", "c", "geojson"))
    assert "GeoJSON and Shapefile are in WGS 84 (EPSG:4326)" in lots
    assert "`lots_ca_spcs`" in lots and "EPSG:2227" in lots
    assert "`districts_ca_spcs`" in readme("1960", "districts", "d", "c", "geojson")
    city = readme("1960", "citywide", "d", "c", "cog")
    assert "EPSG:2227" in city and "4326" not in city


def test_data_fingerprint_covers_only_the_data_code_and_fingerprint_adds_the_text_code(tmp_data, monkeypatch):
    code = tmp_data / "code"
    names = ("readme.py", "readme-template.txt", "package.py", "vectors.py", "rasters.py")
    for n in names:
        _touch(code / n, b"v1")
    monkeypatch.setattr(downloads, "DATA_CODE", {"citywide": [code / "rasters.py"],
                        "lots": [code / "vectors.py"], "districts": [code / "vectors.py"]})
    monkeypatch.setattr(downloads, "TEXT_CODE", [code / n for n in ("readme.py", "readme-template.txt", "package.py")])
    by = {i.key: i for i in downloads.catalog()}
    lots, city = by["1948/lots.geojson"], by["1948/citywide.cog"]
    for n in ("readme.py", "readme-template.txt", "package.py"):  # text code: fingerprint only
        d, f = downloads.data_fingerprint(lots, "1"), downloads.fingerprint(lots, "1")
        _touch(code / n, b"v2")
        assert downloads.data_fingerprint(lots, "1") == d and downloads.fingerprint(lots, "1") != f
    d, dc = downloads.data_fingerprint(lots, "1"), downloads.data_fingerprint(city, "1")
    _touch(code / "vectors.py", b"v2")  # vector code: vectors only
    assert downloads.data_fingerprint(lots, "1") != d and downloads.data_fingerprint(city, "1") == dc
    f = downloads.fingerprint(city, "1")
    _touch(code / "rasters.py", b"v2")
    assert downloads.data_fingerprint(city, "1") != dc and downloads.fingerprint(city, "1") != f


def test_exporter_code_is_the_real_exporter_files():
    assert {k: [p.name for p in v] for k, v in downloads.DATA_CODE.items()} == {
        "citywide": ["rasters.py"], "lots": ["vectors.py", "fields.py"],
        "districts": ["vectors.py", "fields.py"]}
    assert [p.name for p in downloads.TEXT_CODE] == ["readme.py", "readme-template.txt", "package.py", "fields.py"]
    assert all(p.exists() for v in downloads.DATA_CODE.values() for p in v)
    assert all(p.exists() for p in downloads.TEXT_CODE)


def test_today_readme_credits_datasf_zoning_not_a_scan():
    for layer in ("lots", "districts"):
        r = _flat(readme("today", layer, "d", "c", "geojson"))
        assert "3i4a-hu95" in r and "Zoning Districts" in r
        assert "read from the scans" not in r and "read from the map" not in r
        assert "flagged for review" not in r
    lots = readme("today", "lots", "d", "c", "geojson")
    row = next(l for l in lots.splitlines() if l.startswith("source_sheet"))
    assert "datasf" in row
    lots = _flat(lots)
    assert "largest share" in lots
    assert "Always 1" in lots


def test_historic_confidence_names_overrides_and_hand_added_labels():
    r = _flat(readme("1960", "lots", "d", "c", "geojson"))
    assert "inputs/<series>/overrides.csv" in r and "inputs/<series>/label_fixes.csv" in r
    assert "3i4a-hu95" not in r


def test_scan_readmes_describe_the_warp_honestly():
    for layer in ("citywide", "sheets"):
        r = readme("1960", layer, "d", "c", "cog")
        flat = _flat(r)
        assert "not resampled" not in flat and "no warping" not in flat
        assert "thin plate spline" in flat and "today's street corners" in flat
        assert "tinted to the paper colour" in flat and "binding shadows" in flat
        assert "Nothing is vectorized or drawn" in flat
    assert "finest sheet" in _flat(readme("1960", "citywide", "d", "c", "cog"))


def test_citywide_sources_skip_missing_footprint(tmp_data):
    (tmp_data / "work/1948/1/warped_paper.footprint.geojson").unlink()
    cw = {i.key: i for i in downloads.catalog()}["1948/citywide.cog"]
    downloads.fingerprint(cw, "1")


# --- vector exports -------------------------------------------------------

import shutil
import zipfile

from sfzmaps.downloads import Item
from sfzmaps.downloads import vectors
from sfzmaps.downloads.readme import DISTRICT_FIELDS, LOT_FIELDS

needs_gdal = pytest.mark.skipif(not shutil.which("ogr2ogr"), reason="ogr2ogr not on PATH")


def _make_fgb(path, layer):
    import geopandas as gpd
    from shapely.geometry import box

    # two small boxes near Civic Center, in CA State Plane III feet
    geoms = [box(6007000, 2112000, 6007100, 2112100), box(6007200, 2112000, 6007300, 2112100)]
    if layer == "lots":
        df = gpd.GeoDataFrame({
            "blklot": ["0001001", "0001002"], "code": ["R-1", "C-2"],
            "height_code": ["40-X", "80-X"], "category": ["residential", "commercial"],
            "confidence": [0.9, 0.5], "source_sheet": ["1", "2"], "note": ["", "park"],
        }, geometry=geoms, crs=2227)
    else:
        df = gpd.GeoDataFrame({"code": ["R-1", "C-2"], "category": ["residential", "commercial"]},
                              geometry=geoms, crs=2227)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_file(path, driver="FlatGeobuf")


def _item(tmp_path, layer, fmt):
    src = tmp_path / "src" / ("parcels.fgb" if layer == "lots" else "districts.fgb")
    if not src.exists():
        _make_fgb(src, layer)
    return Item(f"1948/{layer}.{fmt}", "1948", layer, fmt, f"1948-{layer}.{fmt}.zip", [src])


def _export(tmp_path, layer, fmt, updated="2026-11-04"):
    from sfzmaps.downloads import package
    item = _item(tmp_path, layer, fmt)
    out = tmp_path / "out" / item.filename
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = tmp_path / "layers"
    tmp.mkdir(exist_ok=True)
    members = {p.name: p.read_bytes() for p in vectors.write_layers(item, tmp)}
    package.write_zip(out, members, item, updated, "abc1234")
    return out


@needs_gdal
def test_geojson_zip_is_wgs84_with_all_lot_fields(tmp_path):
    import geopandas as gpd
    out = _export(tmp_path, "lots", "geojson")
    with zipfile.ZipFile(out) as z:
        name = next(n for n in z.namelist() if n.endswith(".geojson"))
    gdf = gpd.read_file(f"/vsizip/{out}/{name}")
    assert [c for c in gdf.columns if c != "geometry"] == [f[0] for f in LOT_FIELDS]
    assert gdf.crs.to_epsg() == 4326
    minx, miny, maxx, maxy = gdf.total_bounds
    assert -122.6 < minx and maxx < -122.3 and 37.6 < miny and maxy < 37.9


@needs_gdal
def test_gpkg_has_both_crs_layers(tmp_path):
    import geopandas as gpd
    out = _export(tmp_path, "districts", "gpkg")
    with zipfile.ZipFile(out) as z:
        name = next(n for n in z.namelist() if n.endswith(".gpkg"))
    a = gpd.read_file(f"/vsizip/{out}/{name}", layer="districts")
    b = gpd.read_file(f"/vsizip/{out}/{name}", layer="districts_ca_spcs")
    assert a.crs.to_epsg() == 4326 and b.crs.to_epsg() == 2227
    assert len(a) == len(b) == 2
    assert [c for c in a.columns if c != "geometry"] == [f[0] for f in DISTRICT_FIELDS]


@needs_gdal
def test_shapefile_uses_short_names_and_prj(tmp_path):
    import geopandas as gpd
    out = _export(tmp_path, "lots", "shp")
    with zipfile.ZipFile(out) as z:
        names = z.namelist()
    assert any(n.endswith(".prj") for n in names)
    shp = next(n for n in names if n.endswith(".shp"))
    gdf = gpd.read_file(f"/vsizip/{out}/{shp}")
    assert [c for c in gdf.columns if c != "geometry"] == [f[1] for f in LOT_FIELDS]
    assert gdf.crs.to_epsg() == 4326


@needs_gdal
def test_zip_contains_readme_and_license_with_fixed_dates(tmp_path):
    out = _export(tmp_path, "lots", "geojson")
    with zipfile.ZipFile(out) as z:
        assert {"README.txt", "LICENSE.txt"} <= set(z.namelist())
        assert z.namelist() == sorted(z.namelist())
        assert all(i.date_time == (1980, 1, 1, 0, 0, 0) for i in z.infolist())
        assert all(i.compress_type == zipfile.ZIP_DEFLATED for i in z.infolist())
        assert "Based on sfzoning.com by Steven Bacio" in z.read("README.txt").decode()
    out2 = tmp_path / "again.zip"
    shutil.copy(out, out2)
    from sfzmaps.downloads import package
    package.write_zip(out2, package.data_members(out), _item(tmp_path, "lots", "geojson"), "2026-11-04", "abc1234")
    assert out.read_bytes() == out2.read_bytes()


# --- citywide image and sheet entries ---------------------------------------

import hashlib
import subprocess

needs_gdal_raster = pytest.mark.skipif(not shutil.which("gdal_create"), reason="gdal_create not on PATH")


def _make_sheets(work, series="1948"):
    for sid, x0 in (("2", 6007000), ("1", 6007100)):
        d = work / series / sid
        d.mkdir(parents=True)
        subprocess.run(["gdal_create", "-of", "GTiff", "-outsize", "32", "32", "-bands", "4",
                        "-burn", "200", "-burn", "190", "-burn", "170", "-burn", "255",
                        "-a_srs", "EPSG:2227", "-a_ullr", str(x0), "2112100", str(x0 + 100), "2112000",
                        "-ot", "Byte", str(d / "warped_paper.tif")], check=True, capture_output=True)


def _fake_build_vrt(monkeypatch):
    def build(series, scores=None, paper=False):
        assert paper
        tifs = sorted((config.WORK_DIR / series).glob("*/warped_paper.tif"))
        out = config.OUT_DIR / series / f"{series}-paper.vrt"
        out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["gdalbuildvrt", "-overwrite", str(out), *map(str, tifs)], check=True, capture_output=True)
        return out
    monkeypatch.setattr("sfzmaps.stitch.build_vrt", build)


@needs_gdal_raster
def test_citywide_is_cog_with_license_tags(tmp_data, monkeypatch):
    import json
    from sfzmaps.downloads import rasters
    shutil.rmtree(tmp_data / "work" / "1948")
    _make_sheets(tmp_data / "work")
    _fake_build_vrt(monkeypatch)
    item = {i.key: i for i in downloads.catalog()}["1948/citywide.cog"]
    out = tmp_data / "dl" / item.filename
    out.parent.mkdir()
    rasters.write_image(item, out)
    info = json.loads(subprocess.run(["gdalinfo", "-json", str(out)], check=True,
                                     capture_output=True, text=True).stdout)
    assert info["metadata"]["IMAGE_STRUCTURE"]["LAYOUT"] == "COG"
    assert len(info["bands"]) == 4
    assert info["stac"]["proj:epsg"] == 2227
    md = info["metadata"][""]
    assert "Based on sfzoning.com by Steven Bacio for GrowSF (https://growsf.org)" in md["TIFFTAG_COPYRIGHT"]
    assert config.RUMSEY_CREDIT in md["TIFFTAG_COPYRIGHT"]
    assert md["TIFFTAG_IMAGEDESCRIPTION"] == "1948 edition"
    assert not out.with_name(out.name + ".part").exists()


def test_write_sidecars_writes_readme_and_license_and_returns_index_paths(tmp_data):
    from sfzmaps.downloads import package
    item = {i.key: i for i in downloads.catalog()}["1948/citywide.cog"]
    out = tmp_data / "dl" / "1948" / item.filename
    out.parent.mkdir(parents=True)
    assert package.write_sidecars(out, item, "2026-11-04", "abc1234") == [
        "1948/1948-citywide.README.txt", "1948/1948-citywide.LICENSE.txt"]
    assert "Based on sfzoning.com" in (out.parent / "1948-citywide.README.txt").read_text()
    assert "CC BY-NC-SA 4.0" in (out.parent / "1948-citywide.LICENSE.txt").read_text()


@needs_gdal_raster
def test_citywide_image_is_deterministic_and_dateless(tmp_data, monkeypatch):
    from sfzmaps.downloads import rasters
    shutil.rmtree(tmp_data / "work" / "1948")
    _make_sheets(tmp_data / "work")
    _fake_build_vrt(monkeypatch)
    item = {i.key: i for i in downloads.catalog()}["1948/citywide.cog"]
    (tmp_data / "dl").mkdir()
    shas = []
    for day in ("2026-11-04", "2027-02-09"):
        monkeypatch.setattr(downloads, "_today", lambda day=day: day)
        out = tmp_data / "dl" / f"{day.replace('-', '')[4:]}.tif"
        rasters.write_image(item, out)
        shas.append(hashlib.sha256(out.read_bytes()).hexdigest())
        info = subprocess.run(["gdalinfo", str(out)], check=True, capture_output=True, text=True).stdout
        assert "TIFFTAG_IMAGEDESCRIPTION=1948 edition" in info
        assert day[:4] not in info.replace(str(out), "")
    assert shas[0] == shas[1]


def test_sheet_entries_md5_matches_file(tmp_data):
    from sfzmaps.downloads import sheets as rasters
    for sid, data in (("10", b"ten"), ("2", b"two" * 1000)):
        _touch(tmp_data / "work" / "1948" / sid / "warped_paper.tif", data)
    (tmp_data / "work" / "1948" / "9").mkdir()  # no warped_paper.tif: not listed
    es = rasters.sheet_entries("1948")
    assert [e["sheet"] for e in es] == ["1", "2", "10"]
    e = next(e for e in es if e["sheet"] == "2")
    raw = (tmp_data / "work" / "1948" / "2" / "warped_paper.tif").read_bytes()
    assert e == {"sheet": "2", "name": "1948-sheet-2.tif", "md5": hashlib.md5(raw).hexdigest(),
                 "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
                 "mtime_ns": (tmp_data / "work" / "1948" / "2" / "warped_paper.tif").stat().st_mtime_ns}


def test_failed_tool_call_shows_gdal_stderr(tmp_path):
    with pytest.raises(RuntimeError, match="ogr2ogr failed.*nonexistent"):
        downloads.run_tool("ogr2ogr", str(tmp_path / "o.gpkg"), str(tmp_path / "nonexistent.fgb"))


def test_sheet_entries_sort_naturally(tmp_data):
    from sfzmaps.downloads import sheets as rasters
    for sid in ("10", "5a", "2", "5", "1"):
        _touch(tmp_data / "work" / "1948" / sid / "warped_paper.tif")
    assert [e["sheet"] for e in rasters.sheet_entries("1948")] == ["1", "2", "5", "5a", "10"]
    for sid in ("1948-02-01T000000", "1948-01-31T235959"):
        _touch(tmp_data / "work" / "t" / sid / "warped_paper.tif")
    assert [e["sheet"] for e in rasters.sheet_entries("t")] == ["1948-01-31T235959", "1948-02-01T000000"]


@needs_gdal_raster
def test_failed_gdal_translate_leaves_no_part(tmp_data, monkeypatch):
    from sfzmaps.downloads import rasters
    shutil.rmtree(tmp_data / "work" / "1948")
    _make_sheets(tmp_data / "work")
    _fake_build_vrt(monkeypatch)
    monkeypatch.setattr(rasters, "run_tool", lambda *a: (Path(a[-1]).write_bytes(b"half"), 1 / 0))
    item = {i.key: i for i in downloads.catalog()}["1948/citywide.cog"]
    out = tmp_data / "x.tif"
    with pytest.raises(ZeroDivisionError):
        rasters.write_image(item, out)
    assert not out.with_name("x.tif.part").exists()


def test_readme_sheets_layer():
    t = _flat(readme("1948", "sheets", "2026-11-04", "abc1234"))
    assert "own resolution" in t and "EPSG:2227" in t
    assert "David Rumsey" in t and "Based on sfzoning.com by Steven Bacio" in t
    assert "FIELDS" not in t


# --- run() and index.json ---------------------------------------------------

import json
import os


@pytest.fixture
def fake_exports(tmp_data, monkeypatch):
    """Stand-in data writers (GDAL-free); records which items had their data written, and when.

    Packaging (zip, README, LICENSE) is the real code."""
    monkeypatch.setattr(downloads, "DOWNLOADS_DIR", tmp_data / "downloads")
    monkeypatch.setattr(downloads, "_commit", lambda: "abc1234")
    written = []

    def layers(item, tmp):
        written.append((item.key, downloads._today()))
        f = tmp / f"{item.series}-{item.layer}.{item.fmt}.dat"
        f.write_bytes(f"{item.key}".encode())
        g = tmp / f"{item.series}-{item.layer}.aux"
        g.write_bytes(b"aux")
        return [f, g]

    def image(item, out):
        written.append((item.key, downloads._today()))
        out.write_bytes(b"cog")
    monkeypatch.setattr("sfzmaps.downloads.vectors.write_layers", layers)
    monkeypatch.setattr("sfzmaps.downloads.rasters.write_image", image)
    return written


def _mtimes(root):
    return {str(p): p.stat().st_mtime_ns for p in root.rglob("*") if p.is_file()}


def test_rerun_rewrites_nothing(fake_exports, tmp_data):
    first = downloads.run()
    root = tmp_data / "downloads"
    assert len(fake_exports) == 7  # 6 vector + 1 citywide
    entry = first["files"]["1948/lots.geojson"]
    assert entry["path"] == "1948/1948-lots.geojson.zip" and entry["layer"] == "lots"
    assert entry["bytes"] == (root / entry["path"]).stat().st_size
    assert first["files"]["1948/citywide.cog"]["extra"] == [
        "1948/1948-citywide.README.txt", "1948/1948-citywide.LICENSE.txt"]
    assert first["files"]["1948/lots.geojson"]["extra"] == []
    assert first["commit"] == "abc1234" and "CC BY-NC-SA 4.0" in first["license"]
    assert [e["sheet"] for e in first["sheets"]["1948"]] == ["1"]
    assert "Rumsey" in first["readme"]["1948"] or "David Rumsey" in first["readme"]["1948"]
    before, text = _mtimes(root), (root / "index.json").read_text()
    fake_exports.clear()
    second = downloads.run()
    assert fake_exports == [] and _mtimes(root) == before
    assert (root / "index.json").read_text() == text and second == first


def test_changed_source_rewrites_only_that_series(fake_exports, tmp_data, monkeypatch):
    _touch(tmp_data / "out" / "today" / "districts" / "parcels.fgb")
    _touch(tmp_data / "out" / "today" / "districts" / "districts.fgb")
    downloads.run()
    fake_exports.clear()
    monkeypatch.setattr("sfzmaps.downloads._today", lambda: "2030-01-02")
    _touch(tmp_data / "out" / "1948" / "districts" / "parcels.fgb", b"changed")
    idx = downloads.run()
    assert sorted(k for k, _ in fake_exports) == [
        "1948/lots.geojson", "1948/lots.gpkg", "1948/lots.shp"]
    assert {u for _, u in fake_exports} == {"2030-01-02"}
    assert idx["files"]["today/lots.geojson"]["fingerprint"] and idx["files"]["today/lots.geojson"]["data_fingerprint"]


def test_unchanged_sheet_is_not_rehashed(tmp_data, monkeypatch):
    from sfzmaps import stitch
    from sfzmaps.downloads import sheets as rasters
    first = rasters.sheet_entries("1948")
    calls = []
    monkeypatch.setattr(rasters, "_md5", lambda p: calls.append(p) or "x")
    monkeypatch.setattr(stitch, "_sha256", lambda p: calls.append(p) or "x")
    assert rasters.sheet_entries("1948", first) == first and calls == []
    _touch(tmp_data / "work" / "1948" / "1" / "warped_paper.tif", b"changed!")
    assert rasters.sheet_entries("1948", first)[0]["md5"] == "x" and calls


def test_unknown_series_is_rejected(fake_exports):
    with pytest.raises(ValueError, match="unknown series: 19488"):
        downloads.run(["19488"])


def test_index_written_atomically(fake_exports, tmp_data):
    downloads.run()
    root = tmp_data / "downloads"
    assert (root / "index.json").exists() and not (root / "index.json.part").exists()


def test_sheets_readme_date_survives_later_unchanged_rerun(fake_exports, tmp_data, monkeypatch):
    monkeypatch.setattr("sfzmaps.downloads._today", lambda: "2026-11-04")
    first = downloads.run()
    monkeypatch.setattr("sfzmaps.downloads._today", lambda: "2027-03-09")
    second = downloads.run()
    assert "2026-11-04" in first["readme"]["1948"]
    assert second["readme"]["1948"] == first["readme"]["1948"]


def test_sheets_readme_refreshes_when_its_text_changes(fake_exports, tmp_data, monkeypatch):
    from sfzmaps.downloads import readme as readme_mod
    monkeypatch.setattr("sfzmaps.downloads._today", lambda: "2026-11-04")
    downloads.run()
    monkeypatch.setattr("sfzmaps.downloads._today", lambda: "2027-03-09")
    real = readme_mod.readme
    monkeypatch.setattr(readme_mod, "readme", lambda *a: real(*a).replace("WHAT THIS FILE IS", "WHAT THIS IS"))
    second = downloads.run()
    assert "WHAT THIS IS" in second["readme"]["1948"] and "2027-03-09" in second["readme"]["1948"]


def test_run_limited_to_series_keeps_other_entries(fake_exports, tmp_data):
    _touch(tmp_data / "out" / "today" / "districts" / "parcels.fgb")
    _touch(tmp_data / "out" / "today" / "districts" / "districts.fgb")
    downloads.run()
    idx = downloads.run(["1948"])
    assert any(k.startswith("today/") for k in idx["files"])


# --- GIS download names, versions link, made date ----------------------------

def test_latest_name():
    from sfzmaps.downloads import latest_name
    assert latest_name("1948", "lots", "gpkg") == "sfzoning-1948-lots-latest.gpkg.zip"
    assert latest_name("1921", "citywide", "cog") == "sfzoning-1921-citywide-latest.tif"


def test_readme_points_to_latest_and_versions():
    from sfzmaps.downloads import readme as readme_mod
    t = readme_mod.readme("1948", "lots", "2026-11-20", "abc1234", "gpkg")
    assert ("GETTING THE NEWEST VERSION\nFor the newest version of this file, use:\n"
            "  https://sfzoning.com/data/files/1948/sfzoning-1948-lots-latest.gpkg.zip\n"
            "Every version released so far:\n  https://sfzoning.com/data/versions/1948/lots\n") in t
    assert t.index("LAST UPDATED") < t.index("GETTING THE NEWEST VERSION") < t.index("LICENSE\n")


def test_sheets_readme_points_to_latest_zip_and_series_versions():
    from sfzmaps.downloads import readme as readme_mod
    t = readme_mod.readme("1921", "sheets", "2026-11-20", "abc1234")
    assert "https://sfzoning.com/data/files/1921/sfzoning-1921-sheets-latest.zip" in t
    assert "https://sfzoning.com/data/versions/1921\n" in t


def test_index_records_made_date_and_keeps_it(fake_exports, tmp_data, monkeypatch):
    monkeypatch.setattr("sfzmaps.downloads._today", lambda: "2026-11-20")
    first = downloads.run()
    assert first["files"]["1948/lots.geojson"]["made"] == "2026-11-20"
    monkeypatch.setattr("sfzmaps.downloads._today", lambda: "2026-12-01")
    fake_exports.clear()
    second = downloads.run()
    assert fake_exports == []
    assert second["files"]["1948/lots.geojson"]["made"] == "2026-11-20"
    _touch(tmp_data / "out" / "1948" / "districts" / "parcels.fgb", b"changed")
    third = downloads.run()
    assert third["files"]["1948/lots.geojson"]["made"] == "2026-12-01"
    assert third["files"]["1948/districts.geojson"]["made"] == "2026-11-20"


# --- two change checks: text-only changes never run GDAL ----------------------

def _text_only_change(monkeypatch, tmp_data):
    """Point TEXT_CODE at a temp copy of the text code and return a function that edits it."""
    code = tmp_data / "textcode"
    files = [code / n for n in ("readme.py", "readme-template.txt", "package.py")]
    for f in files:
        _touch(f, b"v1")
    monkeypatch.setattr(downloads, "TEXT_CODE", files)
    return lambda: _touch(files[1], b"v2")


def _new_readme_text(monkeypatch, marker="NEW WORDS"):
    from sfzmaps.downloads import package, readme as readme_mod
    real = readme_mod.readme
    monkeypatch.setattr(package, "readme", lambda *a: real(*a) + marker)


def test_text_only_change_rewrites_sidecars_without_gdal(fake_exports, tmp_data, monkeypatch):
    edit = _text_only_change(monkeypatch, tmp_data)
    monkeypatch.setattr(downloads, "_today", lambda: "2026-11-20")
    first = downloads.run()
    root = tmp_data / "downloads"
    tif = root / first["files"]["1948/citywide.cog"]["path"]
    tif_bytes, tif_mtime = tif.read_bytes(), tif.stat().st_mtime_ns
    fake_exports.clear()
    edit()
    _new_readme_text(monkeypatch)
    monkeypatch.setattr(downloads, "_today", lambda: "2026-12-01")
    second = downloads.run()
    assert fake_exports == []  # neither write_image nor write_layers ran
    city = second["files"]["1948/citywide.cog"]
    assert "NEW WORDS" in (root / city["extra"][0]).read_text()
    assert tif.read_bytes() == tif_bytes and tif.stat().st_mtime_ns == tif_mtime
    assert city["sha256"] == first["files"]["1948/citywide.cog"]["sha256"]
    assert city["fingerprint"] != first["files"]["1948/citywide.cog"]["fingerprint"]
    assert city["data_fingerprint"] == first["files"]["1948/citywide.cog"]["data_fingerprint"]
    assert {f["made"] for f in second["files"].values()} == {"2026-12-01"}
    with zipfile.ZipFile(root / second["files"]["1948/lots.geojson"]["path"]) as z:
        assert z.read("README.txt").endswith(b"NEW WORDS")
    third = downloads.run()  # and now it is steady again
    assert third == second and fake_exports == []


def test_text_only_vector_rezip_keeps_data_members(fake_exports, tmp_data, monkeypatch):
    edit = _text_only_change(monkeypatch, tmp_data)
    first = downloads.run()
    root = tmp_data / "downloads"
    path = root / first["files"]["1948/lots.shp"]["path"]
    with zipfile.ZipFile(path) as z:
        before = [(n, z.read(n)) for n in z.namelist() if n not in ("README.txt", "LICENSE.txt")]
        old_readme = z.read("README.txt")
    assert [n for n, _ in before] == ["1948-lots.aux", "1948-lots.shp.dat"]
    edit()
    _new_readme_text(monkeypatch)
    second = downloads.run()
    with zipfile.ZipFile(path) as z:
        after = [(n, z.read(n)) for n in z.namelist() if n not in ("README.txt", "LICENSE.txt")]
        assert z.read("README.txt") != old_readme and z.read("README.txt").endswith(b"NEW WORDS")
        assert "LICENSE.txt" in z.namelist()
    assert after == before
    assert second["files"]["1948/lots.shp"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert second["files"]["1948/lots.shp"]["bytes"] == path.stat().st_size


def test_old_index_without_data_fingerprint_rebuilds_fully(fake_exports, tmp_data):
    downloads.run()
    ip = tmp_data / "downloads" / "index.json"
    idx = json.loads(ip.read_text())
    for f in idx["files"].values():
        del f["data_fingerprint"]
    ip.write_text(json.dumps(idx))
    fake_exports.clear()
    new = downloads.run()
    assert len(fake_exports) == 7
    assert all(f["data_fingerprint"] for f in new["files"].values())


def test_text_only_with_missing_output_rebuilds_fully(fake_exports, tmp_data, monkeypatch):
    edit = _text_only_change(monkeypatch, tmp_data)
    first = downloads.run()
    root = tmp_data / "downloads"
    (root / first["files"]["1948/lots.gpkg"]["path"]).unlink()
    (root / first["files"]["1948/citywide.cog"]["path"]).unlink()
    edit()
    fake_exports.clear()
    downloads.run()
    assert sorted(k for k, _ in fake_exports) == ["1948/citywide.cog", "1948/lots.gpkg"]
    assert (root / first["files"]["1948/lots.gpkg"]["path"]).exists()


def test_text_only_with_missing_sidecar_rebuilds_the_image(fake_exports, tmp_data, monkeypatch):
    edit = _text_only_change(monkeypatch, tmp_data)
    first = downloads.run()
    (tmp_data / "downloads" / first["files"]["1948/citywide.cog"]["extra"][1]).unlink()
    edit()
    fake_exports.clear()
    downloads.run()
    assert fake_exports == [("1948/citywide.cog", downloads._today())]


def test_vector_code_change_does_not_touch_images(fake_exports, tmp_data, monkeypatch):
    code = tmp_data / "code"
    _touch(code / "vectors.py", b"v1")
    _touch(code / "rasters.py", b"v1")
    monkeypatch.setattr(downloads, "DATA_CODE", {"citywide": [code / "rasters.py"],
                        "lots": [code / "vectors.py"], "districts": [code / "vectors.py"]})
    downloads.run()
    fake_exports.clear()
    _touch(code / "vectors.py", b"v2")
    downloads.run()
    keys = sorted(k for k, _ in fake_exports)
    assert keys == sorted(f"1948/{l}.{f}" for l in ("lots", "districts") for f in ("geojson", "gpkg", "shp"))


def test_fields_change_reruns_vector_data_but_not_images(fake_exports, tmp_data, monkeypatch):
    code = tmp_data / "code"
    _touch(code / "vectors.py", b"v1")
    _touch(code / "fields.py", b"v1")
    _touch(code / "rasters.py", b"v1")
    monkeypatch.setattr(downloads, "DATA_CODE", {"citywide": [code / "rasters.py"],
                        "lots": [code / "vectors.py", code / "fields.py"],
                        "districts": [code / "vectors.py", code / "fields.py"]})
    monkeypatch.setattr(downloads, "TEXT_CODE", [code / "fields.py"])
    downloads.run()
    fake_exports.clear()
    _touch(code / "fields.py", b"v2")
    downloads.run()
    keys = sorted(k for k, _ in fake_exports)
    assert keys == sorted(f"1948/{l}.{f}" for l in ("lots", "districts") for f in ("geojson", "gpkg", "shp"))
