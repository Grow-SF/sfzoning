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
    r = readme("1948", "citywide", "2026-11-04", "abc1234")
    assert "Based on sfzoning.com by Steven Bacio for GrowSF (https://growsf.org)" in r
    assert "CC BY-NC-SA 4.0" in r and config.RUMSEY_CREDIT in r and "2026-11-04" in r
    assert config.RUMSEY_CREDIT not in readme("1960", "lots", "2026-11-04", "abc1234")
    assert "not an official record" in readme("1960", "lots", "2026-11-04", "abc1234")
    assert "DRAFT COPY" not in r and "{{" not in r


def test_readme_field_table_only_for_vector_layers():
    r = readme("1960", "lots", "d", "c")
    row = next(l for l in r.splitlines() if l.startswith("height_code"))
    assert row.split()[1] == "height" and "height_cd" not in r
    assert "FIELDS" not in readme("1960", "citywide", "d", "c")


def test_readme_crs_text_per_layer():
    lots = readme("1960", "lots", "d", "c")
    assert "GeoJSON and Shapefile are in WGS 84 (EPSG:4326)" in lots
    assert "`lots_ca_spcs`" in lots and "EPSG:2227" in lots
    assert "`districts_ca_spcs`" in readme("1960", "districts", "d", "c")
    city = readme("1960", "citywide", "d", "c")
    assert "EPSG:2227" in city and "4326" not in city


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


def _export(tmp_path, layer, fmt):
    item = _item(tmp_path, layer, fmt)
    out = tmp_path / "out" / item.filename
    out.parent.mkdir(parents=True, exist_ok=True)
    vectors.export_vector(item, out, "2026-11-04", "abc1234")
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
    vectors.export_vector(_item(tmp_path, "lots", "geojson"), out2, "2026-11-04", "abc1234")
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
    rasters.export_citywide(item, out, "2026-11-04", "abc1234")
    info = json.loads(subprocess.run(["gdalinfo", "-json", str(out)], check=True,
                                     capture_output=True, text=True).stdout)
    assert info["metadata"]["IMAGE_STRUCTURE"]["LAYOUT"] == "COG"
    assert len(info["bands"]) == 4
    assert info["stac"]["proj:epsg"] == 2227
    md = info["metadata"][""]
    assert "Based on sfzoning.com by Steven Bacio for GrowSF (https://growsf.org)" in md["TIFFTAG_COPYRIGHT"]
    assert config.RUMSEY_CREDIT in md["TIFFTAG_COPYRIGHT"]
    assert "2026-11-04" in md["TIFFTAG_IMAGEDESCRIPTION"]
    assert not out.with_name(out.name + ".part").exists()
    assert "Based on sfzoning.com" in (tmp_data / "dl" / "1948-citywide.README.txt").read_text()
    assert "CC BY-NC-SA 4.0" in (tmp_data / "dl" / "1948-citywide.LICENSE.txt").read_text()


def test_sheet_entries_md5_matches_file(tmp_data):
    from sfzmaps.downloads import rasters
    for sid, data in (("10", b"ten"), ("2", b"two" * 1000)):
        _touch(tmp_data / "work" / "1948" / sid / "warped_paper.tif", data)
    (tmp_data / "work" / "1948" / "9").mkdir()  # no warped_paper.tif: not listed
    es = rasters.sheet_entries("1948")
    assert [e["sheet"] for e in es] == ["1", "10", "2"]
    e = next(e for e in es if e["sheet"] == "2")
    raw = (tmp_data / "work" / "1948" / "2" / "warped_paper.tif").read_bytes()
    assert e == {"sheet": "2", "name": "1948-sheet-2.tif", "md5": hashlib.md5(raw).hexdigest(),
                 "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def test_failed_tool_call_shows_gdal_stderr(tmp_path):
    with pytest.raises(RuntimeError, match="ogr2ogr failed.*nonexistent"):
        downloads.run_tool("ogr2ogr", str(tmp_path / "o.gpkg"), str(tmp_path / "nonexistent.fgb"))
