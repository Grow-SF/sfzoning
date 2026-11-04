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
    assert "height_cd" in readme("1960", "lots", "d", "c")
    assert "FIELDS" not in readme("1960", "citywide", "d", "c")
