import shutil
import sqlite3
from pathlib import Path

import pytest

from sfzmaps import tiles


def test_report_from_mbtiles(tmp_path):
    p = tmp_path / "x.mbtiles"
    con = sqlite3.connect(p)
    con.execute("create table tiles (zoom_level int, tile_column int, tile_row int, tile_data blob)")
    con.executemany("insert into tiles values (?,?,?,?)",
                    [(11, 0, 0, b"a" * 5), (11, 1, 0, b"a" * 7), (12, 0, 0, b"b" * 4)])
    con.commit()
    con.close()
    r = tiles.report_from_mbtiles(p)
    assert r.bytes_total == 16 and r.bytes_by_zoom == {11: 12, 12: 4}


def test_missing_gdal_tools_point_to_pixi_and_no_pmtiles_cli_is_needed(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda n: None)
    with pytest.raises(RuntimeError, match="pixi shell") as e:
        tiles.require_tools()
    assert "pmtiles" not in str(e.value)


def test_convert_writes_a_pmtiles_archive_from_mbtiles(tmp_path):
    from pmtiles.reader import MmapSource, Reader
    mb = tmp_path / "t.mbtiles"
    con = sqlite3.connect(mb)
    con.execute("create table metadata (name text, value text)")
    con.execute("create table tiles (zoom_level int, tile_column int, tile_row int, tile_data blob)")
    con.executemany("insert into metadata values (?, ?)", [("format", "webp"), ("minzoom", "11"), ("maxzoom", "11"),
                    ("bounds", "-122.5,37.7,-122.3,37.8"), ("center", "-122.4,37.75,11")])
    con.execute("insert into tiles values (11, 327, 1255, ?)", (b"RIFFtile",))
    con.commit()
    con.close()
    tiles._convert(mb, tmp_path / "t.pmtiles")
    with open(tmp_path / "t.pmtiles", "rb") as f:
        r = Reader(MmapSource(f))
        h = r.header()
        assert r.get(11, 327, (1 << 11) - 1 - 1255) == b"RIFFtile"      # TMS row flipped to XYZ
        assert h["center_zoom"] == 11 and round(h["center_lon_e7"] / 1e7, 1) == -122.4


def test_set_metadata_center(tmp_path):
    p = tmp_path / "x.mbtiles"
    con = sqlite3.connect(p)
    con.execute("create table metadata (name text, value text)")
    con.execute("insert into metadata values ('bounds', '-122.5,37.7,-122.3,37.8')")
    con.commit()
    con.close()
    tiles.set_metadata(p, "SF zoning 1948", "credit", 11)
    md = dict(sqlite3.connect(p).execute("select name, value from metadata").fetchall())
    assert md["center"] == "-122.400000,37.750000,11" and md["attribution"] == "credit"
    assert md["name"] == "SF zoning 1948"


def test_default_max_zoom_is_18():
    import inspect
    from sfzmaps import cli
    assert inspect.signature(tiles.make_pmtiles).parameters["max_zoom"].default == 18
    assert cli._publish.__defaults__ == (11, 18)


def _fake_tools(monkeypatch, fail_convert):
    from sfzmaps import tiles

    def run(cmd):
        Path(cmd[-1]).write_bytes(b"x")

    def convert(mb, out):
        if fail_convert:
            raise RuntimeError("convert failed: boom")
        Path(out).write_bytes(b"NEW")

    monkeypatch.setattr(tiles, "require_tools", lambda: None)
    monkeypatch.setattr(tiles, "_run", run)
    monkeypatch.setattr(tiles, "_convert", convert)
    monkeypatch.setattr(tiles, "report_from_mbtiles", lambda p: tiles.TileReport(1, {11: 1}))
    monkeypatch.setattr(tiles, "set_metadata", lambda *a, **k: None)


def test_failed_convert_keeps_old_pmtiles(tmp_path, monkeypatch):
    """M4: the old PMTiles survives a failing `pmtiles convert`; a good one replaces it."""
    import pytest

    from sfzmaps import tiles

    from sfzmaps import config
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    out = tmp_path / "1948.pmtiles"
    out.write_bytes(b"OLD")
    _fake_tools(monkeypatch, fail_convert=True)
    with pytest.raises(RuntimeError):
        tiles.make_pmtiles(tmp_path / "x.vrt", out)
    assert out.read_bytes() == b"OLD"
    _fake_tools(monkeypatch, fail_convert=False)
    tiles.make_pmtiles(tmp_path / "x.vrt", out)
    assert out.read_bytes() == b"NEW"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["1948.pmtiles", "data"]
    assert list((tmp_path / "data" / "tmp").iterdir()) == []      # scratch cleaned up


def test_tiling_scratch_never_lands_beside_the_output(tmp_path, monkeypatch):
    """data/out is versioned by DVC: a crash mid-tiling must leave nothing in it (crashed runs
    once left 13 GB of web.tif and tile databases there)."""
    import pytest

    from sfzmaps import config
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    out_dir = tmp_path / "data" / "out" / "1948"
    out_dir.mkdir(parents=True)
    seen = []
    _fake_tools(monkeypatch, fail_convert=True)
    real_run = tiles._run
    monkeypatch.setattr(tiles, "_run", lambda cmd: (seen.append(Path(cmd[-1]).parent), real_run(cmd)))
    with pytest.raises(RuntimeError):
        tiles.make_pmtiles(tmp_path / "x.vrt", out_dir / "1948.pmtiles")
    assert seen and all((tmp_path / "data" / "tmp") in p.parents for p in seen)
    assert list(out_dir.iterdir()) == []
