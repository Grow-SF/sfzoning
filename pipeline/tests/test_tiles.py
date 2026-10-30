import shutil
import sqlite3

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


def test_missing_pmtiles_cli_message(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda n: None)
    with pytest.raises(RuntimeError, match="brew install pmtiles"):
        tiles.require_tools()


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
