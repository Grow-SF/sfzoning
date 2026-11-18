import gzip
import hashlib
import json

import pytest
from shapely.geometry import Polygon, mapping

from sfzmaps.solver import paths, table

# three lots, as longitude/latitude boxes near the Ferry Building
_ROWS = [("0001001", " rh-1 ", (-122.40, 37.79)), ("0001002", None, (-122.39, 37.79)),
         ("0001003", "RH-1/RM-1", (-122.38, 37.79))]


def _geojson():
    feats = []
    for blklot, zoning, (x, y) in _ROWS:
        box = Polygon([(x, y), (x + 0.001, y), (x + 0.001, y + 0.001), (x, y + 0.001)])
        feats.append({"type": "Feature", "properties": {"blklot": blklot, "zoning": zoning},
                      "geometry": mapping(box)})
    return json.dumps({"type": "FeatureCollection", "features": feats}).encode()


@pytest.fixture
def table_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "TABLE_DIR", tmp_path)
    (tmp_path / "april2000.geojson.gz").write_bytes(gzip.compress(_geojson()))
    return tmp_path


def test_load_table_projects_tidies_and_keeps_order(table_dir):
    t = table.load_table()
    assert t.crs.to_epsg() == 2227
    assert list(t["blklot"]) == ["0001001", "0001002", "0001003"]
    assert list(t["zoning"]) == ["RH-1", "", "RH-1/RM-1"]
    assert t.geometry.is_valid.all()
    assert t["area"].iloc[0] == pytest.approx(t.geometry.iloc[0].area)
    assert 5.9e6 < t.geometry.iloc[0].centroid.x < 6.1e6  # state plane feet, not degrees


def test_known_and_codes(table_dir):
    known, codes = table.known_and_codes(table.load_table())
    assert list(known) == [True, False, False]
    assert list(codes) == ["RH-1", "", ""]


def test_corrected_codes_apply_only_verdict_a(table_dir):
    (table_dir / "table-corrections.csv").write_text(
        "group,verdict,pass,A,B,lots,blklots\n"
        "0,A,1,RM-1,RH-1,2,0001001 9999999\n"
        "1,B,1,RM-2,RH-1,1,0001002\n")
    codes = table.corrected_codes(table.load_table())
    assert list(codes) == ["RM-1", "", ""]


def test_fetch_table_writes_gz_and_provenance(table_dir, monkeypatch):
    raw = _geojson()
    monkeypatch.setattr(table, "_download", lambda url: raw)
    dest = table_dir / "new"
    info = table.fetch_table(dest)
    assert gzip.decompress((dest / "april2000.geojson.gz").read_bytes()) == raw
    assert info["rows"] == 3 and info["sha256"] == hashlib.sha256(raw).hexdigest()
    text = (dest / "PROVENANCE.md").read_text()
    assert hashlib.sha256(raw).hexdigest() in text and "PDDL 1.0" in text
    assert "| rows | 3 |" in text
    first = (dest / "april2000.geojson.gz").read_bytes()
    table.fetch_table(dest)
    assert (dest / "april2000.geojson.gz").read_bytes() == first
