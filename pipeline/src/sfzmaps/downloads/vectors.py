"""Lot and district downloads: GeoJSON, GeoPackage and Shapefile zips built with ogr2ogr.

The source is the pipeline's FlatGeobuf in EPSG:2227. No geometry is simplified. Zips are
deterministic (fixed member dates, sorted entries) so unchanged data gives identical bytes.
"""
from __future__ import annotations

import subprocess
import tempfile
import zipfile
from pathlib import Path

from sfzmaps.downloads import Item
from sfzmaps.downloads.readme import DISTRICT_FIELDS, LICENSE_TEXT, LOT_FIELDS, readme

_FIELDS = {"lots": LOT_FIELDS, "districts": DISTRICT_FIELDS}
_FIXED_DATE = (1980, 1, 1, 0, 0, 0)


def _ogr(*args: str) -> None:
    subprocess.run(["ogr2ogr", "-q", *args], check=True, capture_output=True, text=True)


def _select(layer: str, src: Path, rename: bool) -> str:
    cols = ", ".join(f"{s} AS {n}" if rename and s != n else s for s, n, _ in _FIELDS[layer])
    return f"SELECT {cols} FROM {src.stem}"


def _write_data(item: Item, tmp: Path) -> list[Path]:
    src = item.sources[0]
    layer = item.layer
    base = f"{item.series}-{layer}"
    if item.fmt == "geojson":
        out = tmp / f"{base}.geojson"
        _ogr("-f", "GeoJSON", "-t_srs", "EPSG:4326", "-lco", "RFC7946=YES",
             "-lco", "COORDINATE_PRECISION=6", "-dialect", "OGRSQL", "-sql", _select(layer, src, False),
             "-nln", layer, str(out), str(src))
        return [out]
    if item.fmt == "gpkg":
        out = tmp / f"{base}.gpkg"
        sql = _select(layer, src, False)
        _ogr("-f", "GPKG", "-t_srs", "EPSG:4326", "-dialect", "OGRSQL", "-sql", sql,
             "-nln", layer, str(out), str(src))
        _ogr("-f", "GPKG", "-update", "-dialect", "OGRSQL", "-sql", sql,
             "-nln", f"{layer}_ca_spcs", str(out), str(src))
        return [out]
    if item.fmt == "shp":
        _ogr("-f", "ESRI Shapefile", "-t_srs", "EPSG:4326", "-dialect", "OGRSQL",
             "-sql", _select(layer, src, True), "-nln", f"{base}", str(tmp), str(src))
        return sorted(p for p in tmp.glob(f"{base}.*"))
    raise ValueError(f"not a vector format: {item.fmt}")


def export_vector(item: Item, out: Path, updated: str, commit: str) -> None:
    """Write the zip for one lots/districts item at `out`."""
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        members = {p.name: p.read_bytes() for p in _write_data(item, tmp)}
    members["README.txt"] = readme(item.series, item.layer, updated, commit).encode()
    members["LICENSE.txt"] = LICENSE_TEXT.encode()
    tmp_out = out.with_name(out.name + ".part")
    with zipfile.ZipFile(tmp_out, "w", zipfile.ZIP_DEFLATED) as z:
        for name in sorted(members):
            info = zipfile.ZipInfo(name, _FIXED_DATE)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            z.writestr(info, members[name])
    tmp_out.replace(out)
