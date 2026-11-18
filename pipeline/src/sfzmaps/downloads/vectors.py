"""Lot and district downloads: GeoJSON, GeoPackage and Shapefile zips built with ogr2ogr.

The source is the pipeline's FlatGeobuf in EPSG:2227. No geometry is simplified.
Packing them into a zip with a README is package.py.
"""
from __future__ import annotations

from pathlib import Path

from sfzmaps.downloads import Item, run_tool
from sfzmaps.downloads.readme import DISTRICT_FIELDS, LOT_FIELDS

_FIELDS = {"lots": LOT_FIELDS, "districts": DISTRICT_FIELDS}


def _ogr(*args: str) -> None:
    run_tool("ogr2ogr", "-q", *args)


def _select(layer: str, src: Path, rename: bool) -> str:
    cols = ", ".join(f"{s} AS {n}" if rename and s != n else s for s, n, _ in _FIELDS[layer])
    return f"SELECT {cols} FROM {src.stem}"


def write_layers(item: Item, tmp: Path) -> list[Path]:
    """Write the GIS data files for one lots/districts item into `tmp`; return them (no zip, no README)."""
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
