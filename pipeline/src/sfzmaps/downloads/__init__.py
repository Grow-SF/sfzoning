"""GIS downloads: what is published (the catalog) and when it is out of date (fingerprints).

No network and no paid reads. Not a DVC stage.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from sfzmaps import config, stitch
from sfzmaps.districts.series import SERIES

DOWNLOADS_DIR = config.DATA_DIR / "downloads"
EXPORTER_VERSION = "1"  # bump when the format of any output changes

VECTOR_FORMATS = ("geojson", "gpkg", "shp")
_EXT = {"geojson": "geojson.zip", "gpkg": "gpkg.zip", "shp": "shp.zip", "cog": "tif"}


@dataclass(frozen=True)
class Item:
    key: str  # "<series>/<layer>.<fmt>"
    series: str
    layer: Literal["lots", "districts", "citywide"]
    fmt: Literal["geojson", "gpkg", "shp", "cog"]
    filename: str
    sources: list[Path]


def _item(series: str, layer: str, fmt: str, sources: list[Path]) -> Item:
    return Item(f"{series}/{layer}.{fmt}", series, layer, fmt,  # type: ignore[arg-type]
                f"{series}-{layer}.{_EXT[fmt]}", sources)


def citywide_sources(series: str) -> list[Path]:
    """Every placed sheet's warped_paper.tif and footprint, plus the stitching code."""
    tifs = sorted((config.WORK_DIR / series).glob("*/warped_paper.tif"))
    out: list[Path] = []
    for t in tifs:
        fp = t.with_name("warped_paper.footprint.geojson")
        out += [t, fp] if fp.exists() else [t]
    return out + [Path(stitch.__file__)] if tifs else []


def catalog() -> list[Item]:
    items: list[Item] = []
    vector_series = [s for s, i in SERIES.items() if i.role == "use"] + ["today"]
    for s in vector_series:
        d = config.OUT_DIR / s / "districts"
        if not (d / "parcels.fgb").exists():
            continue
        for layer, name in (("lots", "parcels.fgb"), ("districts", "districts.fgb")):
            for fmt in VECTOR_FORMATS:
                items.append(_item(s, layer, fmt, [d / name]))
    for s in SERIES:
        src = citywide_sources(s)
        if src:
            items.append(_item(s, "citywide", "cog", src))
    return items


def _rel(p: Path) -> str:
    for base in (config.DATA_DIR, Path(stitch.__file__).parent):
        try:
            return str(p.relative_to(base))
        except ValueError:
            pass
    return p.name


def fingerprint(item: Item, exporter_version: str) -> str:
    h = hashlib.sha256()
    for rel, digest in sorted((_rel(p), stitch._sha256(p)) for p in item.sources):
        h.update(f"{rel}\0{digest}\n".encode())
    h.update(f"exporter:{exporter_version}".encode())
    return h.hexdigest()
