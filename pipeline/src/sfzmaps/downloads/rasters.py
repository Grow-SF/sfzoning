"""Citywide image (a Cloud Optimized GeoTIFF) and the per-sheet download entries.

The image is the same paper stitch the site shows (`stitch.build_vrt(paper=True)`), copied into
a COG with no resampling: the scans stay as scanned.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from sfzmaps import config, stitch
from sfzmaps.downloads import Item, run_tool
from sfzmaps.downloads.readme import LICENSE_TEXT, readme

ATTRIBUTION = "Based on sfzoning.com by Steven Bacio for GrowSF (https://growsf.org)"


def _copyright(series: str) -> str:
    text = f"{ATTRIBUTION}. Licensed CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/)."
    if series == "1948":
        text += f" Imagery: {config.RUMSEY_CREDIT}."
    return text


def export_citywide(item: Item, out: Path, updated: str, commit: str) -> None:
    """Write the citywide COG at `out` plus `<stem>.README.txt` and `<stem>.LICENSE.txt`."""
    vrt = stitch.build_vrt(item.series, paper=True)
    part = out.with_name(out.name + ".part")
    run_tool("gdal_translate", "-q", "-of", "COG", "-co", "COMPRESS=DEFLATE",
             "-co", "BIGTIFF=IF_SAFER", "-co", "OVERVIEWS=AUTO",
             "-mo", f"TIFFTAG_COPYRIGHT={_copyright(item.series)}",
             "-mo", f"TIFFTAG_IMAGEDESCRIPTION={item.series} edition, last updated {updated}",
             str(vrt), str(part))
    part.replace(out)
    stem = out.name.removesuffix(out.suffix)
    out.with_name(f"{stem}.README.txt").write_text(readme(item.series, "citywide", updated, commit))
    out.with_name(f"{stem}.LICENSE.txt").write_text(LICENSE_TEXT)


def sheet_entries(series: str) -> list[dict]:
    """One entry per placed sheet (every dir with warped_paper.tif), sorted by sheet id."""
    entries = []
    for tif in (config.WORK_DIR / series).glob("*/warped_paper.tif"):
        entries.append({
            "sheet": tif.parent.name,
            "name": f"{series}-sheet-{tif.parent.name}.tif",
            "md5": _md5(tif),
            "bytes": tif.stat().st_size,
            "sha256": stitch._sha256(tif),
        })
    return sorted(entries, key=lambda e: e["sheet"])


def _md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
