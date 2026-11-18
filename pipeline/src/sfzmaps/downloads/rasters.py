"""Citywide image (a Cloud Optimized GeoTIFF).

The image is the same paper stitch the site shows (`stitch.build_vrt(paper=True)`) copied into a
COG: the warped sheets (warp.py) at the finest sheet's resolution, nothing vectorized or drawn.
"""
from __future__ import annotations

from pathlib import Path

from sfzmaps import config, stitch
from sfzmaps.downloads import Item, run_tool

ATTRIBUTION = "Based on sfzoning.com by Steven Bacio for GrowSF (https://growsf.org)"


def _copyright(series: str) -> str:
    text = f"{ATTRIBUTION}. Licensed CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/)."
    if series == "1948":
        text += f" Imagery: {config.RUMSEY_CREDIT}."
    return text


def write_image(item: Item, out: Path) -> None:
    """Write the citywide COG at `out`. Same data in, same bytes out: no date or build in it."""
    vrt = stitch.build_vrt(item.series, paper=True)
    part = out.with_name(out.name + ".part")
    try:
        run_tool("gdal_translate", "-q", "-of", "COG", "-co", "COMPRESS=DEFLATE",
                 "-co", "BIGTIFF=IF_SAFER", "-co", "OVERVIEWS=AUTO", "-co", "OVERVIEW_RESAMPLING=AVERAGE",
                 "-mo", f"TIFFTAG_COPYRIGHT={_copyright(item.series)}",
                 "-mo", f"TIFFTAG_IMAGEDESCRIPTION={item.series} edition",
                 str(vrt), str(part))
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    part.replace(out)
