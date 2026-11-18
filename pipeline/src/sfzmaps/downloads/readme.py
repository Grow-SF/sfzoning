"""README.txt and LICENSE.txt text packed into every download."""
import textwrap
from pathlib import Path

from sfzmaps import config

# Beside the code, not under templates/: that directory is a DVC code dependency of the stages.
TEMPLATE = Path(__file__).with_name("readme-template.txt")

LICENSE_TEXT = """Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)

This data is licensed CC BY-NC-SA 4.0: https://creativecommons.org/licenses/by-nc-sa/4.0/
You may share and adapt it with attribution, for non-commercial use, under the same license.
Credit it as:

    Based on sfzoning.com by Steven Bacio for GrowSF (https://growsf.org), licensed CC BY-NC-SA 4.0.

The license covers this project's contribution (the reading, alignment and processing), not the
underlying maps or the DataSF parcels, which keep their own terms. See https://sfzoning.com.
"""

# (name in the data, name in .shp files, meaning); shapefile names are limited to 10 characters.
LOT_FIELDS = [
    ("blklot", "blklot", "DataSF parcel id: block and lot"),
    ("code", "code", "zoning district code read from the map"),
    ("height_code", "height", "height district code (where the period has a height map)"),
    ("category", "category", "category used on sfzoning.com"),
    ("confidence", "confidence", "0 to 1, how sure the reading is"),
    ("source_sheet", "src_sheet", "sheet the label was read from"),
    ("note", "note", "why a park lot shows as public, if so"),
]
DISTRICT_FIELDS = [
    ("code", "code", "zoning district code read from the map"),
    ("category", "category", "category used on sfzoning.com"),
]
# The same fields for the "today" edition, which no scan feeds.
_TODAY_MEANING = {
    "code": "zoning district code from DataSF Zoning Districts",
    "height_code": "empty (today's height districts are not included)",
    "confidence": "always 1 (see below)",
    "source_sheet": 'always "datasf": no map sheet, the code is from DataSF',
}

_CRS = {
    layer: f"GeoJSON and Shapefile are in WGS 84 (EPSG:4326). The GeoPackage has the layer "
           f"`{layer}` in WGS 84 (EPSG:4326) and the same {layer} in California State Plane III, "
           f"feet (EPSG:2227) as `{layer}_ca_spcs`."
    for layer in ("lots", "districts")
}
# How a scan became the image (warp.py and stitch.py), for the two scan layers.
_SCAN = ("Each sheet was warped onto that grid with GDAL (gdalwarp, thin plate spline, cubic "
         "resampling) from control points matched to today's street corners. Sheets scanned in "
         "1-bit black and white are tinted to the paper colour, and printed border rules and "
         "binding shadows are erased. Nothing is vectorized or drawn on the scans.")
_WHAT = {
    "lots": "One polygon per San Francisco lot (today's DataSF parcels) with the zoning district "
            "the {edition} map assigns it. " + _CRS["lots"],
    "districts": "The {edition} zoning districts: the lots merged by district code. " + _CRS["districts"],
    "citywide": "The {edition} zoning map sheets stitched into one citywide georeferenced image "
                "(Cloud Optimized GeoTIFF, California State Plane III, feet, EPSG:2227). " + _SCAN +
                " In the stitch, plain paper fills the gaps between sheets, and sheets coarser than "
                "the finest sheet are resampled up to its resolution.",
    "sheets": "The aligned sheets of the {edition} zoning map, one GeoTIFF per sheet, each at its own "
              "resolution (California State Plane III, feet, EPSG:2227). " + _SCAN,
}
_WHAT_TODAY = {
    "lots": "One polygon per San Francisco lot (today's DataSF parcels) with the zoning district "
            "it is in today, from DataSF Zoning Districts. " + _CRS["lots"],
    "districts": "Today's zoning districts: the lots merged by district code. " + _CRS["districts"],
}

_HOW = ("The zoning maps were read from the scans, one district label per area, and each label was "
        "assigned to today's lots. The boundaries in the lot and district files are modern lot lines "
        "from the DataSF parcels layer. They were never traced from the scan, so a district edge is "
        "only as precise as the lot it falls on. Where a street has moved since the map was drawn, "
        "the scan keeps the historic position and the data follows today's lot lines.")
_HOW_TODAY = ("Today's zoning is not read from a scan. It comes from DataSF Zoning Districts "
              "(dataset 3i4a-hu95): each of today's lots gets the code of the zoning district that "
              "covers the largest share of its area. The boundaries in the lot and district files "
              "are modern lot lines from the DataSF parcels layer, so a district edge follows the "
              "lot it falls on.")
_CONFIDENCE = ("A number from 0 to 1 for how sure the reading is that the lot belongs to the "
               "district shown. Lots set by a person get 1: those whose district was set by an "
               "override (inputs/<series>/overrides.csv in the source repository), and those whose "
               "label a person added by hand (inputs/<series>/label_fixes.csv). Lots read from a "
               "sheet flagged for review are capped at 0.6. 0 means no district could be assigned "
               "to the lot.")
_CONFIDENCE_TODAY = ("Always 1: the code is taken from DataSF's zoning layer, not read from a map. "
                     "A lot with an empty code has no zoning district over it in that layer.")
_TODAY_CREDIT = ('Today\'s zoning: DataSF, "Zoning Districts" (dataset 3i4a-hu95), City and County '
                 "of San Francisco (https://datasf.org).")
_RUMSEY = (f"Imagery: {config.RUMSEY_CREDIT}, licensed CC BY-NC-SA 3.0 "
           "(https://creativecommons.org/licenses/by-nc-sa/3.0/).")


def _wrap(text: str) -> str:
    return textwrap.fill(text, 78, break_long_words=False, break_on_hyphens=False)


def _fields(series: str, layer: str) -> str:
    fields = {"lots": LOT_FIELDS, "districts": DISTRICT_FIELDS}.get(layer)
    if not fields:
        return ""
    if series == "today":
        fields = [(n, sh, _TODAY_MEANING.get(n, d)) for n, sh, d in fields]
    lines = ["FIELDS", f"{'field':<14}{'in .shp':<12}meaning"]
    lines += [f"{n:<14}{s:<12}{d}" for n, s, d in fields]
    return "\n".join(lines) + "\n\n"


def readme(series: str, layer: str, updated: str, commit: str) -> str:
    text = TEMPLATE.read_text().split("\n", 1)[1]  # drop the "draft copy" marker line
    today = series == "today"
    credits = "".join(_wrap(c) + "\n" for c, on in ((_RUMSEY, series == "1948"),
                                                    (_TODAY_CREDIT, today)) if on)
    what = (_WHAT_TODAY if today else _WHAT)[layer]
    for k, v in {
        "TITLE": "San Francisco zoning today" if today else "San Francisco historic zoning",
        "EDITION": series,
        "LAYER_NAME": layer,
        "WHAT": _wrap(what.format(edition=f"{series} edition")),
        "FIELDS": _fields(series, layer),
        "HOW": _wrap(_HOW_TODAY if today else _HOW),
        "CONFIDENCE": _wrap(_CONFIDENCE_TODAY if today else _CONFIDENCE),
        "UPDATED": updated,
        "COMMIT": commit,
        "LICENSE": LICENSE_TEXT.strip(),
        "CREDITS": credits,
    }.items():
        text = text.replace("{{" + k + "}}", v)
    return text.rstrip("\n") + "\n"
