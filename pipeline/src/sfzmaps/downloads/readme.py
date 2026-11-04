"""README.txt and LICENSE.txt text packed into every download."""
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
    ("height_code", "height_cd", "height district code (where the period has a height map)"),
    ("category", "category", "category used on sfzoning.com"),
    ("confidence", "confidence", "0 to 1, how sure the reading is"),
    ("source_sheet", "src_sheet", "sheet the label was read from"),
    ("note", "note", "why a park lot shows as public, if so"),
]
DISTRICT_FIELDS = [
    ("code", "code", "zoning district code read from the map"),
    ("category", "category", "category used on sfzoning.com"),
]

_WHAT = {
    "lots": "One polygon per San Francisco lot (today's DataSF parcels) with the zoning district\n"
            "the {edition} map assigns it. GeoJSON is in WGS 84 (EPSG:4326); GeoPackage and\n"
            "Shapefile are in California State Plane III, feet (EPSG:2227).",
    "districts": "The {edition} zoning districts: the lots merged by district code. GeoJSON is\n"
                 "in WGS 84 (EPSG:4326); GeoPackage and Shapefile are in California State Plane III,\n"
                 "feet (EPSG:2227).",
    "citywide": "The {edition} zoning map sheets stitched into one citywide georeferenced image\n"
                "(Cloud Optimized GeoTIFF, California State Plane III, feet, EPSG:2227). The scans\n"
                "are shown as scanned: not resampled to modern streets, not vectorized, nothing\n"
                "drawn on them.",
}


def _fields(layer: str) -> str:
    fields = {"lots": LOT_FIELDS, "districts": DISTRICT_FIELDS}.get(layer)
    if not fields:
        return ""
    lines = ["FIELDS", f"{'field':<14}{'in .shp':<12}meaning"]
    lines += [f"{n:<14}{s:<12}{d}" for n, s, d in fields]
    return "\n".join(lines) + "\n\n"


def readme(series: str, layer: str, updated: str, commit: str) -> str:
    text = TEMPLATE.read_text().split("\n", 1)[1]  # drop the "draft copy" marker line
    rumsey = (f"Imagery: {config.RUMSEY_CREDIT}, licensed CC BY-NC-SA 3.0\n"
              "(https://creativecommons.org/licenses/by-nc-sa/3.0/).\n") if series == "1948" else ""
    for k, v in {
        "EDITION": series,
        "LAYER_NAME": layer,
        "WHAT": _WHAT[layer].format(edition=f"{series} edition"),
        "FIELDS": _fields(layer),
        "UPDATED": updated,
        "COMMIT": commit,
        "LICENSE": LICENSE_TEXT.strip(),
        "RUMSEY": rumsey,
    }.items():
        text = text.replace("{{" + k + "}}", v)
    return text.rstrip("\n") + "\n"
