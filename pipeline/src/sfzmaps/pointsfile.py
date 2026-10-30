"""Reader for the owner's QGIS georeferencer `.points` files (the answer key)."""
import csv
import re
from pathlib import Path

from sfzmaps.transform import GCP

# QGIS georeferencer pixel coordinates put (0, 0) at the top-left CORNER of the top-left
# pixel; ours (OpenCV / page.png arrays, SheetTransform) put it at that pixel's CENTER.
PIXEL_CENTER_SHIFT = 0.5


def _check_crs(line: str, path: Path) -> None:
    """A `#CRS:` line must name EPSG:2227 (as "EPSG:2227" or a WKT ID["EPSG",2227])."""
    text = line[1:].split(":", 1)[1] if ":" in line else line[1:]
    norm = re.sub(r"\s+", "", text).upper()
    if "EPSG:2227" in norm or 'ID["EPSG",2227]' in norm or "AUTHORITY[\"EPSG\",\"2227\"]" in norm:
        return
    raise ValueError(f"{path}: answer key CRS is not EPSG:2227: {text.strip()[:80]}")


def read_points(path: Path, *, scale: float = 1.0) -> list[GCP]:
    """GCPs from a QGIS CSV (mapX,mapY,pixelX,pixelY,enable; optional leading `#CRS:` line,
    which must be EPSG:2227, else ValueError).

    QGIS stores pixelY negative (y up from the top-left), so py = -pixelY. `scale`
    multiplies px and py, to move the file's raster into our render's pixel space; then
    PIXEL_CENTER_SHIFT (0.5 px) is subtracted, converting QGIS's corner-based pixel
    coordinates to our pixel-center ones. Source is "answer_key"; enable=0 gives
    enabled=False."""
    raw = Path(path).read_text().splitlines()
    for ln in raw:
        if ln.startswith("#CRS"):
            _check_crs(ln, Path(path))
    lines = [ln for ln in raw if ln.strip() and not ln.startswith("#")]
    out = []
    for r in csv.DictReader(lines):
        out.append(GCP(px=float(r["pixelX"]) * scale - PIXEL_CENTER_SHIFT,
                       py=-float(r["pixelY"]) * scale - PIXEL_CENTER_SHIFT,
                       wx=float(r["mapX"]), wy=float(r["mapY"]), source="answer_key",
                       enabled=str(r.get("enable", "1")).strip() not in ("0", "false", "False")))
    return out
