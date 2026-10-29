"""Reader for the owner's QGIS georeferencer `.points` files (the answer key)."""
import csv
from pathlib import Path

from sfzmaps.transform import GCP


def read_points(path: Path, *, scale: float = 1.0) -> list[GCP]:
    """GCPs from a QGIS CSV (mapX,mapY,pixelX,pixelY,enable; optional leading `#CRS:` line).

    QGIS stores pixelY negative (y up from the top-left), so py = -pixelY. `scale`
    multiplies px and py, to move the file's raster into our render's pixel space.
    Source is "answer_key"; enable=0 gives enabled=False."""
    lines = [ln for ln in Path(path).read_text().splitlines() if ln.strip() and not ln.startswith("#")]
    out = []
    for r in csv.DictReader(lines):
        out.append(GCP(px=float(r["pixelX"]) * scale, py=-float(r["pixelY"]) * scale,
                       wx=float(r["mapX"]), wy=float(r["mapY"]), source="answer_key",
                       enabled=str(r.get("enable", "1")).strip() not in ("0", "false", "False")))
    return out
