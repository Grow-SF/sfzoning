"""Default max_zoom is 18: the scans are about 0.63 m/px and z18 is 0.60 m/px at SF's latitude,
so z18 is native resolution; z19 only upsamples 2x for about 44% of the bytes (1948: 140 of 316
MB).  The site can overzoom z18 client-side.  Pass --max-zoom 19 to build it anyway.

VRT -> EPSG:3857 COG -> MBTiles (WEBP) -> overviews to min_zoom -> PMTiles."""
from __future__ import annotations

import math
import shutil
import sqlite3
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

WEB_ORIGIN_RES = 2 * math.pi * 6378137 / 256   # metres per pixel at zoom 0 (256 px tiles)


@dataclass
class TileReport:
    bytes_total: int
    bytes_by_zoom: dict[int, int]


def require_tools() -> None:
    missing = [t for t in ("gdalwarp", "gdal_translate", "gdaladdo", "pmtiles") if shutil.which(t) is None]
    if missing:
        hint = {"pmtiles": "install it with `brew install pmtiles` (https://github.com/protomaps/go-pmtiles)"}
        raise RuntimeError("missing tool(s): " + "; ".join(
            f"{t}: {hint.get(t, 'install GDAL (`brew install gdal`)')}" for t in missing))


def report_from_mbtiles(path: Path) -> TileReport:
    con = sqlite3.connect(path)
    try:
        rows = con.execute("select zoom_level, sum(length(tile_data)) from tiles group by zoom_level").fetchall()
    finally:
        con.close()
    by = {int(z): int(b) for z, b in rows}
    return TileReport(sum(by.values()), by)


def set_metadata(path: Path, name: str, attribution: str | None, min_zoom: int) -> None:
    """name/description/attribution and a center at the middle of the bounds. Without a
    center, pmtiles convert computes a wrong one (longitude 92 for SF)."""
    con = sqlite3.connect(path)
    try:
        b = con.execute("select value from metadata where name='bounds'").fetchone()
        rows = {"name": name, "description": name}
        if attribution:
            rows["attribution"] = attribution
        if b:
            w, s_, e, n = (float(v) for v in b[0].split(","))
            rows["center"] = f"{(w + e) / 2:.6f},{(s_ + n) / 2:.6f},{min_zoom}"
        for k, v in rows.items():  # no UNIQUE constraint to rely on in GDAL's table
            con.execute("delete from metadata where name=?", (k,))
            con.execute("insert into metadata(name, value) values (?, ?)", (k, v))
        con.commit()
    finally:
        con.close()


def _run(cmd: list[str]) -> None:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed: {r.stderr.strip() or r.stdout.strip()}")


def make_pmtiles(vrt: Path, out: Path, *, min_zoom: int = 11, max_zoom: int = 18,
                 tile_format: str = "WEBP", name: str = "sfzmaps",
                 attribution: str | None = None, keep_mbtiles: Path | None = None,
                 quality: int = 85) -> TileReport:
    require_tools()
    out.parent.mkdir(parents=True, exist_ok=True)
    res = WEB_ORIGIN_RES / 2 ** max_zoom
    with tempfile.TemporaryDirectory(prefix="sfzmaps-tiles-", dir=out.parent) as tmp:
        tmp = Path(tmp)
        cog, mb = tmp / "web.tif", tmp / "tiles.mbtiles"
        _run(["gdalwarp", "--config", "GDAL_NUM_THREADS", "ALL_CPUS", "-multi", "-wo", "NUM_THREADS=ALL_CPUS",
              "-t_srs", "EPSG:3857", "-tr", str(res), str(res), "-tap", "-r", "bilinear",
              "-of", "COG", "-co", "COMPRESS=DEFLATE", "-co", "BIGTIFF=YES", "-co", "NUM_THREADS=ALL_CPUS",
              "-overwrite", str(vrt), str(cog)])
        _run(["gdal_translate", "--config", "GDAL_NUM_THREADS", "ALL_CPUS", "-of", "MBTILES", "-co", f"TILE_FORMAT={tile_format}", "-co", f"QUALITY={quality}",
              "-co", "ZOOM_LEVEL_STRATEGY=AUTO", str(cog), str(mb)])
        top = max(report_from_mbtiles(mb).bytes_by_zoom)
        if top > min_zoom:
            _run(["gdaladdo", "--config", "GDAL_NUM_THREADS", "ALL_CPUS", "-r", "bilinear", str(mb), *[str(2 ** i) for i in range(1, top - min_zoom + 1)]])
        report = report_from_mbtiles(mb)
        set_metadata(mb, name, attribution, min_zoom)
        # convert next to `out`, then replace it: a failed convert keeps the old PMTiles
        part = tmp / out.name
        _run(["pmtiles", "convert", str(mb), str(part)])
        part.replace(out)
        if keep_mbtiles:
            shutil.copy(mb, keep_mbtiles)
    return report
