"""Per-year static review page (OUT_DIR/<series>/review.html) and a Range-capable file server."""
from __future__ import annotations

import functools
import http.server
import json
import re
import shutil
import subprocess
from pathlib import Path

from sfzmaps import config

EDITOR_URL = "http://127.0.0.1:8765/"
TEMPLATE = Path(__file__).parent / "templates" / "review.html"


def _read_manifest(series: str) -> dict:
    p = config.OUT_DIR / series / "manifest.json"
    if not p.exists():
        raise RuntimeError(f"no manifest for {series} at {p}; run `sfzmaps tiles --year {series}` first")
    return json.loads(p.read_text())


def sorted_sheets(series: str, sheets: list[dict]) -> list[dict]:
    """needs_review first (worst held-out p95 first), then the rest by sheet id."""
    def key(s):
        bad = s["status"] == "needs_review"
        p95 = s.get("holdout_p95_ft")
        return (0 if bad else 1, -(p95 if (bad and p95 is not None) else 0),
                len(s["sheet_id"]), s["sheet_id"])
    out = []
    for s in sorted(sheets, key=key):
        s = dict(s)
        s["editor_url"] = (f"{EDITOR_URL}?series={series}&sheet={s['sheet_id']}"
                           if s["status"] == "needs_review" else None)
        out.append(s)
    return out


def write_changes(series: str, sheets: list[dict]) -> Path:
    """Combine every sheet's changes.geojson (EPSG:2227) into one WGS84 file for the page."""
    from pyproj import Transformer
    from shapely.geometry import mapping, shape
    from shapely.ops import transform

    t = Transformer.from_crs(config.WORK_CRS, "EPSG:4326", always_xy=True).transform
    feats = []
    for s in sheets:
        p = config.WORK_DIR / series / s["sheet_id"] / "changes.geojson"
        try:
            fc = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        for f in fc.get("features", []):
            props = {"kind": f["properties"].get("kind"), "sheet_id": s["sheet_id"]}
            if "dist_ft" in f["properties"]:
                props["dist_ft"] = round(f["properties"]["dist_ft"], 1)
            feats.append({"type": "Feature", "properties": props,
                          "geometry": mapping(transform(t, shape(f["geometry"])))})
    out = config.OUT_DIR / series / "changes.wgs84.geojson"
    out.write_text(json.dumps({"type": "FeatureCollection", "features": feats}))
    return out


def ensure_blocks() -> Path | None:
    """OUT_DIR/ref/blocks.pmtiles from the DataSF parcels (ogr2ogr, GDAL >= 3.13); skip if fresh.
    Returns None when there is no reference data or no ogr2ogr."""
    src = config.REF_DIR / "parcels.parquet"
    out = config.OUT_DIR / "ref" / "blocks.pmtiles"
    if not src.exists():
        return None
    if out.exists() and out.stat().st_mtime >= src.stat().st_mtime:
        return out
    if not shutil.which("ogr2ogr"):
        return None
    import geopandas as gpd

    from sfzmaps.sources import datasf

    blocks = datasf.build_blocks(gpd.read_parquet(src))
    out.parent.mkdir(parents=True, exist_ok=True)
    gj = out.with_suffix(".geojson")
    blocks[["block_num", "geometry"]].to_crs("EPSG:4326").to_file(gj, driver="GeoJSON")
    tmp = out.with_name("blocks.tmp.pmtiles")
    tmp.unlink(missing_ok=True)
    try:
        subprocess.run(["ogr2ogr", "-f", "PMTiles", str(tmp), str(gj), "-nln", "blocks",
                        "-dsco", "MINZOOM=12", "-dsco", "MAXZOOM=17"], check=True)
        tmp.replace(out)
    finally:
        gj.unlink(missing_ok=True)
        tmp.unlink(missing_ok=True)
    return out


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def write(series: str) -> Path:
    m = _read_manifest(series)
    sheets = sorted_sheets(series, m["sheets"])
    write_changes(series, sheets)
    blocks = ensure_blocks()
    data = {"series": series, "credit": m.get("credit"),
            "pmtiles": m.get("pmtiles", f"{series}.pmtiles"),
            "blocks": "../ref/blocks.pmtiles" if blocks else None,
            "changes": "changes.wgs84.geojson", "sheets": sheets,
            "pass_median_ft": config.PASS_MEDIAN_FT, "pass_p95_ft": config.PASS_P95_FT,
            "pass_ref_ft_per_px": config.PASS_REF_FT_PER_PX}
    blob = json.dumps(data).replace("</", "<\\/")
    html = (TEMPLATE.read_text().replace("__TITLE__", f"SF zoning {series} review")
            .replace("__CREDIT__", _esc(m.get("credit") or "")).replace("__DATA__", blob))
    out = config.OUT_DIR / series / "review.html"
    out.write_text(html)
    return out


def is_stale(series: str) -> bool:
    out = config.OUT_DIR / series / "review.html"
    if not out.exists():
        return True
    t = out.stat().st_mtime
    deps = [config.OUT_DIR / series / "manifest.json", TEMPLATE, Path(__file__)]
    return any(d.exists() and d.stat().st_mtime > t for d in deps)


class RangeHandler(http.server.SimpleHTTPRequestHandler):
    """SimpleHTTPRequestHandler plus a single `Range: bytes=a-b` (206), which pmtiles needs."""

    def log_message(self, *a):
        pass

    def end_headers(self):
        self.send_header("Accept-Ranges", "bytes")
        super().end_headers()

    def send_head(self):
        rng = self.headers.get("Range")
        path = Path(self.translate_path(self.path))
        if not rng or not path.is_file():
            return super().send_head()
        m = re.fullmatch(r"bytes=(\d*)-(\d*)", rng.strip())
        size = path.stat().st_size
        if not m or not (m.group(1) or m.group(2)):
            self.send_error(416, "bad range")
            return None
        if m.group(1):
            a = int(m.group(1))
            b = min(int(m.group(2)), size - 1) if m.group(2) else size - 1
        else:
            a, b = max(size - int(m.group(2)), 0), size - 1
        if a >= size or a > b:
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return None
        f = open(path, "rb")
        f.seek(a)
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(str(path)))
        self.send_header("Content-Range", f"bytes {a}-{b}/{size}")
        self.send_header("Content-Length", str(b - a + 1))
        self.end_headers()
        return _Limited(f, b - a + 1)


class _Limited:
    """File wrapper exposing at most n bytes to copyfile()."""

    def __init__(self, f, n):
        self.f, self.n = f, n

    def read(self, k=-1):
        k = self.n if k is None or k < 0 else min(k, self.n)
        b = self.f.read(k)
        self.n -= len(b)
        return b

    def close(self):
        self.f.close()


def make_server(directory: Path, port: int = 8000, host: str = "127.0.0.1") -> http.server.ThreadingHTTPServer:
    h = functools.partial(RangeHandler, directory=str(directory))
    return http.server.ThreadingHTTPServer((host, port), h)
