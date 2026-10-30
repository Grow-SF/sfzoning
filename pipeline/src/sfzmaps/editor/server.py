"""Control-point editor API (FastAPI). Launched by `sfzmaps edit`, bound to 127.0.0.1 only.

Endpoints (all under /api):
  GET  /queue                              sheets, needs_review first
  GET  /sheet/{series}/{sheet_id}          gcps (+residual_ft), matches, score, image info
  GET  /sheet/{series}/{sheet_id}/image/{z}/{x}/{y}.png   Deep Zoom-style scan tiles
  POST /sheet/{series}/{sheet_id}/preview  fit in memory, score, low-res warped PNG
  GET  /sheet/{series}/{sheet_id}/preview.png
  GET  /sheet/{series}/{sheet_id}/blocks.geojson?layer=blocks|streets   modern blocks near the sheet (WGS84)
  POST /sheet/{series}/{sheet_id}/save     write review.json, rerun the sheet, rebuild series

Scores are the pipeline's (sfzmaps.score): a sheet passes on held-out median <= 5 ft AND
held-out p95 <= 12 ft AND not misplaced; the editor shows holdout_p50_ft / holdout_p95_ft /
reason / passed, never a share-within-tolerance.

Hand-placed points are fitted with config.MANUAL_TPS_SMOOTHING, exactly as the runner does for
review.json, and previews are scored by runner.score_review (the runner's own review
scoring), so a preview equals the saved result. Matches inside change_areas are excluded.

Tile scheme (Deep Zoom): image of w x h px, TILE_SIZE 256, max zoom z_max = ceil(log2(max(w,h))).
Level z is the image downscaled by 2**(z_max - z) (dimensions ceil(w / 2**(z_max-z))); tile
(x, y) at level z covers level pixels [x*256, (x+1)*256) x [y*256, (y+1)*256), clipped (edge
tiles are smaller, not padded). Tiles are created on first request and cached under
work/<series>/<sheet_id>/dz/<z>/<x>_<y>.png. The sheet payload gives image_url as a template
with {z}/{x}/{y}, plus image_tile_size and image_max_zoom.
"""
from __future__ import annotations

import functools
import json
import logging
import math
import re
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image
from pydantic import BaseModel

from sfzmaps import config, runner
from sfzmaps import score as score_mod
from sfzmaps.transform import GCP, SheetTransform

log = logging.getLogger(__name__)

TILE_SIZE = 256
PREVIEW_MAX_PX = 2048
HOST = "127.0.0.1"
PORT = 8765

app = FastAPI(title="sfzmaps control-point editor")
_filter: dict = {"year": None, "sheet": None}
_save_lock = threading.Lock()
_ref_cache: dict = {}


def configure(year: str | None = None, sheet: str | None = None) -> None:
    """Restrict the queue to a series and/or sheet (the CLI's --year / --sheet)."""
    _filter["year"], _filter["sheet"] = year, sheet


# -- collaborators (replaced in tests / by the demo) -------------------------------------

def _get_ref():
    """The modern reference, loaded once; None when it is unavailable (previews then skip the
    label-agreement check)."""
    if "ref" not in _ref_cache:
        try:
            from sfzmaps.sources import datasf
            _ref_cache["ref"] = datasf.load_reference()
        except Exception as e:  # noqa: BLE001 - editor still works without it
            log.warning("reference data unavailable (%s: %s)", type(e).__name__, e)
            _ref_cache["ref"] = None
    return _ref_cache["ref"]


def set_reference(ref) -> None:
    _ref_cache["ref"] = ref


_pages: dict = {}


def register_page(page) -> None:
    _pages[(page.series, page.sheet_id)] = page


def _find_page(series: str, sheet_id: str):
    if (series, sheet_id) in _pages:
        return _pages[(series, sheet_id)]
    from sfzmaps import pages
    for p in pages.discover():
        if p.series == series and p.sheet_id == sheet_id:
            return p
    return None


def _git_email() -> str:
    try:
        out = subprocess.run(["git", "config", "user.email"], capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _rebuild_series(series: str) -> None:
    """Rebuild the series' VRT, PMTiles and manifest (Task 14 code, same steps as
    `sfzmaps tiles`). A failing step is logged and skipped so a save never fails because of
    the rebuild; the manifest is still rewritten, keeping the previous tile_report when the
    tile build did not run."""
    from sfzmaps import manifest, reviewpage, stitch, tiles

    def step(label: str, fn):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            log.warning("rebuild %s: %s failed (%s: %s); skipping", series, label, type(e).__name__, e)
            return None

    scores = stitch.load_scores(series)
    vrt = step("stitch", lambda: stitch.build_vrt(series, scores))
    report = None
    if vrt is not None:
        out = config.OUT_DIR / series / f"{series}.pmtiles"
        report = step("tiles", lambda: tiles.make_pmtiles(
            vrt, out, name=f"SF zoning {series}",
            attribution=config.RUMSEY_CREDIT if series == "1948" else None))
    step("manifest", lambda: manifest.write(series, scores, report))
    step("review page", lambda: reviewpage.write(series))


_rb_guard = threading.Lock()
_rb_state: dict[str, dict] = {}


def _schedule_rebuild(series: str) -> None:
    """Serialize and coalesce rebuilds of one series (overlapping gdal/pmtiles runs would
    corrupt the VRT and PMTiles). If a rebuild is running, mark the series dirty and return;
    the running caller then performs exactly one trailing rebuild."""
    with _rb_guard:
        st = _rb_state.setdefault(series, {"running": False, "dirty": False})
        if st["running"]:
            st["dirty"] = True
            return
        st["running"] = True
    while True:
        try:
            _rebuild_series(series)
        except Exception:  # noqa: BLE001 - a background rebuild must never raise
            log.exception("rebuild of series %s failed", series)
        with _rb_guard:
            if st["dirty"]:
                st["dirty"] = False
                continue
            st["running"] = False
            return


# -- helpers ---------------------------------------------------------------------------

def _clean(o):
    """JSON-safe: NaN/inf -> None, numpy scalars -> python."""
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return float(o) if math.isfinite(o) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def _score_dict(s: score_mod.SheetScore) -> dict:
    from dataclasses import asdict
    return _clean(asdict(s))


_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _sheet_dir(series: str, sheet_id: str) -> Path:
    if not (_ID_RE.fullmatch(series) and _ID_RE.fullmatch(sheet_id)):
        raise HTTPException(404, "no such sheet")
    d = config.WORK_DIR / series / sheet_id
    if not d.resolve().is_relative_to(config.WORK_DIR.resolve()):
        raise HTTPException(404, "no such sheet")
    if not (d / "page.png").exists():
        raise HTTPException(404, f"no sheet {series}/{sheet_id}")
    return d


def _read_json(p: Path):
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


class GCPIn(BaseModel):
    px: float
    py: float
    wx: float
    wy: float
    source: Literal["anchor", "auto", "manual", "answer_key"] = "manual"
    enabled: bool = True


class PreviewBody(BaseModel):
    gcps: list[GCPIn]
    change_areas: dict | None = None


class SaveBody(PreviewBody):
    status: Literal["edited", "accepted"]
    note: str = ""
    reviewer: str | None = None


def _transform_of(gcps: list[GCPIn]) -> SheetTransform:
    """The transform the runner will build from review.json (MANUAL_TPS_SMOOTHING)."""
    t = SheetTransform([GCP(**g.model_dump()) for g in gcps], config.MANUAL_TPS_SMOOTHING)
    try:
        t.fit()
    except (ValueError, np.linalg.LinAlgError) as e:
        raise HTTPException(400, f"cannot fit these control points: {e}")
    return t


def _residuals(t: SheetTransform) -> list:
    """Residual (ft) per GCP in input order; None for disabled GCPs."""
    res = iter(t.residuals_ft())
    return [float(next(res)) if g.enabled else None for g in t.gcps]


def _bounds(t: SheetTransform, size: tuple[int, int]) -> list[float] | None:
    w, h = size
    xs, ys = np.linspace(0, w - 1, 9), np.linspace(0, h - 1, 9)
    P = np.array([[x, y] for y in ys for x in xs])
    try:
        W = t.to_world(P)
    except (ValueError, np.linalg.LinAlgError):
        return None
    return [float(W[:, 0].min()), float(W[:, 1].min()), float(W[:, 0].max()), float(W[:, 1].max())]


def _bounds_wgs84(b: list[float] | None) -> list[float] | None:
    if b is None:
        return None
    try:
        from pyproj import Transformer
        tr = Transformer.from_crs(config.WORK_CRS, "EPSG:4326", always_xy=True)
        x0, y0 = tr.transform(b[0], b[1])
        x1, y1 = tr.transform(b[2], b[3])
        return [x0, y0, x1, y1]
    except Exception:  # noqa: BLE001
        return None


def _png_size(png: Path) -> tuple[int, int]:
    with Image.open(png) as im:
        return im.size


# -- queue -----------------------------------------------------------------------------

def _status(d: Path, score: score_mod.SheetScore | None) -> str:
    rv = runner._load_review(d)
    if rv is not None:
        return rv["status"]
    if score is None:
        return "unscored"
    return "passed" if score.passed else "needs_review"


def _natural(s: str):
    import re
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


@app.get("/api/queue")
def queue() -> list[dict]:
    rows = []
    root = config.WORK_DIR
    for sd in sorted(x for x in root.iterdir() if x.is_dir()) if root.is_dir() else []:
        if _filter["year"] and sd.name != _filter["year"]:
            continue
        for d in sorted(x for x in sd.iterdir() if x.is_dir()):
            if _filter["sheet"] and d.name != _filter["sheet"]:
                continue
            sj = d / "score.json"
            if not (d / "page.png").exists() and not sj.exists():
                continue
            raw = _read_json(sj) if sj.exists() else None
            try:
                sc = runner._score_load(raw) if raw is not None else None
            except KeyError:
                sc = None
            rows.append(_clean({
                "series": sd.name, "sheet_id": d.name, "status": _status(d, sc),
                "holdout_p50_ft": None if sc is None else sc.holdout_p50_ft,
                "holdout_p95_ft": None if sc is None else sc.holdout_p95_ft,
                "reason": None if sc is None else sc.reason,
                "passed": bool(sc and sc.passed),
                "answer_key_fallback": bool(raw and raw.get("answer_key_fallback")),
            }))
    order = {"needs_review": 0, "unscored": 1, "edited": 2, "accepted": 3, "passed": 4}
    rows.sort(key=lambda r: (order.get(r["status"], 5), _natural(r["series"]), _natural(r["sheet_id"])))
    return rows


# -- sheet -----------------------------------------------------------------------------

def _better(a: score_mod.SheetScore, b: score_mod.SheetScore | None) -> bool:
    """a scores better than b: lower held-out median (a finite beats NaN)."""
    if not math.isfinite(a.holdout_p50_ft):
        return False
    return b is None or not math.isfinite(b.holdout_p50_ft) or a.holdout_p50_ft < b.holdout_p50_ft


def _tile_info(d: Path, series: str, sheet_id: str) -> dict:
    w, h = _png_size(d / "page.png")
    return {"image_url": f"/api/sheet/{series}/{sheet_id}/image/{{z}}/{{x}}/{{y}}.png",
            "image_size": {"width": w, "height": h}, "image_tile_size": TILE_SIZE,
            "image_max_zoom": _max_zoom(w, h)}


@app.get("/api/sheet/{series}/{sheet_id}")
def sheet(series: str, sheet_id: str) -> JSONResponse:
    d = _sheet_dir(series, sheet_id)
    ref = _get_ref()
    tj, rfj, lj = d / "transform.json", d / "refine.json", d / "labels.json"
    review = runner._load_review(d)
    tdata = _read_json(tj) if tj.exists() else None
    sj = d / "score.json"
    raw_score = _read_json(sj) if sj.exists() else None
    score = runner._score_load(raw_score) if raw_score is not None else None

    if review is not None:
        gcps, origin, smoothing = review["gcps"], "review", config.MANUAL_TPS_SMOOTHING
    elif tdata is not None:
        gcps, origin = tdata["gcps"], tdata.get("origin", "auto")
        smoothing = tdata["smoothing"] if origin == "auto" else config.MANUAL_TPS_SMOOTHING
    else:
        gcps, origin, smoothing = [], None, config.MANUAL_TPS_SMOOTHING

    # The owner's answer key, when there is one: offered always, adopted when it scores better.
    key = None
    page = None
    try:
        page = _find_page(series, sheet_id)
    except Exception:  # noqa: BLE001 - no raw data on this machine
        page = None
    if page is not None and runner.answer_key_path(page) is not None:
        try:
            kg = runner._answer_key_gcps(page, d / "page.png")
        except Exception:  # noqa: BLE001
            kg = None
        if kg is not None:
            kt = SheetTransform(kg, config.MANUAL_TPS_SMOOTHING)
            ks = runner.score_review(kt, rfj, ref, None, lj) if rfj.exists() else None
            key = {"gcps": [_gcp_json(g, r) for g, r in zip(kg, _res_all(kt))],
                   "score": None if ks is None else _score_dict(ks)}
            adopt = review is None and (tdata is None or (ks is not None and _better(ks, score)))
            if adopt:
                gcps, origin, smoothing = [vars_of(g) for g in kg], "answer_key", config.MANUAL_TPS_SMOOTHING
                score = ks if ks is not None else score

    t = None
    resid: list = []
    if gcps:
        try:
            t = SheetTransform([GCP(**g) for g in gcps], smoothing).fit()
            resid = _residuals(t)
        except (ValueError, np.linalg.LinAlgError):
            t, resid = None, [None] * len(gcps)
    info = _tile_info(d, series, sheet_id)
    w, h = info["image_size"]["width"], info["image_size"]["height"]
    wb = _bounds(t, (w, h)) if t is not None else None
    refine = _read_json(rfj) if rfj.exists() else None
    cj = d / "changes.geojson"
    change_areas = review.get("change_areas") if review else (_read_json(cj) if cj.exists() else None)
    status = review["status"] if review else ("unscored" if score is None else ("passed" if score.passed else "needs_review"))
    return JSONResponse(_clean({
        "series": series, "sheet_id": sheet_id, "status": status, "origin": origin,
        "gcps": [{**g, "residual_ft": r} for g, r in zip(gcps, resid or [None] * len(gcps))],
        "matches": refine["matches"] if refine else [],
        "score": None if score is None else _score_dict(score),
        "answer_key": key,
        "change_areas": change_areas,
        "smoothing": smoothing,
        "world_bounds": wb, "world_bounds_wgs84": _bounds_wgs84(wb),
        "note": review.get("note", "") if review else "",
        "reviewer": review.get("reviewer") if review else None,
        "credit": config.RUMSEY_CREDIT if series == "1948" else None,
        **info,
    }))


def vars_of(g: GCP) -> dict:
    from dataclasses import asdict
    return asdict(g)


def _gcp_json(g: GCP, r) -> dict:
    return {**vars_of(g), "residual_ft": r}


def _res_all(t: SheetTransform) -> list:
    return [float(x) for x in t.residuals_ft()]


# -- reference blocks near the sheet -----------------------------------------------------

BLOCKS_PAD_FT = 800.0
BLOCKS_MAX_FEATURES = 4000


def _sheet_extent(d: Path) -> list[float] | None:
    """World (EPSG:2227 ft) extent of the sheet's control points (review, else transform, else
    matches), or None when it has none."""
    pts: list[tuple[float, float]] = []
    review = runner._load_review(d)
    tdata = _read_json(d / "transform.json") if (d / "transform.json").exists() else None
    rdata = _read_json(d / "refine.json") if (d / "refine.json").exists() else None
    src = (review or tdata or {}).get("gcps") or (rdata or {}).get("matches") or []
    for g in src:
        wx, wy = g.get("wx"), g.get("wy")
        if wx is not None and wy is not None and math.isfinite(wx) and math.isfinite(wy):
            pts.append((wx, wy))
    if not pts:
        return None
    a = np.array(pts)
    return [float(a[:, 0].min()), float(a[:, 1].min()), float(a[:, 0].max()), float(a[:, 1].max())]


@app.get("/api/sheet/{series}/{sheet_id}/blocks.geojson")
def blocks_geojson(series: str, sheet_id: str, layer: Literal["blocks", "streets"] = "blocks") -> JSONResponse:
    """Modern DataSF block outlines (or street centerlines) near the sheet, as WGS84 GeoJSON
    for the editor's map pane. Used because the blocks PMTiles is not served. Empty when the
    reference or the sheet's extent is unavailable."""
    d = _sheet_dir(series, sheet_id)
    empty = {"type": "FeatureCollection", "features": []}
    ref, ext = _get_ref(), _sheet_extent(d)
    gdf = None if ref is None else (ref.blocks if layer == "blocks" else ref.streets)
    if gdf is None or ext is None or len(gdf) == 0:
        return JSONResponse(empty)
    from shapely.geometry import box
    pad = BLOCKS_PAD_FT
    sel = gdf[gdf.intersects(box(ext[0] - pad, ext[1] - pad, ext[2] + pad, ext[3] + pad))]
    sel = sel.iloc[:BLOCKS_MAX_FEATURES]
    if len(sel) == 0:
        return JSONResponse(empty)
    cols = ["block_num"] if layer == "blocks" and "block_num" in sel.columns else []
    sel = sel[cols + [sel.geometry.name]].copy()
    sel.geometry = sel.geometry.simplify(2.0)
    return JSONResponse(json.loads(sel.to_crs("EPSG:4326").to_json()))


# -- tiles -----------------------------------------------------------------------------

def _max_zoom(w: int, h: int) -> int:
    return max(0, math.ceil(math.log2(max(w, h, 1))))


@functools.lru_cache(maxsize=2)
def _open_cached(path: str, mtime_ns: int) -> Image.Image:
    Image.MAX_IMAGE_PIXELS = None  # our own scans; large PNGs are expected
    im = Image.open(path)
    im.load()
    return im if im.mode in ("L", "RGB") else im.convert("RGB")


@app.get("/api/sheet/{series}/{sheet_id}/image/{z}/{x}/{y}.png")
def image_tile(series: str, sheet_id: str, z: int, x: int, y: int) -> FileResponse:
    d = _sheet_dir(series, sheet_id)
    png = d / "page.png"
    w, h = _png_size(png)
    zmax = _max_zoom(w, h)
    if not (0 <= z <= zmax) or x < 0 or y < 0:
        raise HTTPException(404, "no such tile")
    scale = 2 ** (zmax - z)
    lw, lh = math.ceil(w / scale), math.ceil(h / scale)
    if x * TILE_SIZE >= lw or y * TILE_SIZE >= lh:
        raise HTTPException(404, "no such tile")
    out = d / "dz" / str(z) / f"{x}_{y}.png"
    if not out.exists() or out.stat().st_mtime_ns < png.stat().st_mtime_ns:
        im = _open_cached(str(png), png.stat().st_mtime_ns)
        tw, th = min(TILE_SIZE, lw - x * TILE_SIZE), min(TILE_SIZE, lh - y * TILE_SIZE)
        box = (x * TILE_SIZE * scale, y * TILE_SIZE * scale,
               min(w, (x * TILE_SIZE + tw) * scale), min(h, (y * TILE_SIZE + th) * scale))
        tile = im.crop(box)
        if tile.size != (tw, th):
            tile = tile.resize((tw, th), Image.Resampling.BOX if scale > 1 else Image.Resampling.NEAREST)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(out.stem + f".{threading.get_ident()}.tmp.png")
        tile.save(tmp)
        tmp.replace(out)
    return FileResponse(out, media_type="image/png")


# -- preview ---------------------------------------------------------------------------

def render_preview(d: Path, t: SheetTransform, max_px: int = PREVIEW_MAX_PX, step: int = 32) -> dict:
    """Low-res warp of page.png under `t` to d/preview.png (RGBA, transparent paper) with a
    world file d/preview.pgw (EPSG:2227 ft). Returns {width, height, bounds}."""
    from scipy.interpolate import RegularGridInterpolator

    from sfzmaps import warp

    img = cv2.imread(str(d / "page.png"), cv2.IMREAD_UNCHANGED)
    h, w = img.shape[:2]
    s = min(1.0, max_px / max(h, w))
    small = cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA) if s < 1 else img
    b = _bounds(t, (w, h))
    if b is None or not np.isfinite(b).all() or max(b[2] - b[0], b[3] - b[1]) <= 0:
        raise HTTPException(400, "these control points do not give a usable transform")
    minx, miny, maxx, maxy = b
    res = max(maxx - minx, maxy - miny) / max_px          # ft per output px
    ow, oh = max(1, math.ceil((maxx - minx) / res)), max(1, math.ceil((maxy - miny) / res))
    gx = np.unique(np.r_[np.arange(0, ow, step), ow - 1]).astype(float)
    gy = np.unique(np.r_[np.arange(0, oh, step), oh - 1]).astype(float)
    X, Y = np.meshgrid(gx, gy)
    world = np.column_stack([minx + (X.ravel() + 0.5) * res, maxy - (Y.ravel() + 0.5) * res])
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        src = t.to_pixel(world).reshape(len(gy), len(gx), 2) * s
    qx, qy = np.meshgrid(np.arange(ow, dtype=float), np.arange(oh, dtype=float))
    pts = np.column_stack([qy.ravel(), qx.ravel()])
    maps = []
    for k in (0, 1):
        f = RegularGridInterpolator((gy, gx), src[..., k], bounds_error=False, fill_value=np.nan)
        maps.append(f(pts).reshape(oh, ow))
    bad = ~(np.isfinite(maps[0]) & np.isfinite(maps[1]))
    mx, my = np.where(bad, -1.0, maps[0]).astype(np.float32), np.where(bad, -1.0, maps[1]).astype(np.float32)
    rgba = warp.make_rgba(small, [])
    out = cv2.remap(rgba, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    ok = cv2.cvtColor(out, cv2.COLOR_RGBA2BGRA)
    uid = f"{threading.get_ident()}.{time.time_ns()}"  # unique per call: concurrent previews
    tmp = d / f"preview.{uid}.tmp.png"
    cv2.imwrite(str(tmp), ok)
    tmp.replace(d / "preview.png")
    wtmp = d / f"preview.{uid}.tmp.pgw"
    wtmp.write_text(
        "\n".join(repr(v) for v in (res, 0.0, 0.0, -res, minx + res / 2, maxy - res / 2)) + "\n")
    wtmp.replace(d / "preview.pgw")
    return {"width": ow, "height": oh, "bounds": [minx, miny, maxx, maxy]}


def _score_new(d: Path, t: SheetTransform, change_areas: dict | None) -> score_mod.SheetScore:
    return runner.score_review(t, d / "refine.json", _get_ref(), change_areas, d / "labels.json")


@app.post("/api/sheet/{series}/{sheet_id}/preview")
def preview(series: str, sheet_id: str, body: PreviewBody) -> JSONResponse:
    d = _sheet_dir(series, sheet_id)
    t = _transform_of(body.gcps)
    sc = _score_new(d, t, body.change_areas)
    meta = render_preview(d, t)
    return JSONResponse(_clean({
        "score": _score_dict(sc), "residuals": _residuals(t),
        "preview_png_url": f"/api/sheet/{series}/{sheet_id}/preview.png?v={time.time_ns()}",
        "preview_bounds": meta["bounds"], "preview_size": {"width": meta["width"], "height": meta["height"]},
    }))


@app.get("/api/sheet/{series}/{sheet_id}/preview.png")
def preview_png(series: str, sheet_id: str) -> FileResponse:
    p = _sheet_dir(series, sheet_id) / "preview.png"
    if not p.exists():
        raise HTTPException(404, "no preview yet")
    return FileResponse(p, media_type="image/png", headers={"Cache-Control": "no-store"})


# -- save ------------------------------------------------------------------------------

@app.post("/api/sheet/{series}/{sheet_id}/save")
def save(series: str, sheet_id: str, body: SaveBody, background: BackgroundTasks) -> JSONResponse:
    d = _sheet_dir(series, sheet_id)
    t = _transform_of(body.gcps)
    sc = _score_new(d, t, body.change_areas)
    note = body.note.strip()
    if body.status == "accepted" and not sc.passed and not note:
        raise HTTPException(
            400, f"accepting a sheet that does not pass ({sc.reason}; held-out median "
                 f"{sc.holdout_p50_ft:.1f} ft, p95 {sc.holdout_p95_ft:.1f} ft) requires a note")
    review = {"status": body.status, "reviewer": body.reviewer or _git_email(),
              "at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "note": note,
              "gcps": [g.model_dump() for g in body.gcps], "change_areas": body.change_areas}
    page = _find_page(series, sheet_id)
    if page is None:
        raise HTTPException(404, f"no source page for {series}/{sheet_id}")
    with _save_lock:
        runner._write_json(d / "review.json", review)  # atomic
        try:
            new = runner.run_sheet(page, _get_ref())
        except Exception as e:  # noqa: BLE001 - review.json is saved; report the failed rerun
            log.exception("run_sheet failed after save of %s/%s", series, sheet_id)
            raise HTTPException(500, f"review saved, but rerunning the sheet failed: {type(e).__name__}: {e}")
    background.add_task(_schedule_rebuild, series)
    return JSONResponse(_clean({"status": body.status, "score": _score_dict(new),
                                "preview_score": _score_dict(sc), "reviewer": review["reviewer"]}))


# -- optional static front end (Task 17) -----------------------------------------------

_static = Path(__file__).parent / "static"
if _static.is_dir():
    from fastapi.staticfiles import StaticFiles

    class _ReadOnlyStatic(StaticFiles):
        """Static files answer GET/HEAD only; anything else that fell through the API routes
        (a mangled /api path) is a plain 404, not the static handler's 405."""

        async def __call__(self, scope, receive, send):
            if scope["type"] == "http" and scope["method"] not in ("GET", "HEAD"):
                await JSONResponse({"detail": "Not Found"}, status_code=404)(scope, receive, send)
                return
            await super().__call__(scope, receive, send)

    app.mount("/", _ReadOnlyStatic(directory=_static, html=True), name="static")


def serve(port: int = PORT) -> None:
    import uvicorn
    uvicorn.run(app, host=HOST, port=port, log_level="info")
