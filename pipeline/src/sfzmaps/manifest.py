"""manifest.json for one series: the site and review page read it (sheet status, credit, tiles)."""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import rasterio
from rasterio.warp import transform_bounds

from sfzmaps import config
from sfzmaps.score import SheetScore


def _num(x) -> float | None:
    return None if x is None or not math.isfinite(float(x)) else float(x)


def _json(path: Path) -> dict | None:
    try:
        d = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return d if isinstance(d, dict) else None


def status_of(series: str, sheet_id: str, score: SheetScore) -> str:
    """"edited" / "accepted" from review.json, else "passed" or "needs_review"."""
    r = _json(config.WORK_DIR / series / sheet_id / "review.json")
    if r and r.get("status") in ("edited", "accepted"):
        return r["status"]
    return "passed" if score.passed else "needs_review"


def bounds_wgs84(warped: Path) -> list[float] | None:
    if not warped.exists():
        return None
    with rasterio.open(warped) as ds:
        b = transform_bounds(ds.crs, "EPSG:4326", *ds.bounds)
    return [round(v, 6) for v in b]


def write(series: str, scores: dict[str, SheetScore], tile_report=None) -> Path:
    sheets = []
    for sid in sorted(scores, key=lambda s: (len(s), s)):
        sc = scores[sid]
        d = config.WORK_DIR / series / sid
        extra = _json(d / "score.json") or {}
        sheets.append({
            "sheet_id": sid,
            "status": status_of(series, sid, sc),
            "holdout_p50_ft": _num(sc.holdout_p50_ft),
            "holdout_p95_ft": _num(sc.holdout_p95_ft),
            "reason": sc.reason,
            "answer_key_fallback": bool(extra.get("answer_key_fallback", False)),
            "bounds_wgs84": bounds_wgs84(d / "warped.tif"),
        })
    out = config.OUT_DIR / series / "manifest.json"
    if tile_report is None:   # a rewrite without a tile build keeps the last tile report
        prev = (_json(out) or {}).get("tile_report")
    else:
        prev = {"bytes_total": tile_report.bytes_total,
                "bytes_by_zoom": {str(z): b for z, b in sorted(tile_report.bytes_by_zoom.items())}}
    data = {
        "series": series,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "credit": config.RUMSEY_CREDIT if series == "1948" else None,
        "pmtiles": f"{series}.pmtiles",
        "tile_report": prev,
        "sheets": sheets,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=1))
    tmp.replace(out)
    return out
