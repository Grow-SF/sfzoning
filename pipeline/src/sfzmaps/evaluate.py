"""Answer-key evaluation (`sfzmaps eval`): our alignment next to the owner's QGIS points.

For each sheet of a series that has a work dir, reads score.json (our SheetScore) and
transform.json, and, when the owner's .points file exists, measures the distance from our
to_world(px) to the owner's (mapX, mapY) at every enabled answer-key point.

The owner's points are hand-placed and coarse (~8 per sheet), so a large distance means
"look at this sheet", not automatically "we are wrong". When our transform *is* the answer
key (origin "answer_key", the rough-fit fallback) the distances are only the smooth fit's
residuals, and the row says so in `origin`.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sfzmaps import config
from sfzmaps.pages import Page
from sfzmaps.transform import SheetTransform


@dataclass
class EvalRow:
    sheet: str
    origin: str | None             # transform.json origin, None when there is no transform
    score: dict | None             # score.json as written by the runner
    n_key: int                     # enabled answer-key points (0 when there is no file)
    key_median_ft: float           # NaN when not measurable
    key_max_ft: float
    note: str = ""


def answer_key_distances(t: SheetTransform, gcps) -> np.ndarray:
    """Distance (ft) from t.to_world(px) to the owner's world point, per enabled GCP."""
    gs = [g for g in gcps if g.enabled]
    if not gs:
        return np.empty(0)
    P = np.array([[g.px, g.py] for g in gs], float)
    W = np.array([[g.wx, g.wy] for g in gs], float)
    return np.linalg.norm(t.to_world(P) - W, axis=1)


def eval_sheet(page: Page) -> EvalRow | None:
    """One row, or None when the sheet was never run (no work dir) or is not a map."""
    from sfzmaps import runner

    d = config.WORK_DIR / page.series / page.sheet_id
    sj, tj, png = d / "score.json", d / "transform.json", d / "page.png"
    if not sj.exists() and not tj.exists():
        return None
    score = json.loads(sj.read_text()) if sj.exists() else None
    if score is not None and str(score.get("reason") or "").startswith("skipped"):
        return None
    nan = float("nan")
    origin, t = None, None
    if tj.exists():
        tdata = json.loads(tj.read_text())
        origin, t = tdata.get("origin"), SheetTransform.from_json(tdata)
    key = runner.answer_key_path(page)
    if key is None:
        return EvalRow(page.sheet_id, origin, score, 0, nan, nan, "no answer key")
    gcps = runner._answer_key_gcps(page, png) if png.exists() else None
    if not gcps:
        return EvalRow(page.sheet_id, origin, score, 0, nan, nan, "answer key unreadable")
    n = sum(g.enabled for g in gcps)
    if t is None:
        return EvalRow(page.sheet_id, origin, score, n, nan, nan, "no transform")
    dist = answer_key_distances(t, gcps)
    return EvalRow(page.sheet_id, origin, score, n, float(np.median(dist)), float(dist.max()),
                   "fit to the key itself" if origin == "answer_key" else "")


def evaluate_series(series: str, pages_list: list[Page]) -> list[EvalRow]:
    rows = [eval_sheet(p) for p in pages_list if p.series == series]
    return [r for r in rows if r is not None]


def _f(x, spec: str) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "-"
    return format(x, spec)


COLUMNS = ("sheet", "frac_within", "p50", "p95", "max", "n_scored", "n_rc", "n_noeq",
           "reason", "origin", "n_key", "key_med", "key_max", "note")


def format_eval_table(rows: list[EvalRow]) -> str:
    """Fixed-width table; distances in ft. A summary line follows the rows."""
    out = [COLUMNS]
    for r in rows:
        s = r.score or {}
        scored = bool(s.get("n_scored"))
        out.append((r.sheet, _f(s.get("frac_within") if scored else None, ".3f"),
                    _f(s.get("p50_ft"), ".1f"), _f(s.get("p95_ft"), ".1f"),
                    _f(s.get("max_ft"), ".1f"), str(s.get("n_scored", "-")),
                    str(s.get("n_real_change", "-")), str(s.get("n_no_equivalent", "-")),
                    ("pass" if s.get("passed") else (s.get("reason") or "-")) if s else "no score",
                    r.origin or "-", str(r.n_key), _f(r.key_median_ft, ".1f"),
                    _f(r.key_max_ft, ".1f"), r.note))
    w = [max(len(row[i]) for row in out) for i in range(len(COLUMNS))]
    lines = ["  ".join(c.ljust(w[i]) for i, c in enumerate(row)).rstrip() for row in out]
    n_pass = sum(bool((r.score or {}).get("passed")) for r in rows)
    p95 = [r.score["p95_ft"] for r in rows if r.score and r.score.get("p95_ft") is not None]
    km = [r.key_median_ft for r in rows if math.isfinite(r.key_median_ft) and r.origin != "answer_key"]
    lines.append(f"passed {n_pass}/{len(rows)}; median p95 {_f(float(np.median(p95)) if p95 else None, '.1f')} ft; "
                 f"median key distance {_f(float(np.median(km)) if km else None, '.1f')} ft "
                 f"(tolerance {config.TOLERANCE_FT:g} ft, pass at {config.PASS_FRACTION:.0%})")
    return "\n".join(lines)
