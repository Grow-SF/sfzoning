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
from sfzmaps.score import SheetScore
from sfzmaps.transform import SheetTransform


@dataclass
class EvalRow:
    sheet: str
    origin: str | None             # transform.json origin, None when there is no transform
    score: SheetScore | None       # score.json as written by the runner
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
    score = runner._score_load(json.loads(sj.read_text())) if sj.exists() else None
    if score is not None and str(score.reason or "").startswith("skipped"):
        return None
    nan = float("nan")
    origin, t = None, None
    if tj.exists():
        tdata = json.loads(tj.read_text())
        origin, t = tdata.get("origin"), SheetTransform.from_json(tdata)
    key = runner.answer_key_path(page)
    if key is None:
        return EvalRow(page.sheet_id, origin, score, 0, nan, nan, "no answer key")
    # page.png is only needed to rescale a PDF's key (R7); image keys are in source pixels
    needs_png = page.source.suffix.lower() == ".pdf"
    gcps = runner._answer_key_gcps(page, png) if (png.exists() or not needs_png) else None
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


COLUMNS = ("sheet", "pass", "reason", "ho_p50", "ho_p95", "lim50", "lim95", "in_p50", "in_p95", "in_f6", "n_scored",
           "n_rc", "n_noeq", "agree", "inl", "origin", "n_key", "key_med", "key_max", "note")


def format_eval_table(rows: list[EvalRow]) -> str:
    """Fixed-width table; distances in ft. ho_* = held-out (decides the pass), lim50/lim95 =
    the sheet's scale-aware pass limits for them (score.pass_limits), in_* =
    in-sample (display), in_f6 = in-sample share within config.TOLERANCE_FT, agree = share
    of the sheet's labels landing in their block / on their street, inl = inlier share of
    scan corners. A summary line follows the rows."""
    out = [COLUMNS]
    for r in rows:
        s = r.score
        if s is None:
            out.append((r.sheet, "-", "no score", *["-"] * 12, r.origin or "-", str(r.n_key),
                        _f(r.key_median_ft, ".1f"), _f(r.key_max_ft, ".1f"), r.note))
            continue
        out.append((r.sheet, "yes" if s.passed else "no", s.reason or "-",
                    _f(s.holdout_p50_ft, ".1f"), _f(s.holdout_p95_ft, ".1f"),
                    _f(s.pass_median_limit_ft, ".1f"), _f(s.pass_p95_limit_ft, ".1f"),
                    _f(s.insample_p50_ft, ".1f"), _f(s.insample_p95_ft, ".1f"),
                    _f(s.insample_frac_within if s.n_scored else None, ".2f"), str(s.n_scored),
                    str(s.n_real_change), str(s.n_no_equivalent), _f(s.label_agreement, ".2f"),
                    _f(s.inlier_frac, ".2f"), r.origin or "-", str(r.n_key),
                    _f(r.key_median_ft, ".1f"), _f(r.key_max_ft, ".1f"), r.note))
    w = [max(len(row[i]) for row in out) for i in range(len(COLUMNS))]
    lines = ["  ".join(c.ljust(w[i]) for i, c in enumerate(row)).rstrip() for row in out]
    sc = [r.score for r in rows if r.score is not None]

    def med(v):
        v = [x for x in v if x is not None and math.isfinite(x)]
        return _f(float(np.median(v)) if v else None, ".1f")

    km = [r.key_median_ft for r in rows if r.origin != "answer_key"]

    def span(v):
        v = [x for x in v if x is not None and math.isfinite(x)]
        if not v:
            return "-"
        lo, hi = min(v), max(v)
        return f"{lo:.1f}" if abs(hi - lo) < 0.05 else f"{lo:.1f}-{hi:.1f}"

    lim50 = span([s.pass_median_limit_ft for s in sc])
    lim95 = span([s.pass_p95_limit_ft for s in sc])
    lines.append(f"passed {sum(s.passed for s in sc)}/{len(rows)}; median held-out p50 "
                 f"{med([s.holdout_p50_ft for s in sc])} ft, p95 {med([s.holdout_p95_ft for s in sc])} ft "
                 f"(in-sample p95 {med([s.insample_p95_ft for s in sc])} ft); median key distance "
                 f"{med(km)} ft (pass: held-out median <= {lim50} ft and p95 <= {lim95} ft, the "
                 f"per-sheet limits {config.PASS_MEDIAN_FT:g} / {config.PASS_P95_FT:g} ft x k, "
                 f"k = max(1, ft_per_px / {config.PASS_REF_FT_PER_PX:g}); not misplaced)")
    return "\n".join(lines)
