"""Per-sheet orchestration with artifact-based staleness and human-review precedence.

Stages, in order: render -> classify -> labels -> blocks -> rough -> refine -> changes ->
score -> warp. Artifacts live in config.WORK_DIR/<series>/<sheet_id>/:

  page.png          render                        (input: the source scan)
  page.json         classify                      (page.png)
  labels.json       labels                        (page.png, page.json)
  blocks_px.geojson blocks: pixel polygons + "corners" (page.png, page.json)
  rough.json        rough: {"transform": ...|null, "scale_hint_ft_per_px"}
                    (labels.json, blocks_px.geojson, reference parquet files)
  refine_raw.json   refine, before change classification (blocks_px, rough.json, reference)
  refine.json       changes: final matches (px, py, wx, wy, dist_ft, label, ref_idx) plus
                    converged, vote_status, classify_converged (Task 16's editor reads it)
  changes.geojson   changes                       (refine_raw.json)
  transform.json    final transform, with "origin": auto | review | answer_key
  score.json        score                         (transform.json, refine.json, labels.json)
  warped.tif        warp                          (transform.json, page.json, page.png)
  review.json       optional, written by the human review tool; never touched here

Staleness: a stage is skipped when its artifact exists and is strictly newer than every
input artifact. The reference parquet files count as inputs of rough/refine. A changed
`config` value (thresholds, label scale) does NOT trigger a rerun; pass force=True
(`sfzmaps run --force`) to rerun every stage. Labels are cached per vision call, so
reruns cost nothing in API calls.

Review: with review.json present (status "edited" or "accepted") and not force_auto, its
GCPs become transform.json (origin "review"); rough/refine/changes are skipped, the score
is still computed for display and the warp uses the review GCPs. force_auto ignores the
review for that run (the file itself is never modified). If the automatic rough fit fails
on a 1948/1960 sheet and the owner's .points file exists, its GCPs are the transform
(origin "answer_key"), score.json keeps reason "insufficient_anchors" and flags
"answer_key_fallback"; refine and changes are skipped.
If refine finishes but the sheet scores "misplaced" and a key exists, the key is likewise the
transform (origin "answer_key"); score.json keeps reason "misplaced", passed false. A rescore
of such a sheet uses the auto fit stored in refine.json, never the key transform.
For every answer_key_fallback sheet score.json carries "metrics_of": "auto_fit": the
numbers describe the automatic fit (none for insufficient_anchors), not the key warp.
score.json also carries "score_version" (config.SCORE_VERSION); a missing or lower
version is stale and rescored. Bump it whenever pass or fallback logic changes.
"""
from __future__ import annotations

import json
import logging
import math
import re
import subprocess
import time
import traceback
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import mapping

from sfzmaps import anchors, config, linework, llm, pages, reading, warp
from sfzmaps import refine as refine_mod
from sfzmaps import score as score_mod
from sfzmaps.pages import Page
from sfzmaps.pointsfile import read_points
from sfzmaps.refine import Match, RefineResult
from sfzmaps.score import SheetScore
from sfzmaps.sources.datasf import Reference
from sfzmaps.transform import GCP, SheetTransform

log = logging.getLogger(__name__)

STAGES = ["render", "classify", "labels", "blocks", "rough", "refine", "changes", "score", "warp"]
last_timings: dict[str, float | None] = {}  # stage -> seconds, None when skipped (last run_sheet)


# -- small helpers ---------------------------------------------------------------------

def _num(x):
    x = float(x)
    return x if math.isfinite(x) else None


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=1))
    tmp.replace(path)


def _mtime(p: Path) -> int:
    return p.stat().st_mtime_ns


def _fresh(out: Path, inputs: list[Path], force: bool = False) -> bool:
    if force or not out.exists() or not inputs:  # no inputs: nothing proves it current
        return False
    return all(i.exists() and _mtime(out) > _mtime(i) for i in inputs)


def _ref_inputs() -> list[Path]:
    return [p for p in (config.REF_DIR / "parcels.parquet", config.REF_DIR / "streets.parquet") if p.exists()]


def _transform_json(t: SheetTransform, origin: str) -> dict:
    return {**t.to_json(), "origin": origin}


def _origin(p: Path) -> str | None:
    try:
        return json.loads(p.read_text()).get("origin")
    except (OSError, ValueError):
        return None


def _match_json(m: Match) -> dict:
    return {"px": _num(m.px), "py": _num(m.py), "wx": _num(m.wx), "wy": _num(m.wy),
            "dist_ft": _num(m.dist_ft), "label": m.label, "ref_idx": int(m.ref_idx)}


def _match_load(d: dict) -> Match:
    nan = float("nan")
    return Match(d["px"] if d["px"] is not None else nan, d["py"] if d["py"] is not None else nan,
                 d["wx"] if d["wx"] is not None else nan, d["wy"] if d["wy"] is not None else nan,
                 d["dist_ft"] if d["dist_ft"] is not None else float("inf"), d["label"], d["ref_idx"])


def _result_json(r: RefineResult) -> dict:
    return {"transform": r.transform.to_json(), "matches": [_match_json(m) for m in r.matches],
            "iterations": r.iterations, "converged": r.converged,
            "vote_shift_ft": list(r.vote_shift_ft), "vote_status": r.vote_status,
            "final_radius_ft": r.final_radius_ft, "classify_converged": r.classify_converged}


def _result_load(d: dict, ref: Reference | None, transform: SheetTransform | None = None) -> RefineResult:
    W, blk = refine_mod._ref_arrays(ref.corners) if ref is not None else (None, None)
    return RefineResult(transform or SheetTransform.from_json(d["transform"]),
                        [_match_load(m) for m in d["matches"]], d["iterations"], d["converged"],
                        tuple(d["vote_shift_ft"]), d["vote_status"], d["final_radius_ft"], W, blk,
                        classify_converged=d.get("classify_converged", True))


def _score_json(s: SheetScore, **extra) -> dict:
    return {**{k: (_num(v) if isinstance(v, float) else v) for k, v in asdict(s).items()},
            "score_version": config.SCORE_VERSION, **extra}


def _score_current(d: dict) -> bool:
    """False for a score.json without holdout_p50_ft (pre-held-out format) or with a missing
    or older config.SCORE_VERSION: it was judged by older pass/fallback logic, so it is
    stale and must be rescored."""
    return "holdout_p50_ft" in d and d.get("score_version", 0) >= config.SCORE_VERSION


def _score_load(d: dict) -> SheetScore:
    """SheetScore from score.json; also reads pre-Task-13-fix files (frac_within, p50_ft...),
    which never count as passed (no held-out fields)."""
    nan = float("nan")

    def g(k, old=None):
        v = d.get(k, d.get(old) if old else None)
        return nan if v is None else v

    stale = not _score_current(d)
    return SheetScore(d["n_scored"], g("insample_frac_within", "frac_within"),
                      g("insample_p50_ft", "p50_ft"), g("insample_p95_ft", "p95_ft"),
                      g("insample_max_ft", "max_ft"), d["n_real_change"], d["n_no_equivalent"],
                      False if stale else d["passed"], "stale_score" if stale else d["reason"],
                      g("holdout_p50_ft"), g("holdout_p95_ft"),
                      d.get("n_holdout", 0), g("label_agreement"), g("inlier_frac"))


# -- answer key ------------------------------------------------------------------------

def answer_key_path(page: Page) -> Path | None:
    """The owner's .points file for a 1948/1960 sheet (ruling R9), or None."""
    if page.series not in ("1948", "1960") or page.page_index is not None:
        return None
    a = config.RAW_DIR / page.series / "aligned"
    for name in (f"{page.sheet_id}.jpg.points", f"{page.sheet_id}.pdf.points",
                 f"{page.sheet_id}.jpg.just-2.points"):
        if (a / name).exists():
            return a / name
    return None


def _gdal_width(pdf: Path) -> int:
    try:
        import rasterio
        with rasterio.open(pdf) as ds:
            return int(ds.width)
    except Exception:  # noqa: BLE001 - driver missing; try the CLI
        out = subprocess.run(["gdalinfo", str(pdf)], capture_output=True, text=True, check=True).stdout
        m = re.search(r"Size is (\d+),\s*(\d+)", out)
        if not m:
            raise RuntimeError(f"cannot read raster size of {pdf}")
        return int(m.group(1))


def _answer_key_gcps(page: Page, png: Path) -> list[GCP] | None:
    p = answer_key_path(page)
    if p is None:
        return None
    scale = 1.0
    if page.source.suffix.lower() == ".pdf":  # R7: file pixels are in GDAL's raster of the PDF
        img = cv2.imread(str(png), cv2.IMREAD_UNCHANGED)
        scale = img.shape[1] / _gdal_width(page.source)
    gcps = read_points(p, scale=scale)
    return gcps if sum(g.enabled for g in gcps) >= 3 else None


# -- review ----------------------------------------------------------------------------

def _load_review(d: Path) -> dict | None:
    p = d / "review.json"
    if not p.exists():
        return None
    r = json.loads(p.read_text())
    return r if r.get("status") in ("edited", "accepted") else None


def _review_result(review_t: SheetTransform, refine_json: Path, ref: Reference | None) -> RefineResult:
    """Matches for display scoring: refine.json's, re-measured under the review transform,
    else one inlier per enabled review GCP (its residual)."""
    if refine_json.exists():
        r = _result_load(json.loads(refine_json.read_text()), ref, review_t)
        P = np.array([[m.px, m.py] for m in r.matches], float)
        ok = np.isfinite(P).all(axis=1) & np.array([m.label in ("inlier", "outlier") for m in r.matches], bool)
        if ok.any():
            w = review_t.to_world(P[ok])
            for m, wp in zip([m for m, k in zip(r.matches, ok) if k], w):
                m.dist_ft = float(np.hypot(wp[0] - m.wx, wp[1] - m.wy))
        return r
    res = review_t.residuals_ft()
    gs = review_t._enabled()
    ms = [Match(g.px, g.py, g.wx, g.wy, float(e), "inlier", -1) for g, e in zip(gs, res)]
    return RefineResult(review_t, ms, 0, True)


# -- the runner ------------------------------------------------------------------------

def run_sheet(page: Page, ref: Reference, *, until: str | None = None, force_auto: bool = False,
              force: bool = False) -> SheetScore:
    if until is not None and until not in STAGES:
        raise ValueError(f"unknown stage {until!r}; stages: {', '.join(STAGES)}")
    d = config.WORK_DIR / page.series / page.sheet_id
    d.mkdir(parents=True, exist_ok=True)
    png, pj, lj, bj = d / "page.png", d / "page.json", d / "labels.json", d / "blocks_px.geojson"
    rj, rawj, rfj, cj = d / "rough.json", d / "refine_raw.json", d / "refine.json", d / "changes.geojson"
    tj, sj, wt = d / "transform.json", d / "score.json", d / "warped.tif"
    last_timings.clear()
    cache: dict = {}
    if sj.exists():
        try:
            if not _score_current(json.loads(sj.read_text())):
                sj.unlink()  # old-format score: rescored below, never reused
        except ValueError:
            sj.unlink()

    def stage(name: str, out: Path, inputs: list[Path], fn) -> bool:
        if _fresh(out, inputs, force):
            last_timings[name] = None
            log.info("%-8s skipped (fresh)", name)
            return False
        t0 = time.monotonic()
        fn()
        last_timings[name] = time.monotonic() - t0
        log.info("%-8s done in %.1fs", name, last_timings[name])
        return True

    def stopped(name: str) -> bool:
        return until == name

    def partial(why: str) -> SheetScore:
        if sj.exists():
            return _score_load(json.loads(sj.read_text()))
        return SheetScore.empty(why)

    def img():
        if "img" not in cache:
            cache["img"] = cv2.imread(str(png), cv2.IMREAD_UNCHANGED)
        return cache["img"]

    def info() -> reading.PageInfo:
        if "info" not in cache:
            cache["info"] = reading.load_page(pj)
        return cache["info"]

    def mask():
        if "mask" not in cache:
            cache["mask"] = reading.exclusion_mask(img(), info())
        return cache["mask"]

    # render / classify
    stage("render", png, [page.source], lambda: pages.render(page, png))
    if stopped("render"):
        return partial("until: render")
    stage("classify", pj, [png], lambda: reading.save(pj, reading.classify_page(png)))
    if info().kind != "map":
        log.info("skipped: %s", info().kind)
        return SheetScore.empty(f"skipped: {info().kind}")
    if stopped("classify"):
        return partial("until: classify")

    # labels / blocks
    stage("labels", lj, [png, pj], lambda: reading.save(
        lj, reading.read_labels(png, scale=config.label_scale(page.series), skip=mask())))
    if stopped("labels"):
        return partial("until: labels")

    def do_blocks():
        polys = linework.extract_blocks(img(), exclude=mask())
        corners = np.asarray(linework.block_corners_px(polys), float).reshape(-1, 2)
        _write_json(bj, {"type": "FeatureCollection",
                         "features": [{"type": "Feature", "properties": {}, "geometry": mapping(p)}
                                      for p in polys],
                         "corners": [[_num(x), _num(y)] for x, y in corners]})

    stage("blocks", bj, [png, pj], do_blocks)
    if stopped("blocks"):
        return partial("until: blocks")

    review = None if force_auto else _load_review(d)
    answer_key = False
    result: RefineResult | None = None

    if review is not None:
        rt = SheetTransform([GCP(**g) for g in review["gcps"]], config.MANUAL_TPS_SMOOTHING)
        rpath = d / "review.json"

        def do_review():
            _write_json(tj, _transform_json(rt, "review"))
            if review.get("change_areas") is not None:
                _write_json(cj, review["change_areas"])
            else:
                cj.unlink(missing_ok=True)  # no stale auto changes next to a reviewed transform

        if not (_origin(tj) == "review" and _fresh(tj, [rpath], force)):
            do_review()
            log.info("review   transform from review.json (%s)", review["status"])
        for s in ("rough", "refine", "changes"):
            last_timings[s] = None
        if until in ("rough", "refine", "changes"):
            return partial(f"until: {until}")
    else:
        def do_rough():
            from shapely.geometry import shape
            labels = reading.load_labels(lj)
            polys = [shape(f["geometry"]) for f in json.loads(bj.read_text())["features"]]
            street_px, measured = linework.estimate_street_px(polys)
            hint = linework.STREET_FT / street_px if measured else None  # ft/px; a default is no measurement
            fit = anchors.rough_fit(anchors.match_labels(labels, ref), scale_hint=hint)
            if fit is not None:
                fit = anchors.street_snap(fit, labels, ref)
            _write_json(rj, {"transform": _transform_json(fit.transform, "auto") if fit else None,
                             "scale_hint_ft_per_px": None if hint is None else _num(hint)})

        stage("rough", rj, [lj, bj, *_ref_inputs()], do_rough)
        if stopped("rough"):
            return partial("until: rough")
        rough = json.loads(rj.read_text())["transform"]
        if rough is None:
            for s in ("refine", "changes"):
                last_timings[s] = None
            for p in (rawj, rfj, cj):
                p.unlink(missing_ok=True)
            gcps = _answer_key_gcps(page, png) if answer_key_path(page) else None
            if gcps is not None:
                answer_key = True
                if not (_origin(tj) == "answer_key" and _fresh(tj, [rj], force)):
                    _write_json(tj, _transform_json(SheetTransform(gcps, config.MANUAL_TPS_SMOOTHING), "answer_key"))
                    log.info("fallback transform from answer key (%d GCPs)", len(gcps))
            else:
                tj.unlink(missing_ok=True)
                wt.unlink(missing_ok=True)  # a warp of an older transform must not survive
        else:
            def do_refine():
                data = json.loads(bj.read_text())
                corners = np.array([[np.nan if v is None else v for v in c] for c in data["corners"]],
                                   float).reshape(-1, 2)
                res = refine_mod.refine(corners, ref.corners, SheetTransform.from_json(rough))
                _write_json(rawj, _result_json(res))

            stage("refine", rawj, [bj, rj, *_ref_inputs()], do_refine)
            if stopped("refine"):
                return partial("until: refine")

            def do_changes():
                res = refine_mod.classify_changes(_result_load(json.loads(rawj.read_text()), ref))
                _write_json(rfj, _result_json(res))
                _write_json(cj, refine_mod.changes_geojson(res))
                _write_json(tj, _transform_json(res.transform, "auto"))  # last: newest artifact

            # changes also rewrites transform.json, so it reruns when that is a review's
            # (force_auto after a review) as well as when its own artifacts are stale.
            # (a misplaced sheet's tj is the answer key's while rfj exists: still current)
            if (_origin(tj) == "auto" or (_origin(tj) == "answer_key" and rfj.exists())) \
                    and _fresh(cj, [rawj, rfj], force) and _fresh(tj, [rawj], force):
                last_timings["changes"] = None
                log.info("%-8s skipped (fresh)", "changes")
            else:
                t0 = time.monotonic()
                do_changes()
                last_timings["changes"] = time.monotonic() - t0
                log.info("%-8s done in %.1fs", "changes", last_timings["changes"])
            if stopped("changes"):
                return partial("until: changes")

    # score
    def agreement(t: SheetTransform):
        from sfzmaps import placement
        return placement.label_agreement(t, reading.load_labels(lj), ref) if lj.exists() else None

    def do_score():
        extra = {}
        if review is not None:
            rt_ = SheetTransform.from_json(json.loads(tj.read_text()))
            res = _review_result(rt_, rfj, ref)
            sc = score_mod.score(res, agreement=agreement(rt_))
            extra["source"] = "review"
        elif answer_key or not rfj.exists():
            sc = score_mod.score(None)
            if answer_key:
                extra["answer_key_fallback"] = True
                extra["metrics_of"] = "auto_fit"
        else:
            tdata = json.loads(tj.read_text())
            keyed = tdata.get("origin") == "answer_key"
            if keyed:  # a previous misplaced fallback: score the auto fit, not the key
                t_ = SheetTransform.from_json(json.loads(rfj.read_text())["transform"])
            else:
                t_ = SheetTransform.from_json(tdata)
            res = _result_load(json.loads(rfj.read_text()), ref, t_)
            sc = score_mod.score(res, agreement=agreement(t_))
            gcps = None
            if sc.reason == "misplaced":
                # The auto transform is in the wrong place; a sheet with an owner key is
                # warped from that instead.  Still not passed, so review sees it.
                gcps = _answer_key_gcps(page, png) if answer_key_path(page) else None
            if gcps is not None:
                _write_json(tj, _transform_json(SheetTransform(gcps, config.MANUAL_TPS_SMOOTHING), "answer_key"))
                extra["answer_key_fallback"] = True
                extra["metrics_of"] = "auto_fit"
                log.info("misplaced: transform from answer key (%d GCPs)", len(gcps))
            elif keyed:
                _write_json(tj, _transform_json(t_, "auto"))  # no longer misplaced: back to auto
        _write_json(sj, _score_json(sc, **extra))

    stage("score", sj, [p for p in (tj, rfj, lj) if p.exists()] or [rj], do_score)
    score_val = _score_load(json.loads(sj.read_text()))
    if stopped("score"):
        return score_val

    # warp
    if tj.exists():
        stage("warp", wt, [tj, pj, png, Path(warp.__file__)],  # warp.py: code change re-warps
              lambda: warp.warp_sheet(png, SheetTransform.from_json(json.loads(tj.read_text())),
                                      info().mask_boxes, wt))
    else:
        wt.unlink(missing_ok=True)
        last_timings["warp"] = None
    return score_val


def format_table(results: dict[str, SheetScore]) -> str:
    def f(x, spec):
        return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else format(x, spec)

    rows = [("sheet", "passed", "holdout_p50", "holdout_p95", "insample_p95", "reason")]
    for k, s in results.items():
        rows.append((k, str(s.passed), f(s.holdout_p50_ft, ".1f"), f(s.holdout_p95_ft, ".1f"),
                     f(s.insample_p95_ft, ".1f"), s.reason or ""))
    w = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    return "\n".join("  ".join(c.ljust(w[i]) for i, c in enumerate(r)).rstrip() for r in rows)


def run_series(series: str, *, sheet: str | None = None, until: str | None = None,
               force_auto: bool = False, force: bool = False, ref: Reference | None = None,
               pages_list: list[Page] | None = None) -> dict[str, SheetScore]:
    """Run every sheet of a series. A failing sheet is logged (with traceback) to
    OUT_DIR/run-<timestamp>.log, recorded with reason 'error: ...', and the run continues."""
    if ref is None:
        from sfzmaps.sources import datasf
        ref = datasf.load_reference()
    todo = [p for p in (pages_list if pages_list is not None else pages.discover())
            if p.series == series and (sheet is None or p.sheet_id == sheet)]
    results: dict[str, SheetScore] = {}
    logfile: Path | None = None
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    for p in todo:
        log.info("== %s/%s", p.series, p.sheet_id)
        try:
            results[p.sheet_id] = run_sheet(p, ref, until=until, force_auto=force_auto, force=force)
        except llm.LLMAuthError as e:  # every remaining sheet would fail the same way
            print(f"error: {e}; aborting series {series} at sheet {p.sheet_id}")
            results[p.sheet_id] = SheetScore.empty(f"error: {e}")
            break
        except Exception as e:  # noqa: BLE001 - one bad sheet must not stop the series
            if logfile is None:
                config.OUT_DIR.mkdir(parents=True, exist_ok=True)
                logfile = config.OUT_DIR / f"run-{stamp}.log"
            with logfile.open("a") as fh:
                fh.write(f"=== {p.series}/{p.sheet_id}\n{traceback.format_exc()}\n")
            results[p.sheet_id] = SheetScore.empty(f"error: {type(e).__name__}: {e}")
    print(format_table(results))
    if logfile is not None:
        print(f"errors logged to {logfile}")
    return results
