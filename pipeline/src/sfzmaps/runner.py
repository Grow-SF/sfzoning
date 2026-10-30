"""Per-sheet orchestration with artifact-based staleness and human-review precedence.

Stages, in order: render -> classify -> labels -> blocks -> rough -> refine -> changes ->
score -> warp. Artifacts live in config.WORK_DIR/<series>/<sheet_id>/:

  page.png          render                        (input: the source scan)
  page.json         classify                      (page.png)
  labels.json       labels                        (page.png, page.json)
                    both store "page_sha256", the sha256 of the page.png they were read
                    from; when it matches the current page.png they are fresh whatever the
                    mtimes (a byte-identical re-render costs no vision calls)
  blocks_px.geojson blocks: pixel polygons + "corners", the good-quality block corners
                    (linework.good_corners_px)  (page.png, page.json)
  rough.json        rough: {"transform": ...|null, "scale_hint_ft_per_px"}
                    (labels.json, blocks_px.geojson, reference parquet files)
  refine_raw.json   refine, before change classification (blocks_px, rough.json, reference)
  refine.json       changes: final matches (px, py, wx, wy, dist_ft, label, ref_idx) plus
                    converged, vote_status, classify_converged (Task 16's editor reads it)
  changes.geojson   changes                       (refine_raw.json)
  transform.json    final transform, with "origin": auto | review | answer_key
  score.json        score                         (transform.json, refine.json, labels.json, rough.json)
  warped.tif        warp                          (transform.json, page.json, page.png)
  review.json       optional, written by the human review tool; never touched here

Staleness: a stage is skipped when its artifact exists and is strictly newer than every
input artifact. The reference parquet files count as inputs of rough/refine. A changed
`config` value (thresholds, label scale) does NOT trigger a rerun; pass force=True
(`sfzmaps run --force`) to rerun every stage. Labels are cached per vision call, so
reruns cost nothing in API calls.

Review: with review.json present (status "edited" or "accepted") and not force_auto, its
GCPs become transform.json (origin "review"), with review.json's "smoothing" (else chosen
by density, smoothing_for); transform.json is rewritten whenever its content differs from
the review (not by mtime); rough/refine/changes are skipped, the score
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
A sheet that raises, or is classified as non-map, gets an "error: ..." / "skipped: ..."
score.json and loses transform.json and warped.tif (clear_outputs); the next run reruns it.
score.json also carries "score_version" (config.SCORE_VERSION); a missing or lower
version is stale and rescored. Bump it whenever pass or fallback logic changes.
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
import subprocess
import time
import traceback
from dataclasses import asdict, replace
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


def _file_sha256(p: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _add_page_sha(p: Path, sha: str) -> None:
    """Store page_sha256 in an existing page.json / labels.json, keeping its mtime (so no
    downstream stage looks stale)."""
    try:
        d = json.loads(p.read_text())
    except (OSError, ValueError):
        return
    d = {"page_sha256": sha, "labels": d} if isinstance(d, list) else {**d, "page_sha256": sha}
    st = p.stat()
    _write_json(p, d)
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns))


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
                      d.get("n_holdout", 0), g("label_agreement"), g("inlier_frac"),
                      g("label_agreement_local"), g("ft_per_px"),
                      d.get("pass_scale_k") or 1.0,
                      d.get("pass_median_limit_ft") or config.PASS_MEDIAN_FT,
                      d.get("pass_p95_limit_ft") or config.PASS_P95_FT)


def page_shape(png: Path) -> tuple[int, int] | None:
    """(h, w) of a page image, read from its header only; None if unreadable."""
    try:
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None
        with Image.open(png) as im:
            w, h = im.size
        return int(h), int(w)
    except Exception:  # noqa: BLE001 - missing or unreadable: score falls back to GCP extent
        return None


# -- answer key ------------------------------------------------------------------------

def answer_key_path(page: Page) -> Path | None:
    """The owner's .points file for a 1948/1960 sheet (ruling R9), or None. A page cut from a
    two-page photo (page.other set) uses the key for that page of the photo,
    "<photo>.just-<sheet>.points" (1948 sheet 2: "2-and-3.jpg.just-2.points"); its pixels are
    the photo's, which _answer_key_gcps shifts by the crop origin."""
    if page.series not in ("1948", "1960") or page.page_index is not None:
        return None
    a = config.RAW_DIR / page.series / "aligned"
    names = ((f"{page.source.name}.just-{page.sheet_id}.points",) if page.other is not None else
             (f"{page.sheet_id}.jpg.points", f"{page.sheet_id}.pdf.points"))
    for name in names:
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
    box = pages.crop_box(page)
    if box is not None:  # key pixels are the whole photo's; page.png is the crop
        gcps = [replace(g, px=g.px - box[0], py=g.py - box[1]) for g in gcps]
    return gcps if sum(g.enabled for g in gcps) >= 3 else None


# -- review ----------------------------------------------------------------------------

def smoothing_for(gcps: list[GCP]) -> float:
    """TPS smoothing for a reviewed GCP set, by density: config.TPS_SMOOTHING when at least
    config.REVIEW_DENSE_MIN_GCPS are enabled (a review seeded from the dense automatic
    transform), else config.MANUAL_TPS_SMOOTHING (sparse hand-placed points)."""
    n = sum(1 for g in gcps if g.enabled)
    return config.TPS_SMOOTHING if n >= config.REVIEW_DENSE_MIN_GCPS else config.MANUAL_TPS_SMOOTHING


def review_smoothing(review: dict) -> float:
    """The smoothing stored in review.json (the editor stores the one it previewed with),
    else smoothing_for its GCPs."""
    v = review.get("smoothing")
    if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v > 0:
        return float(v)
    return smoothing_for([GCP(**g) for g in review["gcps"]])


def review_transform(review: dict) -> SheetTransform:
    return SheetTransform([GCP(**g) for g in review["gcps"]], review_smoothing(review))


def _review_applied(review: dict, tj: Path, cj: Path) -> bool:
    """transform.json (and changes.geojson) hold exactly this review: same GCPs, same
    smoothing, same change areas. Compared by content, not mtime, so a transform.json
    replaced by `pull` or by an automatic run is never mistaken for the review."""
    try:
        tdata = json.loads(tj.read_text())
    except (OSError, ValueError):
        return False
    rt = review_transform(review)
    if tdata.get("origin") != "review" or tdata.get("smoothing") != rt.smoothing:
        return False
    if tdata.get("gcps") != rt.to_json()["gcps"]:
        return False
    ca = review.get("change_areas")
    if ca is None:
        return not cj.exists()
    try:
        return json.loads(cj.read_text()) == ca
    except (OSError, ValueError):
        return False


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


def _apply_change_areas(res: RefineResult, change_areas: dict | None) -> None:
    """Relabel scored matches whose modern corner lies inside a reviewer-drawn change area
    as real_change (counted, never scored). change_areas is a GeoJSON FeatureCollection in
    EPSG:2227 (the format of changes.geojson)."""
    if not change_areas or not change_areas.get("features"):
        return
    from shapely import contains_xy
    from shapely.geometry import shape
    from shapely.ops import unary_union
    area = unary_union([shape(f["geometry"]) for f in change_areas["features"] if f.get("geometry")])
    if area.is_empty:
        return
    for m in res.matches:
        if m.label in ("inlier", "outlier") and np.isfinite([m.wx, m.wy]).all() \
                and contains_xy(area, m.wx, m.wy):
            m.label = "real_change"


def score_review(t: SheetTransform, refine_json: Path, ref: Reference | None,
                 change_areas: dict | None = None, labels_json: Path | None = None) -> SheetScore:
    """Score of a human-edited transform `t`: refine.json's matches re-measured under `t`
    (else one inlier per GCP), reviewer change areas excluded. Used by run_sheet's review
    branch and by the editor's preview, so both always agree."""
    res = _review_result(t, refine_json, ref)
    _apply_change_areas(res, change_areas)
    agreement = None
    if labels_json is not None and Path(labels_json).exists() and ref is not None:
        from sfzmaps import placement
        agreement = placement.label_agreement(t, reading.load_labels(labels_json), ref)
    return score_mod.score(res, agreement=agreement,
                           shape=page_shape(Path(refine_json).parent / "page.png"),
                           holdout_fit_gcps=True)


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
            old = json.loads(sj.read_text())
            # old-format score: rescored below, never reused; an error or skip score
            # describes a run that did not finish, so the sheet is rerun and rescored
            if not _score_current(old) or str(old.get("reason") or "").startswith(("error:", "skipped:")):
                sj.unlink()
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
    def png_sha() -> str:
        if "sha" not in cache:
            cache["sha"] = _file_sha256(png)
        return cache["sha"]

    def read_stage(name: str, out: Path, inputs: list[Path], fn, own_sha_only: bool = True) -> None:
        """A stage that calls the vision model (classify, labels). Fresh by mtime as usual,
        or, when page.png was re-rendered (newer mtime), by content: the page_sha256 stored
        in `out` (labels.json: else page.json's) equals page.png's. Then no call is made, so
        a byte-identical render on another machine re-reads nothing. A fresh `out` without
        the hash (an older file) gets it written in, keeping its mtime."""
        if not force and out.exists():
            if _fresh(out, inputs):
                if reading.stored_page_sha256(out) is None:
                    _add_page_sha(out, png_sha())
                last_timings[name] = None
                log.info("%-8s skipped (fresh)", name)
                return
            stored = reading.stored_page_sha256(out)
            if stored is None and not own_sha_only:
                stored = reading.stored_page_sha256(pj)
            if stored is not None and stored == png_sha() and all(i.exists() for i in inputs):
                last_timings[name] = None
                log.info("%-8s skipped (page.png content unchanged)", name)
                return
        t0 = time.monotonic()
        sha = png_sha()
        reading.save(out, fn(), page_sha256=sha)
        last_timings[name] = time.monotonic() - t0
        log.info("%-8s done in %.1fs", name, last_timings[name])

    read_stage("classify", pj, [png], lambda: reading.classify_page(png))
    if info().kind != "map":
        log.info("skipped: %s", info().kind)
        sc = SheetScore.empty(f"skipped: {info().kind}")
        clear_outputs(d, sc)  # a sheet reclassified as non-map must not stay published
        return sc
    if stopped("classify"):
        return partial("until: classify")

    # labels / blocks
    read_stage("labels", lj, [png, pj],
               lambda: reading.read_labels(png, scale=config.label_scale(page.series), skip=mask()),
               own_sha_only=False)
    if stopped("labels"):
        return partial("until: labels")

    def do_blocks():
        polys = linework.extract_blocks(img(), exclude=mask())
        # Only corners whose outline evidence is clean (linework.corner_quality) are
        # matched and scored; noisy ones (district lines or lettering merged into the
        # outline, short jogs, odd angles) carry extraction error, not alignment error.
        corners = np.asarray(linework.good_corners_px(polys), float).reshape(-1, 2)
        _write_json(bj, {"type": "FeatureCollection",
                         "features": [{"type": "Feature", "properties": {}, "geometry": mapping(p)}
                                      for p in polys],
                         "corners": [[_num(x), _num(y)] for x, y in corners]})

    # linework.py: a code change (extraction, corner quality) re-extracts, like warp.py
    stage("blocks", bj, [png, pj, Path(linework.__file__)], do_blocks)
    if stopped("blocks"):
        return partial("until: blocks")

    review = None if force_auto else _load_review(d)
    answer_key = False
    result: RefineResult | None = None

    if review is not None:
        rt = review_transform(review)

        def do_review():
            _write_json(tj, _transform_json(rt, "review"))
            if review.get("change_areas") is not None:
                _write_json(cj, review["change_areas"])
            else:
                cj.unlink(missing_ok=True)  # no stale auto changes next to a reviewed transform

        if force or not _review_applied(review, tj, cj):
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
                kt = _transform_json(SheetTransform(gcps, config.MANUAL_TPS_SMOOTHING), "answer_key")
                cur = None
                try:
                    cur = json.loads(tj.read_text()) if tj.exists() else None
                except ValueError:
                    pass
                if force or cur != kt:  # by content: a changed key (or reader) rewrites it
                    _write_json(tj, kt)
                    log.info("fallback transform from answer key (%d GCPs)", len(gcps))
            else:
                tj.unlink(missing_ok=True)
                wt.unlink(missing_ok=True)  # a warp of an older transform must not survive
                sj.unlink(missing_ok=True)  # nor a score of it: rescored as insufficient_anchors
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
            sc = score_review(rt_, rfj, ref, review.get("change_areas"), lj)
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
            sc = score_mod.score(res, agreement=agreement(t_), shape=page_shape(png))
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

    stage("score", sj, [p for p in (tj, rfj, lj, rj) if p.exists()], do_score)
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


def clear_outputs(d: Path, sc: SheetScore) -> None:
    """Record sc (an error or skip) as the sheet's score.json and remove its transform and
    warp, so nothing from an earlier successful run is published for it. run_sheet discards
    such a score on its next run."""
    for p in (d / "transform.json", d / "warped.tif", d / "warped.footprint.geojson"):
        p.unlink(missing_ok=True)
    if d.is_dir():
        _write_json(d / "score.json", _score_json(sc))


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
            try:
                clear_outputs(config.WORK_DIR / p.series / p.sheet_id, results[p.sheet_id])
            except OSError:
                log.exception("could not clear outputs of %s/%s", p.series, p.sheet_id)
    print(format_table(results))
    if logfile is not None:
        print(f"errors logged to {logfile}")
    return results
