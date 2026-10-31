"""Cross-edition alignment: place a sheet that failed on its own by matching its drawing to the
same sheet in another edition that passed.

The zoning map books reuse their base sheets from edition to edition (1960-1980 share sections
1-13 and the H height sheets; 1921, 1929 and 1935 share theirs), redrawing district lines and
labels on the same streets. So a failing sheet F and a passing sheet R with the same printed
sheet number share most of their linework.

  1. SIFT features on both pages (downscaled to MATCH_MAX_PX, inside each page's map area),
     Lowe's ratio test, then a RANSAC homography F px -> R px.
  2. Accept it only with at least MIN_INLIERS consistent matches, spread over the page.
  3. Compose with R's transform: world(F px) = T_R(H(F px)), sampled on a dense grid where H
     lands inside R's map area, fit as F's rough transform (origin "crossref").
  4. Rerun F from refine onward: the existing refine snaps it to today's block corners and the
     existing score judges it against the same pass rule as every other sheet.

The rerun is kept only if it beats the sheet's previous result (see better()); otherwise
every file the rerun touched is restored. No vision calls: this is image processing only.
"""
from __future__ import annotations

import json
import logging
import math
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from sfzmaps import config, reading
from sfzmaps.transform import GCP, SheetTransform

log = logging.getLogger(__name__)

MATCH_MAX_PX = 2400       # feature matching on pages downscaled to at most this side
RATIO = 0.75              # Lowe's ratio test
RANSAC_PX = 15.0          # homography reprojection tolerance, full-resolution px
MIN_INLIERS = 150         # fewer consistent matches: not the same plate, or too little shared
MIN_SPREAD = 0.25         # inliers' bounding box must cover this share of the page in x and y
GRID_STEP_PX = 150        # composed GCPs every this many px on the failing page
PLACED_SLACK = 3.0        # an unplaced sheet is kept once placed within this x its pass limits
H_SCALE_RANGE = (0.5, 2.0)   # same plate scanned twice: the homography's scale is near 1
H_MAX_ANISOTROPY = 1.3       # ... and it barely skews (ratio of its singular values)
SCALE_TOLERANCE = 0.3        # composed ft/px within 30% of the reference's ft/px / H scale

# Files a rerun from refine onward can rewrite; restored when the result isn't better.
SHEET_FILES = ["rough.json", "refine_raw.json", "refine.json", "changes.geojson", "transform.json",
               "score.json", "warped.tif", "warped.footprint.geojson", "warped_paper.tif",
               "warped_paper.footprint.geojson"]


@dataclass
class Match:
    ref_series: str
    ref_sheet: str
    homography: np.ndarray    # 3x3, failing-page px -> reference-page px (full resolution)
    inliers: int


def _features(d: Path):
    img = cv2.imread(str(d / "page.png"), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"cannot read {d / 'page.png'}")
    keep = ~reading.exclusion_mask(img, reading.load_page(d / "page.json"))
    f = min(1.0, MATCH_MAX_PX / max(img.shape))
    small = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA) if f < 1 else img
    mask = cv2.resize(keep.astype(np.uint8) * 255, (small.shape[1], small.shape[0]),
                      interpolation=cv2.INTER_NEAREST)
    kp, des = cv2.SIFT_create(nfeatures=20000).detectAndCompute(small, mask)
    pts = np.float32([k.pt for k in kp]) / f if kp else np.zeros((0, 2), np.float32)
    return img.shape, keep, pts, des


def match(failing: Path, reference: Path) -> tuple[np.ndarray, int] | None:
    """Homography failing px -> reference px, and its inlier count; None if not convincing."""
    shape_f, _, pf, df = _features(failing)
    _, _, pr, dr = _features(reference)
    if df is None or dr is None or len(pf) < MIN_INLIERS or len(pr) < MIN_INLIERS:
        return None
    knn = cv2.FlannBasedMatcher({"algorithm": 1, "trees": 5}, {"checks": 64}).knnMatch(df, dr, k=2)
    good = [m for m, n in (p for p in knn if len(p) == 2) if m.distance < RATIO * n.distance]
    if len(good) < MIN_INLIERS:
        return None
    a = pf[[m.queryIdx for m in good]]
    b = pr[[m.trainIdx for m in good]]
    H, inl = cv2.findHomography(a, b, cv2.RANSAC, RANSAC_PX, maxIters=5000, confidence=0.999)
    if H is None:
        return None
    inl = inl.ravel().astype(bool)
    n = int(inl.sum())
    if n < MIN_INLIERS:
        return None
    span = np.ptp(a[inl], axis=0) / np.array([shape_f[1], shape_f[0]], float)
    if (span < MIN_SPREAD).any():
        return None
    if not plausible_homography(H):
        return None
    return H, n


def plausible_homography(H: np.ndarray) -> bool:
    """Same plate scanned twice: near-unit scale, little skew, almost no perspective."""
    sv = np.linalg.svd(H[:2, :2] / H[2, 2], compute_uv=False)
    scale = float(np.sqrt(sv[0] * sv[1]))
    return (H_SCALE_RANGE[0] <= scale <= H_SCALE_RANGE[1] and sv[0] / sv[1] <= H_MAX_ANISOTROPY
            and float(np.abs(H[2, :2] / H[2, 2]).max()) < 1e-4)


def composed_transform(failing: Path, reference: Path, H: np.ndarray) -> SheetTransform | None:
    """F's rough transform: R's transform after H, on a dense grid of F's map area."""
    t_ref = SheetTransform.from_json(json.loads((reference / "transform.json").read_text()))
    img_f = cv2.imread(str(failing / "page.png"), cv2.IMREAD_GRAYSCALE)
    img_r = cv2.imread(str(reference / "page.png"), cv2.IMREAD_GRAYSCALE)
    keep_f = ~reading.exclusion_mask(img_f, reading.load_page(failing / "page.json"))
    keep_r = ~reading.exclusion_mask(img_r, reading.load_page(reference / "page.json"))
    h, w = img_f.shape
    ys, xs = np.mgrid[GRID_STEP_PX // 2:h:GRID_STEP_PX, GRID_STEP_PX // 2:w:GRID_STEP_PX]
    pf = np.column_stack([xs.ravel(), ys.ravel()]).astype(float)
    pf = pf[keep_f[pf[:, 1].astype(int), pf[:, 0].astype(int)]]
    if len(pf) < 10:
        return None
    pr = cv2.perspectiveTransform(pf.reshape(-1, 1, 2), H).reshape(-1, 2)
    hr, wr = img_r.shape
    inside = (pr[:, 0] >= 0) & (pr[:, 0] < wr) & (pr[:, 1] >= 0) & (pr[:, 1] < hr)
    pf, pr = pf[inside], pr[inside]
    ok = keep_r[pr[:, 1].astype(int), pr[:, 0].astype(int)]
    pf, pr = pf[ok], pr[ok]
    if len(pf) < 10:
        return None
    world = t_ref.to_world(pr)
    gcps = [GCP(float(x), float(y), float(wx), float(wy), "auto")
            for (x, y), (wx, wy) in zip(pf, world)]
    t = SheetTransform(gcps, config.TPS_SMOOTHING).fit()
    # The composed scale must agree with the reference's scale through H; anything else is a
    # degenerate fit (it once produced 0.0005 ft/px and a warp that never finished).
    sv = np.linalg.svd(H[:2, :2] / H[2, 2], compute_uv=False)
    expected = t_ref.median_scale_ft_per_px(img_r.shape) * float(np.sqrt(sv[0] * sv[1]))
    got = t.median_scale_ft_per_px(img_f.shape)
    if not (abs(got - expected) <= SCALE_TOLERANCE * expected):
        log.warning("crossref: composed scale %.3g ft/px vs expected %.3g; rejected", got, expected)
        return None
    return t


def better(old, new) -> bool:
    """Keep the cross-edition result? Passing beats failing; a sheet that was never placed is
    kept once placed within PLACED_SLACK x its limits; otherwise a lower held-out median."""
    if new is None or not math.isfinite(new.holdout_p50_ft):
        return False
    if new.passed:
        return not (old is not None and old.passed and old.holdout_p50_ft <= new.holdout_p50_ft)
    if old is not None and old.passed:
        return False
    if old is None or not math.isfinite(old.holdout_p50_ft):
        return (new.holdout_p50_ft <= PLACED_SLACK * new.pass_median_limit_ft
                and new.holdout_p95_ft <= PLACED_SLACK * new.pass_p95_limit_ft)
    return new.holdout_p50_ft < old.holdout_p50_ft


def candidates(series: str, sheet_number: str | None, passed: set[tuple[str, str]],
               numbers: dict[tuple[str, str], str | None]) -> list[tuple[str, str]]:
    """Passing sheets with the same printed number in other series, nearest edition first."""
    if not sheet_number:
        return []
    year = _year(series)
    same = [k for k in passed if k[0] != series and numbers.get(k) == sheet_number]
    return sorted(same, key=lambda k: (abs(_year(k[0]) - year), k))


def _year(series: str) -> int:
    return int(series[:4])


def backup(d: Path) -> Path:
    tmp = Path(tempfile.mkdtemp(prefix="crossref-", dir=d))
    for name in SHEET_FILES:
        if (d / name).exists():
            shutil.copy2(d / name, tmp / name)
    return tmp


def restore(d: Path, saved: Path) -> None:
    for name in SHEET_FILES:
        src = saved / name
        if src.exists():
            shutil.copy2(src, d / name)
        else:
            (d / name).unlink(missing_ok=True)
    shutil.rmtree(saved, ignore_errors=True)
