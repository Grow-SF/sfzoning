"""Paths, CRS, thresholds, model id and credit strings."""
import json
import os
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("SFZMAPS_DATA_DIR") or PIPELINE_DIR / "data")
RAW_DIR = DATA_DIR / "raw"
REF_DIR = DATA_DIR / "ref"
CACHE_DIR = DATA_DIR / "cache"
WORK_DIR = DATA_DIR / "work"
OUT_DIR = DATA_DIR / "out"

WORK_CRS = "EPSG:2227"
TOLERANCE_FT = 6.0  # kept at 6 after Task 13 tuning (see the tuning notes)
PASS_FRACTION = 0.95  # superseded as the pass rule (owner, 2026-10-29); TOLERANCE_FT stays for display
# Pass rule (owner, 2026-10-29): on held-out blocks, median <= PASS_MEDIAN_FT and
# p95 <= PASS_P95_FT, and the sheet is not misplaced (sfzmaps.score).
PASS_MEDIAN_FT = 5.0
PASS_P95_FT = 12.0
# Misplacement detector (sfzmaps.placement / score). A sheet is misplaced when
#  * fewer than MISPLACED_MIN_INLIER_FRAC of its corners are inliers (geometry alone); or
#  * fewer than MISPLACED_MAX_LABEL_AGREEMENT of its "local" labels (within
#    MISPLACED_LOCAL_FT of their named block/street) land on it, over at least
#    MISPLACED_MIN_LABELS such labels: a shift of a few blocks; or
#  * fewer than MISPLACED_MAX_LABEL_AGREEMENT of ALL its checkable labels land on their
#    feature (at least MISPLACED_MIN_LABELS) AND fewer than MISPLACED_WEAK_INLIER_FRAC of
#    its corners are inliers. Label failure alone is not enough: renumbered blocks and
#    renamed streets (1921) fail it on correctly placed sheets (Task 13c: correctly placed
#    1921 sheets have inlier shares 0.30-0.41 at agreement 0.14-0.18; the misplaced
#    1948/10 and 1921 section 11 have 0.02-0.03).
MISPLACED_MAX_LABEL_AGREEMENT = 0.2
MISPLACED_MIN_LABELS = 20
MISPLACED_MIN_INLIER_FRAC = 0.05
MISPLACED_WEAK_INLIER_FRAC = 0.15
MISPLACED_LOCAL_FT = 1000.0
MAX_REAL_CHANGE_FRAC = 0.15  # more real change than this share of scored+real_change: review
ROUGH_TOL_FT = 150.0  # rough fit (Task 8) error budget; caps the refine translation vote
# Ridge in ft^2 on the normalized TPS (units in sfzmaps.transform) of the final sheet
# warp. Task 13 tuned it by block-held-out cross-validation on the 1948/1960 sheets: 1.0 was
# too stiff to follow real paper/drafting distortion; 0.01-0.1 predicts unseen blocks best.
# 0.1 keeps nearly all of that gain while a whole block that really moved 15 ft pulls the
# warp only ~5 ft (0.03: ~8 ft), so history is not warped away. Notes:
# docs/superpowers/notes/2026-10-29-tuning.md.
TPS_SMOOTHING = 0.1
# Stiff smoothing for coarse fits: the rough anchor fit (label anchors are only good to
# ~100 ft) and refine's widest-radius stage (divided by 10 per stage, never below
# TPS_SMOOTHING). Independent of TPS_SMOOTHING so tuning the final warp cannot loosen them.
COARSE_TPS_SMOOTHING = 1000.0
# Smoothing for sparse hand-placed control points (review/editor GCPs, the answer-key
# fallback: 8-40 points). TPS_SMOOTHING is tuned for 600-2500 dense corners and would nearly
# interpolate a sparse set (a 40 ft misclick moves a 10x10 grid 16-27 ft at 0.1, 5.5-14 ft
# at 1.0, the Task 7 value).
MANUAL_TPS_SMOOTHING = 1.0

VISION_PROVIDER = "openai"
VISION_MODEL = "gpt-6.1-sol"
VISION_REASONING_EFFORT = "low"  # lowest level documented for this model (low..max)
VISION_CONCURRENCY = 8

ROOT_FOLDER_ID = "10p74RwqmzeQFBrBqjxXdmEOUREx0BLgO"

RUMSEY_CREDIT = "© Cartography Associates, David Rumsey Map Collection"

# Per-series scale for label reading (calibrated in Task 5); default 0.5.
LABEL_SCALE: dict[str, float] = {}


def _scale_file() -> Path:
    return WORK_DIR / "label_scale.json"


def _load_scales() -> dict[str, float]:
    try:
        d = json.loads(_scale_file().read_text())
        return {str(k): float(v) for k, v in d.items()} if isinstance(d, dict) else {}
    except (OSError, ValueError, TypeError):
        return {}


def label_scale(series: str) -> float:
    """Calibrated scale (WORK_DIR/label_scale.json) over LABEL_SCALE over the 0.5 default."""
    return {**LABEL_SCALE, **_load_scales()}.get(series, 0.5)


def save_label_scale(series: str, scale: float) -> None:
    path = _scale_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {**_load_scales(), series: scale}
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1, sort_keys=True))
    tmp.replace(path)


# Vision provider parameters (provider-specific; change when the client is swapped).
# gpt-6.1-sol is not listed in the images guide's detail table; the closest listed sibling,
# gpt-5.6-sol, has detail "high" = fit 2048x2048 and <= 2,500 32-px patches, 1.2 tokens/patch
# (same as gpt-6-astra). Verify with live usage. Source:
# https://developers.openai.com/api/docs/guides/images-vision
VISION_TILE = 768
VISION_OVERLAP = 96
VISION_MAX_EDGE = 2048
VISION_THUMB = 1024
PATCH = 32
PATCH_BUDGET = 2500
PATCH_MULTIPLIER = 1.2
# gpt-6.1-sol: $2/M in, $10/M out (https://developers.openai.com/api/docs/models/gpt-6.1-sol)
PRICE_IN_PER_M = 2.0
PRICE_OUT_PER_M = 10.0
EST_OUT_TOKENS = 800


def estimate_image_tokens(w: int, h: int) -> int:
    """Billable input tokens for one high-detail image (OpenAI patch formula)."""
    import math

    if max(w, h) > VISION_MAX_EDGE:  # fit within the 2048 box first
        f = VISION_MAX_EDGE / max(w, h)
        w, h = int(w * f), int(h * f)
    patches = math.ceil(w / PATCH) * math.ceil(h / PATCH)
    if patches > PATCH_BUDGET:
        shrink = math.sqrt(PATCH * PATCH * PATCH_BUDGET / (w * h))
        sw, sh = w * shrink / PATCH, h * shrink / PATCH
        shrink *= min(math.floor(sw) / sw, math.floor(sh) / sh)
        rw, rh = int(w * shrink), int(h * shrink)
        patches = math.ceil(rw / PATCH) * math.ceil(rh / PATCH)
    return math.ceil(patches * PATCH_MULTIPLIER)

# score.json format/logic version: bump whenever pass or answer-key-fallback logic changes;
# a score.json with a missing or lower version is stale and rescored.
SCORE_VERSION = 3  # 3: Task 13c misplacement rule (label failure needs weak geometry)
