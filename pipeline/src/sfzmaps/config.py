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
TOLERANCE_FT = 6.0
PASS_FRACTION = 0.95
TPS_SMOOTHING = 1.0  # ridge in ft^2 on normalized TPS; units and rationale in sfzmaps.transform

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
