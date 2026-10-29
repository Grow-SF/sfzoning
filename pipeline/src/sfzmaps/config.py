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
TPS_SMOOTHING = 50.0

MODEL = "claude-opus-5-5"

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
VISION_TILE = 1568
VISION_OVERLAP = 160
VISION_MAX_EDGE = 2576
VISION_THUMB = 1024
PRICE_IN_PER_M = 4.0
PRICE_OUT_PER_M = 20.0
EST_OUT_TOKENS = 800


def estimate_image_tokens(w: int, h: int) -> int:
    return round(w * h / 750)
