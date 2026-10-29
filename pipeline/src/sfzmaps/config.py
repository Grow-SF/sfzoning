"""Paths, CRS, thresholds, model id and credit strings."""
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


def label_scale(series: str) -> float:
    return LABEL_SCALE.get(series, 0.5)


# Vision provider parameters (provider-specific; change when the client is swapped).
VISION_TILE = 1568
VISION_OVERLAP = 160
VISION_MAX_EDGE = 2576
VISION_THUMB = 1024
PRICE_IN_PER_M = 4.0
PRICE_OUT_PER_M = 20.0


def estimate_image_tokens(w: int, h: int) -> int:
    return round(w * h / 750)
