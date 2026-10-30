"""Shared helpers for the tuning scripts: output dir and a cached DataSF reference."""
import pickle
from pathlib import Path

from sfzmaps import config

OUT = config.OUT_DIR / "tuning"


def out_dir() -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    return OUT


def blocks_corners():
    """(blocks, corners) GeoDataFrames, cached in OUT/ref.pkl (delete it after `sfzmaps refdata`)."""
    cache = out_dir() / "ref.pkl"
    if cache.exists():
        return pickle.loads(cache.read_bytes())
    from sfzmaps.sources import datasf
    ref = datasf.load_reference()
    cache.write_bytes(pickle.dumps((ref.blocks, ref.corners)))
    return ref.blocks, ref.corners
