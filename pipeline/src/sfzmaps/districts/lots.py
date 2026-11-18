"""The lots as they were in a map's year, from DataSF's parcel history.

DataSF's parcels include retired lots with the date each was dropped.  A lot that replaced one or
more retired lots (a merger, a split, a condo conversion's dozens of unit lots on one footprint)
was created when the last of them was dropped; a lot with no recorded predecessor is taken as
always there.  So a lot is valid in year Y when created <= Y < dropped (an active lot: never
dropped).  Drawing each historic map on the lots of its year shows a 2000s merger as the two lots
the old map zoned apart, and a rail yard as the one parcel it was, not today's dozens.

The records only reach back to the late 1980s (almost every retirement is after 2000), so older
changes stay invisible; a reader's correction can still split such a lot by hand.

A predecessor is a retired lot dropped before its successor whose overlap with it covers more
than PRED_SHARE of the smaller of the two.
"""
from __future__ import annotations

import hashlib
import logging

import geopandas as gpd
import numpy as np
import pandas as pd

from sfzmaps import config

log = logging.getLogger(__name__)

PRED_SHARE = 0.5
LIVES = "parcel_lives.parquet"


def _drop_year(p: pd.DataFrame) -> np.ndarray:
    """Fractional year each lot was dropped (inf for active lots)."""
    d = pd.to_datetime(p["date_map_drop"], errors="coerce").fillna(
        pd.to_datetime(p["date_rec_drop"], errors="coerce"))
    y = (d.dt.year + d.dt.day_of_year / 366).to_numpy(dtype=float)
    return np.where(p["active"].to_numpy(dtype=bool) | np.isnan(y), np.inf, y)


def compute_lives(p: gpd.GeoDataFrame) -> pd.DataFrame:
    """blklot, created, dropped (fractional years; -inf / inf when unbounded) for every lot."""
    from shapely import STRtree

    p = p.reset_index(drop=True)
    dropped = _drop_year(p)
    created = np.full(len(p), -np.inf)
    retired = np.flatnonzero(np.isfinite(dropped))
    if len(retired):
        geoms = p.geometry.to_numpy()
        area = p.geometry.area.to_numpy()
        tree = STRtree(geoms[retired])
        a_idx, r_pos = tree.query(geoms, predicate="intersects")
        r_idx = retired[r_pos]
        keep = (a_idx != r_idx) & (dropped[r_idx] < dropped[a_idx])
        a_idx, r_idx = a_idx[keep], r_idx[keep]
        import shapely
        ov = shapely.area(shapely.intersection(geoms[a_idx], geoms[r_idx]))
        pred = ov > PRED_SHARE * np.minimum(area[a_idx], area[r_idx])
        np.maximum.at(created, a_idx[pred], dropped[r_idx[pred]])
    return pd.DataFrame({"blklot": p["blklot"].astype(str).to_numpy(), "created": created, "dropped": dropped})


def _source_sha() -> str:
    return hashlib.sha256((config.REF_DIR / "parcels.parquet").read_bytes()).hexdigest()[:16]


_MEMO: dict[str, pd.DataFrame] = {}


def lives() -> pd.DataFrame:
    """compute_lives for the reference parcels, kept per process and on disk in data/cache/ (not
    a DVC output) keyed by the parcels' sha, so each build stage computes it at most once."""
    sha = _source_sha()
    if sha in _MEMO:
        return _MEMO[sha]
    path = config.DATA_DIR / "cache" / LIVES
    if path.exists():
        cached = pd.read_parquet(path)
        if len(cached) and cached["source_sha"].iloc[0] == sha:
            _MEMO[sha] = cached[["blklot", "created", "dropped"]]
            return _MEMO[sha]
    out = compute_lives(gpd.read_parquet(config.REF_DIR / "parcels.parquet"))
    path.parent.mkdir(parents=True, exist_ok=True)
    import os
    tmp = path.with_suffix(f".{os.getpid()}.tmp")    # build stages run side by side
    out.assign(source_sha=sha).to_parquet(tmp)
    tmp.replace(path)
    log.info("parcel lives: %d lots, %d with a recorded predecessor", len(out), int(np.isfinite(out.created).sum()))
    _MEMO[sha] = out
    return out


def valid_in(table: pd.DataFrame, year: float) -> np.ndarray:
    """Mask of the lots that existed in `year`."""
    return ((table["created"] <= year) & (year < table["dropped"])).to_numpy()


def era_lot_of(year: float) -> dict[str, str]:
    """{lot of another time: the lot of `year` it sits in} for every recorded lot that isn't a lot
    of `year`, by the year's lot covering most of it.  Inputs keyed by today's lots (overrides,
    parks, answer keys) reach a historic map through this."""
    import shapely
    from shapely import STRtree

    src = config.REF_DIR / "parcels.parquet"
    if not src.exists():
        return {}
    p = gpd.read_parquet(src).reset_index(drop=True)
    t = lives().set_index("blklot").reindex(p["blklot"].astype(str)).reset_index()
    ok = valid_in(t, year + 0.5)
    era, other = p[ok].reset_index(drop=True), p[~ok].reset_index(drop=True)
    if not len(other) or not len(era):
        return {}
    eg, og = era.geometry.to_numpy(), other.geometry.to_numpy()
    o_idx, e_idx = STRtree(eg).query(og, predicate="intersects")
    ov = shapely.area(shapely.intersection(og[o_idx], eg[e_idx]))
    best = pd.DataFrame({"o": o_idx, "e": e_idx, "a": ov}).sort_values("a", ascending=False).drop_duplicates("o")
    best = best[best["a"] > 0]
    return dict(zip(other["blklot"].astype(str).to_numpy()[best["o"]],
                    era["blklot"].astype(str).to_numpy()[best["e"]]))
