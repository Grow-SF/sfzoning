"""Which side of the street a printed code speaks for.  A code printed on street ground sits between the lots on the
two sides; the solver reads it for the lot under its text (`lab_lot`, side A) and, at a lower trust, for the nearest
lot across the street (side B).

For each print: the nearest street centreline (`data/ref/streets.parquet`), whether that centreline is a freeway or
ramp, and side B's lot.  The side of a lot is the side of the centreline its nearest point to the print lies on,
measured along the centreline's normal (its direction from SIDE_STEP_FT before the print's foot to SIDE_STEP_FT
after).  Side B is the nearest lot other than A on the side opposite A's, searched within FAR_SEARCH_FT of the print,
the reach doubling up to FAR_SEARCH_LIMIT_FT.

Four rates say how often the table's code at each side is the print's top reading (its most trusted, from
readings_{q}.npz), for ordinary streets and for freeways.  They are fitted once per held-out quarter q against the
April 2000 table as downloaded (`table.known_and_codes`; the by-eye corrections are never read).  A print counts
for side A's rate when the lot under its text is outside q, and for side B's when the lot across is: no fit reads
the table's code of a lot in its own quarter.

Writes, under data/solver/labels/:
  streetside_{q}.npz  for each quarter q (nw, ne, sw, se), per print (prints.parquet's rows):
                        lotB  int64, side B's lot (-1 for none: no lot under the text, or none across within reach)
                        onst  bool, the print is on street ground (prints.parquet's `street`)
                        isfw  bool, its nearest centreline is a freeway or ramp
                      and the rates fitted with q held out (float, 0 where no print counts):
                        rA_st, rB_st  ordinary streets: the share of prints whose top reading is the code of side A,
                                      and of side B
                        rA_fw, rB_fw  the same on freeways and ramps
                      The joint solver (`scripts/joint_solver/labels.py`) copies a reading of a print on an ordinary
                      street (onst, not isfw, lotB >= 0) to side B at trust x min(rB_st / rA_st, 1).
"""
import re

import geopandas as gpd
import numpy as np
import shapely

from sfzmaps import config
from sfzmaps.solver import paths
from sfzmaps.solver.lots import load_lots
from sfzmaps.solver.trust import load_prints

FREEWAY_TYPES = ("HWY", "RAMP")          # street types that are freeways
FREEWAY_NAME = re.compile(r"^(HWY|I-)\s*\d")   # and names: a numbered highway or interstate
SIDE_STEP_FT = 5.0                       # the centreline's direction at the print's foot: from 5 ft before to 5 ft after
FAR_SEARCH_FT = 100.0                    # side B is searched within this reach of the print,
FAR_SEARCH_LIMIT_FT = 3200.0             # doubling up to this
STREETS_FILE = config.DATA_DIR / "ref" / "streets.parquet"


def freeways(streets: gpd.GeoDataFrame) -> np.ndarray:
    """Whether each centreline is a freeway or ramp (bool): its `st_type` is HWY or RAMP, or its `streetname` is a
    numbered highway or interstate (FREEWAY_NAME)."""
    by_name = np.array([bool(FREEWAY_NAME.match(str(n))) for n in streets["streetname"].values], bool)
    return streets["st_type"].isin(FREEWAY_TYPES).to_numpy(bool) | by_name


def far_side(xy: np.ndarray, under: np.ndarray, lots, street_lines) -> tuple[np.ndarray, np.ndarray]:
    """Side B of each print.  `xy` (n, 2): the prints, map ft; `under`: the lot under each print's text (-1 for
    none); `lots`: every lot's geometry, by lot index; `street_lines`: the street centrelines.  Returns `far` (int64
    per print: side B's lot, -1 for none) and `nearest` (int64: the nearest centreline)."""
    xy = np.asarray(xy, float)
    lots = np.asarray(lots, dtype=object)
    street_lines = np.asarray(street_lines, dtype=object)
    points = shapely.points(xy)
    nearest = shapely.STRtree(street_lines).nearest(points).astype(np.int64)
    lot_tree = shapely.STRtree(lots)
    far = np.full(len(xy), -1, np.int64)
    for j in np.flatnonzero(under >= 0):
        p, a = points[j], int(under[j])
        line = street_lines[nearest[j]]
        s = shapely.line_locate_point(line, p)
        foot = shapely.get_coordinates(shapely.line_interpolate_point(line, s))[0]
        ahead = shapely.get_coordinates(shapely.line_interpolate_point(line, min(s + SIDE_STEP_FT,
                                                                                 shapely.length(line))))[0]
        behind = shapely.get_coordinates(shapely.line_interpolate_point(line, max(s - SIDE_STEP_FT, 0)))[0]
        t = ahead - behind
        normal = np.array([-t[1], t[0]]) / max(np.hypot(*t), 1e-9)

        def side(lot, p=p, foot=foot, normal=normal):
            """The side of the centreline (+1, -1, 0 on it) the lot's nearest point to the print lies on."""
            return np.sign((shapely.get_coordinates(shapely.shortest_line(p, lots[lot]))[1] - foot) @ normal)

        side_a = side(a) if shapely.distance(p, lots[a]) > 0 else np.sign((xy[j] - foot) @ normal)
        reach = FAR_SEARCH_FT
        while far[j] < 0 and reach <= FAR_SEARCH_LIMIT_FT:
            near = lot_tree.query(p, predicate="dwithin", distance=reach)
            for c in near[np.argsort(shapely.distance(p, lots[near]), kind="stable")]:
                if c != a and side(c) == -side_a:
                    far[j] = c
                    break
            reach *= 2
    return far, nearest


def top_codes(names, print_index, code_index, trust, prints: int) -> np.ndarray:
    """Each print's top reading (str, "" for a print with none): the code of its most trusted reading, the first
    such in row order on a tie.  The arguments are readings_{q}.npz's arrays; `prints`: how many prints."""
    names = np.asarray(names).astype(str)
    best = np.full(prints, -1, np.int64)
    best_trust = np.zeros(prints)
    for j, k, t in zip(print_index, code_index, trust):
        if t > best_trust[j]:
            best[j], best_trust[j] = k, t
    return np.where(best >= 0, names[np.maximum(best, 0)], "")


def rates(top, under, far, street, freeway, codes, known, quarter, q: str) -> dict[str, float]:
    """The four rates of the fit that holds out quarter q.  Per print: `top` (its top reading, "" for none), `under`
    (side A's lot), `far` (side B's lot, -1 for none), `street` (on street ground) and `freeway` (its nearest
    centreline is one); per lot: `codes` (the table's), `known` and `quarter` (nw 0, ne 1, sw 2, se 3).  A print
    counts for a side when it has a top reading, is on street ground, and that side's lot has a known code and is
    outside q; the rate is the share of those prints whose top reading is that lot's code (0 for none)."""
    qi = paths.QUARTERS.index(q)
    top, under, far = np.asarray(top).astype(str), np.asarray(under), np.asarray(far)
    has = street & (under >= 0) & (top != "")
    out = {}
    for tag, kind in (("st", ~freeway), ("fw", freeway)):
        for name, lot in (("A", under), ("B", far)):
            at = np.maximum(lot, 0)
            counts = has & kind & (lot >= 0) & known[at] & (quarter[at] != qi)
            out[f"r{name}_{tag}"] = float(np.mean(codes[at][counts] == top[counts])) if counts.any() else 0.0
    return {k: out[k] for k in ("rA_st", "rB_st", "rA_fw", "rB_fw")}


def build_streetsides() -> dict:
    """Find each print's side B and fit the rates once per held-out quarter; write streetside_{q}.npz into
    LABELS_DIR (after readings_{q}.npz, which gives each print's top reading).  Returns the rates by quarter and
    how many prints are on street ground, by freeways and have a side B."""
    found = load_prints()
    city = load_lots()
    lots = gpd.read_parquet(paths.LOTS_DIR / "table.parquet").geometry.values
    streets = gpd.read_parquet(STREETS_FILE)
    xy = found[["x", "y"]].to_numpy(float)
    under = found["lab_lot"].to_numpy(np.int64)
    street = found["street"].to_numpy(bool)
    far, nearest = far_side(xy, under, lots, streets.geometry.values)
    freeway = freeways(streets)[nearest]
    summary = {"on_street": int(street.sum()), "by_freeway": int((street & freeway).sum()),
               "with_side_b": int((far >= 0).sum()), "rates": {}}
    for q in paths.QUARTERS:
        r = np.load(paths.LABELS_DIR / f"readings_{q}.npz")
        top = top_codes(r["names"], r["print_index"], r["code_index"], r["trust"], len(found))
        fitted = rates(top, under, far, street, freeway, city.codes, city.known, city.quarter, q)
        np.savez_compressed(paths.LABELS_DIR / f"streetside_{q}.npz", lotB=far, onst=street, isfw=freeway, **fitted)
        summary["rates"][q] = {k: round(v, 4) for k, v in fitted.items()}
    return summary
