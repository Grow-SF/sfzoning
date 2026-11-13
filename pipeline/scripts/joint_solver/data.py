"""Where the data is, and the city's lots.

Everything lives in one folder, SFZ_RND (default pipeline/data/rnd, git-ignored).  Inputs, made by the earlier R&D
chain:
  zon2000/april2000.parquet       the April 2000 table (DataSF 'Historical Zoning Districts'): a code on every lot
  zon2000/truth_grid.npz          the lots painted on a 4-ft grid ('lot'), every pixel grown to its nearest lot
                                  ('grown'), and every pair of neighbouring lots ('pairs': a, b, touching px, street px)
  zon2000/lotcode_run.npz         'scored': the lots the scores count
  zon2000/zodds_Q1x{q}.npz        'lo_ab': per pair, the log odds that a printed district line runs between them,
                                  read from the ink, fitted without quarter q
  zon2000/lotclf6_cwfwx{q}.npz    every reading of every printed label (label, code, trust), print set of holdout q
  zon2000/labread2_cwfwx{q}.npz   each label's position, sheet and the lot under its text
  zon2000/labside_cwfwx{q}.npz    labels printed in a street: the lot across the street and how much to trust it
  zon2000/zstrokes_old.npz        pen strokes on the page (leader lines), for placement
  zon2000/zlotchar2.npz           what a lot's size says about each code, for placement
  k8/judge2/final2.csv            lots where the map was judged by eye to print another code than the table's
  k8/city/footprint.geojson       the mapped area (sets the grid's origin)
  k8/MALLP_c1/raw.png             the merged raw scan, 1 px = 4 ft, nothing erased
  data_ay2/                       a mirror of the pipeline's data/work for 2000-digital-zoning (sheet transforms)
Outputs go to joint/ in the same folder."""
import csv
import os
from functools import cached_property, lru_cache
from pathlib import Path

import numpy as np

from .codes import canon, tidy

PIPELINE = Path(__file__).resolve().parents[2]
RND = Path(os.environ.get("SFZ_RND", PIPELINE / "data" / "rnd"))
INPUTS = RND / "zon2000"
PAGES = RND / "k8"
OUT = RND / "joint"
RUNS = OUT / "runs"
LABEL_FIXES = PIPELINE / "inputs" / "2000-digital-zoning" / "label_fixes.csv"
QUARTERS = ("nw", "ne", "sw", "se")
FT_PER_PX = 4.0
_PAGE_ORIGIN = (5979300.0, 2124100.0)        # map feet of the merged page's top-left pixel


def wall_odds(q):
    """per pair of neighbouring lots: log odds of a printed district line between them (fitted without quarter q).
    Pairs the ink reader never saw get 10 (a line)."""
    return np.nan_to_num(np.load(INPUTS / f"zodds_Q1x{q}.npz")["lo_ab"].astype(float), nan=10.0)


class City:
    """The lots, their neighbours and the table's codes.  Rasters load on first use."""

    def __init__(self):
        import geopandas as gpd
        import shapely
        self.lots = gpd.read_parquet(INPUTS / "april2000.parquet")
        pairs = np.load(INPUTS / "truth_grid.npz")["pairs"]
        self.n = len(self.lots)
        self.a, self.b = pairs[:, 0].astype(int), pairs[:, 1].astype(int)
        self.border_ft = (pairs[:, 2] + pairs[:, 3]) * FT_PER_PX
        tidied = self.lots["zoning"].map(tidy)
        self.known = ((tidied != "") & ~tidied.str.contains("/")).values
        self.table = self._table_codes(tidied.values)
        self.scored = np.load(INPUTS / "lotcode_run.npz")["scored"]
        self.centroid = shapely.get_coordinates(shapely.centroid(self.lots.geometry.values))
        self.quarter = quarter_of(self.centroid, self.centroid[self.scored])
        x0, y0, _, _ = shapely.from_geojson((PAGES / "city" / "footprint.geojson").read_text()).bounds
        self.origin = (x0 - 40, y0 - 40)

    def _table_codes(self, tidied):
        """the table's code per lot in its one spelling ("" where blank or split), with the by-eye checks applied"""
        code = np.array([canon(c) for c in tidied], dtype=object)
        row = {b: i for i, b in enumerate(self.lots["blklot"].astype(str).values)}
        for r in csv.DictReader(open(PAGES / "judge2" / "final2.csv")):
            if r["verdict"] == "A":
                for b in r["blklots"].split():
                    if b in row:
                        code[row[b]] = canon(r["A"])
        return np.where(self.known, code, "").astype(str)

    @cached_property
    def lot_raster(self):
        return np.load(INPUTS / "truth_grid.npz")["lot"]

    @cached_property
    def grown_raster(self):
        return np.load(INPUTS / "truth_grid.npz")["grown"]

    def to_grid(self, xy):
        """map feet -> (column, row) on the grid"""
        return np.round((np.asarray(xy, float)[..., :2] - self.origin) / FT_PER_PX).astype(np.int64)

    def scan(self):
        """the merged raw scan as ink (True) and where grid (0, 0) falls on it: page row = row0 - grid row,
        page column = grid column - col0"""
        import cv2
        page = cv2.imread(str(PAGES / "MALLP_c1" / "raw.png"), 0)
        col0, row0 = self.to_grid(np.array(_PAGE_ORIGIN))
        return page < 128, int(row0), int(col0)

    def lot_ink(self):
        """share of each lot's own pixels that are inked on the raw scan"""
        ink, row0, col0 = self.scan()
        rr, cc = np.nonzero(self.lot_raster >= 0)
        pr, pc = row0 - rr, cc - col0
        ok = (pr >= 0) & (pr < ink.shape[0]) & (pc >= 0) & (pc < ink.shape[1])
        inked = np.zeros(len(rr), bool)
        inked[ok] = ink[pr[ok], pc[ok]]
        ids = self.lot_raster[rr, cc]
        return np.bincount(ids, inked, minlength=self.n) / np.maximum(np.bincount(ids, minlength=self.n), 1)


@lru_cache(maxsize=1)
def city():
    return City()


def quarter_of(xy, ref):
    """index into QUARTERS for each point: the city splits at the scored lots' median centroid, east-west and
    north-south"""
    xy = np.asarray(xy, float)
    east = xy[:, 0] >= np.median(ref[:, 0])
    north = xy[:, 1] >= np.median(ref[:, 1])
    return np.where(north, np.where(east, 1, 0), np.where(east, 3, 2))


def load_run(name):
    """a run's code per lot, in the one spelling"""
    R = np.load(RUNS / f"{name}.npz", allow_pickle=True)
    return np.array([canon(x) for x in np.asarray(R["names"]).astype(str)])[R["pred"]]


def save_run(path, codes):
    names, pred = np.unique(np.asarray(codes).astype(str), return_inverse=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, pred=pred, names=names)
