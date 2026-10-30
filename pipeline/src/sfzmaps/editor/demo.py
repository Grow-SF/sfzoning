"""A deliberately corrupted synthetic sheet for the editor (`sfzmaps edit --demo`, tests).

Builds a small grid of modern blocks, renders it as a fake scan, and writes the work-dir
artifacts a finished pipeline run would leave: page.png, page.json, labels.json,
blocks_px.geojson, refine.json (matches against the modern corners), transform.json (a
sparse 5x5 GCP set in which three GCPs are moved ~50 ft), and score.json (not passed).
Dragging the corrupted GCPs back to their true positions fixes the sheet. Everything is
self-contained so the installed package needs no test code.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import geopandas as gpd
import numpy as np
from shapely.geometry import box

from sfzmaps import config, runner
from sfzmaps import score as score_mod
from sfzmaps.pages import Page
from sfzmaps.refine import Match, RefineResult
from sfzmaps.sources.datasf import Reference, block_corners
from sfzmaps.transform import GCP, SheetTransform

SERIES, SHEET = "demo", "1"
PX_PER_FT = 0.3
CORRUPT_FT = 50.0
X0, Y0 = 6_005_000.0, 2_115_000.0
BAD_INDICES = [6, 12, 18]
MARGIN = 80


@dataclass
class Demo:
    page: Page
    ref: Reference
    series: str
    sheet_id: str
    work_dir: Path
    bad_indices: list[int]          # indices into transform.json gcps
    true_gcps: list[GCP]            # the uncorrupted GCPs


def _blocks(n: int = 9) -> gpd.GeoDataFrame:
    rows, geoms = [], []
    for i in range(n):
        for j in range(n):
            rows.append(f"{i * n + j + 1:04d}")
            x, y = X0 + j * 360, Y0 + i * 310
            geoms.append(box(x, y, x + 300, y + 250))
    return gpd.GeoDataFrame({"block_num": rows}, geometry=geoms, crs=config.WORK_CRS)


def build_demo(root: Path, *, set_config: bool = True) -> Demo:
    """Write the demo sheet under root (work dir root/work). With set_config, point
    config.WORK_DIR (and RAW/REF/OUT) at root so the runner and editor use it."""
    root = Path(root)
    work = root / "work"
    if set_config:
        config.WORK_DIR, config.RAW_DIR = work, root / "raw"
        config.REF_DIR, config.OUT_DIR = root / "ref", root / "out"
    d = work / SERIES / SHEET
    d.mkdir(parents=True, exist_ok=True)

    blocks = _blocks()
    corners = block_corners(blocks)
    inter = gpd.GeoDataFrame({"node_cnn": [], "streets": []}, geometry=[], crs=config.WORK_CRS)
    ref = Reference(blocks, corners, inter, None)
    W = np.column_stack([corners.geometry.x, corners.geometry.y])

    # The scan: world -> pixels by a 5 degree rotation, uniform scale and a y flip.
    th = np.radians(5.0)
    rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    centre = W.mean(0)
    q = (W - centre) @ rot.T
    lo, hi = q.min(0), q.max(0)
    w = int(np.ceil((hi[0] - lo[0]) * PX_PER_FT)) + 2 * MARGIN
    h = int(np.ceil((hi[1] - lo[1]) * PX_PER_FT)) + 2 * MARGIN

    def world_to_px(p):
        qq = (np.asarray(p, float) - centre) @ rot.T
        return np.column_stack([(qq[:, 0] - lo[0]) * PX_PER_FT + MARGIN,
                                (hi[1] - qq[:, 1]) * PX_PER_FT + MARGIN])

    def px_to_world(p):
        qq = np.column_stack([(p[:, 0] - MARGIN) / PX_PER_FT + lo[0],
                              hi[1] - (p[:, 1] - MARGIN) / PX_PER_FT])
        return qq @ rot + centre

    img = np.full((h, w), 255, np.uint8)
    for g in blocks.geometry:
        ring = np.round(world_to_px(np.asarray(g.exterior.coords))).astype(np.int32)
        cv2.polylines(img, [ring], True, 0, 3)
    P = world_to_px(W)

    # Sparse GCP lattice over the corners' extent; three of them moved ~50 ft.
    xs = np.linspace(P[:, 0].min(), P[:, 0].max(), 5)
    ys = np.linspace(P[:, 1].min(), P[:, 1].max(), 5)
    lattice = np.array([[x, y] for y in ys for x in xs])
    tw = px_to_world(lattice)
    true_gcps = [GCP(float(p[0]), float(p[1]), float(v[0]), float(v[1]), "anchor")
                 for p, v in zip(lattice, tw)]
    gcps = []
    for i, g in enumerate(true_gcps):
        if i in BAD_INDICES:
            s = 1.0 if i % 2 == 0 else -1.0
            g = GCP(g.px, g.py, g.wx + s * CORRUPT_FT, g.wy - s * CORRUPT_FT / 2, g.source)
        gcps.append(g)
    t = SheetTransform(gcps, config.MANUAL_TPS_SMOOTHING)

    Wt = t.to_world(P)
    ms = [Match(float(p[0]), float(p[1]), float(wv[0]), float(wv[1]),
                float(np.hypot(*(wt - wv))), "inlier", i)
          for i, (p, wv, wt) in enumerate(zip(P, W, Wt))]
    res = RefineResult(t, ms, 1, True, (0.0, 0.0), "ok", 15.0)
    sc = score_mod.score(res)

    source = root / "raw" / SERIES / "demo.jpg"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"demo")
    os.utime(source, ns=(1, 1))
    t0 = time.time_ns() - 60_000_000_000  # a minute ago, so any later write is strictly newer
    steps = [
        (d / "page.png", lambda p: cv2.imwrite(str(p), img)),
        (d / "page.json", lambda p: p.write_text(json.dumps(
            {"kind": "map", "sheet_number": SHEET, "neighbor_refs": [], "mask_boxes": []}))),
        (d / "labels.json", lambda p: p.write_text("[]")),
        (d / "blocks_px.geojson", lambda p: p.write_text(json.dumps(
            {"type": "FeatureCollection", "features": [], "corners": P.tolist()}))),
        (d / "refine.json", lambda p: runner._write_json(p, runner._result_json(res))),
        (d / "transform.json", lambda p: runner._write_json(p, runner._transform_json(t, "auto"))),
        (d / "score.json", lambda p: runner._write_json(p, runner._score_json(sc))),
    ]
    for i, (p, fn) in enumerate(steps):  # strictly increasing mtimes: every stage looks fresh
        fn(p)
        os.utime(p, ns=(t0 + i * 10_000_000, t0 + i * 10_000_000))
    return Demo(Page(SERIES, SHEET, source, None), ref, SERIES, SHEET, work, list(BAD_INDICES), true_gcps)
