"""Synthetic scanned sheets with a known pixel <-> world transform.

``make_sheet`` renders modern blocks (EPSG:2227 feet) as a fake scan:
world -> (rotate, scale, y-flip) -> "undistorted" pixels -> paper distortion
-> sheet pixels.  ``distortion`` maps *sheet* pixel coordinates (N x 2 or
H x W x 2, x right / y down) to *undistorted* pixel coordinates; the identity
``lambda p: p`` gives an undistorted sheet.  Defining it in that direction
lets the image be rendered by exact inverse warping (cv2.remap) and makes
``true_px_to_world`` an exact closed-form composition.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

import cv2
import geopandas as gpd
import numpy as np
from shapely.affinity import translate
from shapely.geometry import MultiPolygon, Point, Polygon

from sfzmaps.reading import Label

MARGIN_PX = 100
MAX_INVERSE_ITERS = 100
OUTLINE_PX = 3
_SHIFT = 4  # cv2 sub-pixel bits
_FONT = cv2.FONT_HERSHEY_SIMPLEX


@dataclass
class SyntheticSheet:
    img: np.ndarray                                   # uint8 grayscale
    true_px_to_world: Callable[[np.ndarray], np.ndarray]
    labels: list[Label]
    world_to_px: Callable[[np.ndarray], np.ndarray] = field(repr=False)


def _parts(g) -> list[Polygon]:
    if isinstance(g, Polygon):
        return [g]
    if isinstance(g, MultiPolygon):
        return list(g.geoms)
    return []


def _fix(pts: np.ndarray) -> np.ndarray:
    return np.round(pts * (1 << _SHIFT)).astype(np.int32)


def make_sheet(blocks: gpd.GeoDataFrame, *, px_per_ft: float, rotation_deg: float,
               distortion: Callable[[np.ndarray], np.ndarray],
               style: Literal["outline", "filled"],
               shift_blocks: dict[str, tuple[float, float]] = {},
               seed: int = 0) -> SyntheticSheet:
    rng = np.random.default_rng(seed)
    geoms = []
    for num, g in zip(blocks["block_num"], blocks.geometry):
        dx, dy = shift_blocks.get(num, (0.0, 0.0))
        geoms.append((str(num), translate(g, dx, dy) if (dx or dy) else g))

    th = np.radians(rotation_deg)
    rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    allpts = np.vstack([np.asarray(p.exterior.coords) for _, g in geoms for p in _parts(g)])
    centre = allpts.mean(0)
    r = (allpts - centre) @ rot.T
    lo, hi = r.min(0), r.max(0)
    w = int(np.ceil((hi[0] - lo[0]) * px_per_ft)) + 2 * MARGIN_PX
    h = int(np.ceil((hi[1] - lo[1]) * px_per_ft)) + 2 * MARGIN_PX
    u0 = MARGIN_PX - lo[0] * px_per_ft
    v0 = MARGIN_PX + hi[1] * px_per_ft

    def world_to_undist(p):
        q = (np.asarray(p, float) - centre) @ rot.T
        return np.stack([q[..., 0] * px_per_ft + u0, -q[..., 1] * px_per_ft + v0], axis=-1)

    def undist_to_world(p):
        p = np.asarray(p, float)
        q = np.stack([(p[..., 0] - u0) / px_per_ft, -(p[..., 1] - v0) / px_per_ft], axis=-1)
        return q @ rot + centre

    def true_px_to_world(p):
        return undist_to_world(distortion(np.asarray(p, float)))

    def world_to_px(p):
        u = world_to_undist(p)
        x = u.copy()
        # Fixed-point inverse of the distortion; converges to machine
        # precision for mild (contractive) distortions.  Used for labels
        # and by tests; the forward true_px_to_world is closed-form.
        for _ in range(MAX_INVERSE_ITERS):
            step = distortion(x) - u
            x = x - step
            if np.abs(step).max() < 1e-9:
                break
        else:
            raise ValueError("distortion inverse did not converge; distortion is not contractive")
        return x

    # Draw the undistorted sheet, then inverse-warp it through the distortion.
    canvas = np.full((h, w), 255, np.uint8)
    ink, paper = (0, 255)
    font_scale = max(0.4, 40 * px_per_ft / 30.0)
    labels: list[Label] = []
    for num, g in geoms:
        for part in _parts(g):
            ring = _fix(world_to_undist(np.asarray(part.exterior.coords)))
            if style == "filled":
                cv2.fillPoly(canvas, [ring], ink, cv2.LINE_8, _SHIFT)
            else:
                cv2.polylines(canvas, [ring], True, ink, OUTLINE_PX, cv2.LINE_8, _SHIFT)
        # Block number at the centroid (white on black when filled) plus up
        # to two random text-like words inside the block.
        txt_col = paper if style == "filled" else ink
        c = g.centroid
        cu = world_to_undist(np.array([c.x, c.y]))
        (tw, tht), _ = cv2.getTextSize(num, _FONT, font_scale, 2)
        cv2.putText(canvas, num, (int(cu[0] - tw / 2), int(cu[1] + tht / 2)), _FONT, font_scale, txt_col, 2)
        cx, cy = world_to_px(np.array([c.x, c.y]))
        labels.append(Label(kind="block_number", text=num, x=int(round(cx)), y=int(round(cy)), confidence=1.0))
        bx = world_to_undist(np.asarray(g.envelope.exterior.coords))
        for _ in range(2):
            t = rng.uniform(0.2, 0.8, 2)
            pos = bx.min(0) + t * (bx.max(0) - bx.min(0))
            if g.contains(Point(*undist_to_world(pos))):
                word = "".join(rng.choice(list("ABCDEFGHKMNRST"), rng.integers(2, 6)))
                cv2.putText(canvas, word, (int(pos[0]), int(pos[1])), _FONT, 0.6 * font_scale, txt_col, 1)

    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
    src = distortion(np.stack([xs, ys], axis=-1))
    img = cv2.remap(canvas, src[..., 0].astype(np.float32), src[..., 1].astype(np.float32),
                    cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    img = np.where(img < 128, 0, 255).astype(np.uint8)

    # Speckle: isolated 1-2 px dots of both polarities.
    n = int(h * w * 2e-4)
    yy, xx = rng.integers(0, h - 2, n), rng.integers(0, w - 2, n)
    s = rng.integers(1, 3, n)
    for y, x, k in zip(yy, xx, s):
        img[y:y + k, x:x + k] = 255 - img[y, x]
    return SyntheticSheet(img=img, true_px_to_world=true_px_to_world, labels=labels, world_to_px=world_to_px)

