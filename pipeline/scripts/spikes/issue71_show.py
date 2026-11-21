"""Issue #71: show held-out predictions — the whole sheet coloured by map probability (blue map, red frame), the label
outline in green, and full-resolution close-ups of the label's corners and side middles, with the scores.

    uv run python scripts/spikes/issue71_show.py SERIES/SHEET [SERIES/SHEET ...] --name NAME
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from sfzmaps.frame import data, look

HALF = 110


def colour(lum, p):
    ink = np.clip(1 - lum, 0, 1)[..., None]
    col = p[..., None] * np.array([44, 123, 182.0]) + (1 - p[..., None]) * np.array([215, 25, 28.0])
    return Image.fromarray((255 * (1 - ink) + col * ink).clip(0, 255).astype(np.uint8))


def sample_points(polys, W, H):
    pts = []
    for poly in polys:
        P = np.array(poly, float)
        n = len(P)
        dense = np.concatenate([np.linspace(P[i], P[(i + 1) % n], 50, endpoint=False) for i in range(n)])
        for tx, ty in [(0, 0), (W, 0), (W, H), (0, H), (W / 2, 0), (W, H / 2), (W / 2, H), (0, H / 2)]:
            k = np.argmin((dense[:, 0] - tx) ** 2 + (dense[:, 1] - ty) ** 2)
            pts.append(tuple(np.round(dense[k]).astype(int)))
    return pts


def row(sid):
    lum = data.load_luminance(data.page_path(*sid))
    pr = np.load(data.OUT / sid[0] / sid[1] / "map_prob.npz")["map"].astype(np.float32)
    sc = json.load(open(data.OUT / sid[0] / sid[1] / "scores.json"))
    polys, holes = data.load_label(*sid), data.label_holes(*sid)
    H, W = lum.shape
    whole = look.sheet_picture(lum, pr, side=700).convert("RGB")
    f = whole.width / W
    d = ImageDraw.Draw(whole)
    for poly in polys + holes:
        q = [(x * f, y * f) for x, y in poly]
        d.line(q + [q[0]], fill=(0, 170, 0), width=1)
    zooms = []
    for x, y in sample_points(polys, W, H):
        x0, y0 = max(0, x - HALF), max(0, y - HALF)
        c = colour(lum[y0:y + HALF, x0:x + HALF], pr[y0:y + HALF, x0:x + HALF]).resize(
            ((min(W, x + HALF) - x0) * 2, (min(H, y + HALF) - y0) * 2), Image.NEAREST)
        zd = ImageDraw.Draw(c)
        for poly in polys + holes:
            q = [(2 * (px - x0), 2 * (py - y0)) for px, py in poly]
            zd.line(q + [q[0]], fill=(0, 170, 0), width=1)
        zooms.append(c)
    zs = 4 * HALF
    R = Image.new("RGB", (whole.width + 10 + 4 * (zs + 8), max(whole.height, 2 * (zs + 8)) + 28), "white")
    R.paste(whole, (0, 28))
    for k, z in enumerate(zooms[:8]):
        R.paste(z, (whole.width + 10 + (k % 4) * (zs + 8), 28 + (k // 4) * (zs + 8)))
    txt = (f"{sid[0]}/{sid[1]}  log-loss: frame band {sc['log_loss_frame_band']:.3f}, map band "
           f"{sc['log_loss_map_band']:.3f}, all ink {sc['log_loss_all_ink']:.3f} | frame ink called map "
           f"{sc['frame_called_map']:.3f} | map ink called frame {sc['map_called_frame']:.3f} | edge (side, MapSeg) "
           f"median {sc['edge_median_px']:.0f} px, HD95 {sc['edge_hd95_px']:.0f} px")
    ImageDraw.Draw(R).text((4, 6), txt, fill=(0, 0, 0))
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sheets", nargs="+")
    ap.add_argument("--name", required=True)
    a = ap.parse_args()
    rows = [row(tuple(s.split("/", 1))) for s in a.sheets]
    W = max(r.width for r in rows)
    S = Image.new("RGB", (W, sum(r.height + 10 for r in rows)), (230, 230, 230))
    y = 0
    for r in rows:
        S.paste(r, (0, y))
        y += r.height + 10
    out = data.OUT / "look" / f"{a.name}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    S.save(out)
    print(out)


if __name__ == "__main__":
    main()
