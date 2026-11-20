"""Issue #71: review pictures of the hand-drawn map-area labels, one PNG per series.

Each row: the page (outside the outline tinted red, the outline in magenta) and a 1:1 close-up of every vertex.

    uv run python scripts/spikes/issue71_label_review.py
"""

import json
from pathlib import Path

from PIL import Image, ImageDraw

Image.MAX_IMAGE_PIXELS = None
ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "data" / "work"
LABELS = ROOT / "inputs" / "frame-labels"
OUT = ROOT / "data" / "research" / "issue71" / "label_review"
PAGE = 560  # longest side of the page picture
HALF = 110  # close-up half-size, page px (shown at 2x)


def sample_points(polys, W, H):
    """The outline's points nearest the four page corners and nearest the middle of each page side."""
    import numpy as np

    pts = []
    for poly in polys:
        P = np.array(poly, float)
        n = len(P)
        dense = np.concatenate([np.linspace(P[i], P[(i + 1) % n], 50, endpoint=False) for i in range(n)])
        for tx, ty in [(0, 0), (W, 0), (W, H), (0, H), (W / 2, 0), (W, H / 2), (W / 2, H), (0, H / 2)]:
            k = np.argmin((dense[:, 0] - tx) ** 2 + (dense[:, 1] - ty) ** 2)
            pts.append(tuple(np.round(dense[k]).astype(int)))
    return pts


def row(series: str, sheet: str, lab: dict) -> Image.Image:
    im = Image.open(WORK / series / sheet / "page.png").convert("RGB")
    W, H = im.size
    f = PAGE / max(W, H)
    sm = im.resize((round(W * f), round(H * f)), Image.LANCZOS)
    mask = Image.new("L", sm.size, 0)
    md = ImageDraw.Draw(mask)
    for poly in lab["polygons"]:
        md.polygon([(x * f, y * f) for x, y in poly], fill=255)
    tint = Image.new("RGB", sm.size, (255, 120, 120))
    outside = Image.blend(sm, tint, 0.45)
    sm = Image.composite(sm, outside, mask)
    d = ImageDraw.Draw(sm)
    for poly in lab["polygons"]:
        pts = [(x * f, y * f) for x, y in poly]
        d.line(pts + [pts[0]], fill=(200, 0, 160), width=2)
    zooms = []
    for x, y in sample_points(lab["polygons"], W, H):
        c = im.crop((x - HALF, y - HALF, x + HALF, y + HALF)).resize((4 * HALF, 4 * HALF), Image.NEAREST)
        zd = ImageDraw.Draw(c)
        for poly in lab["polygons"]:
            pts = [(2 * HALF + 2 * (px - x), 2 * HALF + 2 * (py - y)) for px, py in poly]
            zd.line(pts + [pts[0]], fill=(230, 0, 160), width=2)
        ImageDraw.Draw(c).rectangle([0, 0, c.width - 1, c.height - 1], outline=(120, 120, 120))
        zooms.append(c)
    zs = 4 * HALF
    per_row = 4
    zrows = (len(zooms) + per_row - 1) // per_row
    width = sm.width + 10 + per_row * (zs + 12)
    height = max(sm.height, zrows * (zs + 12)) + 24
    R = Image.new("RGB", (width, height), "white")
    R.paste(sm, (0, 24))
    for k, z in enumerate(zooms):
        R.paste(z, (sm.width + 10 + (k % per_row) * (zs + 12), 24 + (k // per_row) * (zs + 12)))
    ImageDraw.Draw(R).text((4, 4), f"{series} / {sheet}   {lab.get('notes', '')[:160]}", fill=(0, 0, 0))
    return R


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for sdir in sorted(p for p in LABELS.iterdir() if p.is_dir()):
        rows = [row(sdir.name, j.stem, json.load(open(j))) for j in sorted(sdir.glob("*.json"))]
        if not rows:
            continue
        W = max(r.width for r in rows)
        S = Image.new("RGB", (W, sum(r.height + 12 for r in rows)), (235, 235, 235))
        y = 0
        for r in rows:
            S.paste(r, (0, y))
            y += r.height + 12
        out = OUT / f"{sdir.name}.png"
        S.save(out)
        print(out, len(rows))


if __name__ == "__main__":
    main()
