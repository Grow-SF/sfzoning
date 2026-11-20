"""Issue #71: helper for drawing map-area labels by eye.

Labels live in pipeline/inputs/frame-labels/<series>/<sheet>.json:

    {"polygons": [[[x, y], ...], ...],    # page.png px (full resolution, as stored; x right, y down)
     "notes": "...", "drawn_by": "..."}

Commands (pictures go to data/research/issue71/labels/<series>/<sheet>/):

    overview SERIES SHEET            the whole page, longest side ~1400 px, a labelled grid every GRID page px
    zoom SERIES SHEET X,Y [X,Y ...] [--half N]
                                     the page around each (X, Y), N page px each way (default 120), shown at 1:1 or
                                     larger, ticks every 10 px and labels every 50 px in absolute page px; the
                                     polygon(s) drawn if a label exists; all points in one picture
    check SERIES SHEET               the overview with the polygon(s) drawn, plus one zoom per vertex in a grid
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

Image.MAX_IMAGE_PIXELS = None
ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "data" / "work"
LABELS = ROOT / "inputs" / "frame-labels"
OUT = ROOT / "data" / "research" / "issue71" / "labels"


def page(series: str, sheet: str) -> Image.Image:
    return Image.open(WORK / series / sheet / "page.png").convert("RGB")


def label(series: str, sheet: str):
    p = LABELS / series / f"{sheet}.json"
    return json.load(open(p)) if p.exists() else None


def outdir(series, sheet) -> Path:
    d = OUT / series / sheet
    d.mkdir(parents=True, exist_ok=True)
    return d


def draw_polys(d: ImageDraw.ImageDraw, polys, f, ox=0, oy=0, width=2):
    for poly in polys:
        pts = [((x - ox) * f, (y - oy) * f) for x, y in poly]
        d.line(pts + [pts[0]], fill=(230, 0, 160), width=width)
        for i, (x, y) in enumerate(pts):
            d.ellipse([x - 4, y - 4, x + 4, y + 4], outline=(0, 160, 255), width=2)


def overview(series, sheet, polys=None, name="overview.png"):
    im = page(series, sheet)
    W, H = im.size
    f = 1400 / max(W, H)
    sm = im.resize((round(W * f), round(H * f)), Image.LANCZOS)
    pad = 40
    can = Image.new("RGB", (sm.width + pad, sm.height + pad), "white")
    can.paste(sm, (pad, pad))
    d = ImageDraw.Draw(can)
    grid = 250 if max(W, H) < 4000 else 500
    for x in range(0, W, grid):
        X = pad + x * f
        d.line([(X, pad), (X, can.height)], fill=(0, 200, 0), width=1)
        d.text((X + 2, 2 + (14 if (x // grid) % 2 else 0)), str(x), fill=(0, 120, 0))
    for y in range(0, H, grid):
        Y = pad + y * f
        d.line([(pad, Y), (can.width, Y)], fill=(0, 200, 0), width=1)
        d.text((2, Y + 2), str(y), fill=(0, 120, 0))
    if polys:
        sub = Image.new("RGB", can.size, "white")
        draw_polys(ImageDraw.Draw(can), [[(x + pad / f, y + pad / f) for x, y in p] for p in polys], f)
    out = outdir(series, sheet) / name
    can.save(out)
    print(out, f"page {W}x{H}")
    return out


def zoom_img(im, X, Y, half, polys):
    W, H = im.size
    x0, y0 = max(0, X - half), max(0, Y - half)
    x1, y1 = min(W, X + half), min(H, Y + half)
    crop = im.crop((x0, y0, x1, y1))
    f = max(1, round(480 / (2 * half)))
    crop = crop.resize((crop.width * f, crop.height * f), Image.NEAREST)
    pad = 36
    can = Image.new("RGB", (crop.width + pad, crop.height + pad), "white")
    can.paste(crop, (pad, pad))
    d = ImageDraw.Draw(can)
    for x in range((x0 // 10) * 10, x1, 10):
        if x < x0:
            continue
        X0 = pad + (x - x0) * f
        big = x % 50 == 0
        d.line([(X0, pad - (10 if big else 4)), (X0, pad)], fill=(0, 0, 0))
        if big:
            d.line([(X0, pad), (X0, can.height)], fill=(0, 200, 0))
            d.text((X0 + 1, 2), str(x), fill=(0, 100, 0))
    for y in range((y0 // 10) * 10, y1, 10):
        if y < y0:
            continue
        Y0 = pad + (y - y0) * f
        big = y % 50 == 0
        d.line([(pad - (10 if big else 4), Y0), (pad, Y0)], fill=(0, 0, 0))
        if big:
            d.line([(pad, Y0), (can.width, Y0)], fill=(0, 200, 0))
            d.text((1, Y0 + 1), str(y), fill=(0, 100, 0))
    if polys:
        draw_polys(d, [[(x + pad / f, y + pad / f) for x, y in p] for p in polys], f, x0, y0, width=1)
    return can


def zoom(series, sheet, pts, half=120):
    lab = label(series, sheet)
    im = page(series, sheet)
    tiles = [zoom_img(im, X, Y, half, lab["polygons"] if lab else None) for X, Y in pts]
    out = outdir(series, sheet) / ("zoom_" + "_".join(f"{x}-{y}" for x, y in pts)[:120] + ".png")
    grid(tiles, 2).save(out)
    print(out)


def grid(tiles, cols):
    tw = max(t.width for t in tiles)
    th = max(t.height for t in tiles)
    rows = (len(tiles) + cols - 1) // cols
    S = Image.new("RGB", (cols * tw + (cols - 1) * 8, rows * th + (rows - 1) * 8), (255, 220, 220))
    for i, t in enumerate(tiles):
        S.paste(t, ((i % cols) * (tw + 8), (i // cols) * (th + 8)))
    return S


def check(series, sheet):
    lab = label(series, sheet)
    if not lab:
        sys.exit(f"no label for {series}/{sheet}")
    overview(series, sheet, lab["polygons"], "check_overview.png")
    im = page(series, sheet)
    tiles = [zoom_img(im, round(x), round(y), 80, lab["polygons"]) for poly in lab["polygons"] for x, y in poly]
    S = grid(tiles, 4)
    out = outdir(series, sheet) / "check_vertices.png"
    S.save(out)
    print(out, f"{len(tiles)} vertices")


if __name__ == "__main__":
    cmd, series, sheet, *rest = sys.argv[1:]
    if cmd == "overview":
        overview(series, sheet)
    elif cmd == "zoom":
        half = 120
        if "--half" in rest:
            i = rest.index("--half")
            half = int(rest[i + 1])
            rest = rest[:i] + rest[i + 2:]
        zoom(series, sheet, [tuple(int(v) for v in a.split(",")) for a in rest], half)
    elif cmd == "check":
        check(series, sheet)
    else:
        sys.exit(__doc__)
