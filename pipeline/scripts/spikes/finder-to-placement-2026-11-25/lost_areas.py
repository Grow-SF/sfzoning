# One-off check, not pipeline code: which of the finder's areas on a trial sheet become blocks, are cut at the rim, or are dropped by the outline checks.
"""Scratch check (not pipeline code): of the finder's areas on a trial sheet, which become blocks, which are cut at
the rim, and which the tracing step's outline code drops, by the check that dropped them.  Writes a RESULT picture."""
import json, math, sys
from collections import defaultdict
from pathlib import Path
import cv2, numpy as np
from scipy import ndimage as ndi
from sfzmaps import linework
from sfzmaps.linework import (CLEAN_PX, MIN_BLOCK_AREA, SIMPLIFY_PX, POLY_EPS_STREET_FRAC, CLOSE_STREET_FRAC,
                              MIN_BLOCK_SOLIDITY, SMALL_EPS_FRAC, _polygon, _disk, _odd, _drop_lettering,
                              street_width, stroke_width, estimate_street_px, sheet_scale)
from sfzmaps.find import areas, pages, read, wholesheet, placeblocks, trial

run = pages.OUT / "runs" / "all-pieces"
root = trial.arm_root("all-pieces", "finder")
out = trial.TRIAL / "all-pieces" / "look"
for s in sys.argv[1:]:
    series, sheet = s.split("/")
    sid = (series, sheet)
    work = root / "work" / series / sheet
    img = cv2.imread(str(work / "page.png"), cv2.IMREAD_UNCHANGED)
    ch = read.load(wholesheet.paths(run, sid)[0])
    keep = pages.load_keep(sid, img.shape[:2])
    ink = linework.binarize(img)
    lab = areas.areas(ch[0], keep)
    cut = {int(g) for g in placeblocks.edge_cut(lab, keep)}
    d = json.loads((work / "blocks_px.geojson").read_text())
    kept = {int(f["properties"]["area"]) for f in d["features"]}
    photo = linework.is_photo(img)
    sw = street_width(~ink & keep); stroke = stroke_width(ink)
    got = placeblocks._pass(lab, cut, ink, keep, photo, sw, stroke, 1.0)
    est, measured = estimate_street_px(got.polys)
    scale = sheet_scale(est) if measured else 1.0
    clean_px = _odd(CLEAN_PX * scale); min_area = MIN_BLOCK_AREA * scale ** 2
    eps = max(SIMPLIFY_PX, POLY_EPS_STREET_FRAC * sw)
    close_px = max(clean_px, int(round(CLOSE_STREET_FRAC * sw)) | 1)
    size = np.bincount(lab.ravel(), minlength=lab.max() + 1)
    why = {}
    h, w = lab.shape
    for g, sl in enumerate(ndi.find_objects(lab), start=1):
        if sl is None:
            continue
        if g in cut:
            why[g] = "cut at the rim"; continue
        if g in kept:
            why[g] = "block"; continue
        y0, y1 = max(0, sl[0].start - 1), min(h, sl[0].stop + 1)
        x0, x1 = max(0, sl[1].start - 1), min(w, sl[1].stop + 1)
        pad = close_px + 1
        sub = np.zeros((y1 - y0 + 2 * pad, x1 - x0 + 2 * pad), np.uint8)
        sub[pad:-pad, pad:-pad] = lab[y0:y1, x0:x1] == g
        sub = cv2.morphologyEx(sub, cv2.MORPH_CLOSE, _disk(close_px))
        sub = cv2.morphologyEx(sub, cv2.MORPH_OPEN, _disk(clean_px))
        if int(sub.sum()) < min_area:
            why[g] = "too small"; continue
        cs, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        c = max(cs, key=cv2.contourArea)
        area = cv2.contourArea(c); hull = cv2.contourArea(cv2.convexHull(c))
        if area < min_area:
            why[g] = "too small"; continue
        if hull <= 0 or area / hull < MIN_BLOCK_SOLIDITY:
            why[g] = "too ragged (not solid)"; continue
        poly = _polygon(c, min(eps, SMALL_EPS_FRAC * math.sqrt(area)), clean_px)
        if poly is None or poly.area < min_area:
            why[g] = "no outline after simplifying"; continue
        why[g] = "taken for lettering (or lost when moved in)"
    tally = defaultdict(lambda: [0, 0]); big = defaultdict(list)
    for g, r in why.items():
        tally[r][0] += 1; tally[r][1] += int(size[g]); big[r].append((int(size[g]), g))
    inside = int(keep.sum()); notstreet = int((lab > 0).sum())
    print(f"\n{s}: scale {scale:.3f}  min block area {min_area:.0f} px  solidity floor {MIN_BLOCK_SOLIDITY}  "
          f"street width {sw:.1f} px  areas {int(lab.max())}; ground inside the frame {inside} px, not street {notstreet} px")
    for r, (n, px) in sorted(tally.items(), key=lambda kv: -kv[1][1]):
        top = sorted(big[r], reverse=True)[:6]
        cen = [(sz, tuple(int(v) for v in ndi.center_of_mass(lab == g))[::-1]) for sz, g in top] if r != "block" else []
        print(f"  {r:45s} {n:4d} areas  {px:9d} px  {100 * px / notstreet:5.1f}% of not-street ground"
              + (f"   largest (px, centre x,y): {cen}" if cen else ""))
    # RESULT picture: the scan, each area tinted by what became of it
    lum = img if img.ndim == 2 else cv2.cvtColor(img[..., :3], cv2.COLOR_BGR2GRAY)
    if lum.dtype != np.uint8:
        lum = (lum > 0).astype(np.uint8) * 255
    if lum.max() == 1:
        lum = lum * 255
    pic = np.repeat(lum[..., None], 3, 2).astype(np.float32)
    pts = np.asarray(placeblocks.lot_block_points(sid, lab.shape), float).reshape(-1, 2)
    xi = np.rint(pts[:, 0]).astype(int); yi = np.rint(pts[:, 1]).astype(int)
    on = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    per = np.bincount(lab[yi[on], xi[on]], minlength=lab.max() + 1)
    for g in list(why):
        if why[g] == "block" and per[g] > 1:
            why[g] = "joined"
    col = {"block": (120, 200, 120), "joined": (235, 150, 60), "cut at the rim": (60, 60, 230), "too ragged (not solid)": (0, 150, 255),
           "too small": (200, 200, 60), "no outline after simplifying": (200, 60, 200),
           "taken for lettering (or lost when moved in)": (200, 60, 200)}            # BGR
    lut = np.zeros((lab.max() + 1, 3), np.float32); has = np.zeros(lab.max() + 1, bool)
    for g, r in why.items():
        lut[g] = col[r]; has[g] = True
    m = has[lab]
    pic[m] = 0.55 * pic[m] + 0.45 * lut[lab[m]]
    edge = (~keep) & ndi.binary_dilation(keep, iterations=2)
    pic[edge] = (0, 0, 0)
    from sfzmaps.find import look
    key = (("green = an area that became one block outline, holding one of today's blocks (or none)", (120, 200, 120)),
           ("blue = an area that became ONE outline but holds two or more of today's blocks (blocks joined where the read breaks a street)", (235, 150, 60)),
           ("red = an area that touches the edge of the map, so it is dropped (open land, water, the margin, and every block joined to them)", (60, 60, 230)),
           ("orange / purple = dropped by the outline cleaning as too ragged / as lettering;  white = read as street", (0, 150, 255)))
    f = out / series / f"{sheet}__what-became-of-areas.png"
    k = 3000 / max(pic.shape[:2])
    small = cv2.resize(pic.astype(np.uint8), None, fx=k, fy=k, interpolation=cv2.INTER_AREA) if k < 1 else pic.astype(np.uint8)
    title = f"RESULT (debugging picture): {series} / {sheet}: what became of each area the finder calls not street (run all-pieces)"
    cv2.imwrite(str(f), look._keyed(pic.astype(np.uint8), title, key))
    cv2.imwrite(str(f.with_suffix(".small.png")), look._keyed(small, title, key))
    print("  picture", f)
