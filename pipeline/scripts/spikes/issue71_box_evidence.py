"""Issue #71, label boxes: evidence only, no fix.  For the pages with wrong boxes, what the box step sees and chooses.

Per candidate piece (a connected piece of "removed" chance over the cut level inside the kept region): its size, the
model's chances on its ink, the gain of the best rectangle at the frame's angle, of the best rectangle at the piece's
own angle, and of the best ellipse, which one the step chose, and whether a labelled box lies there.
Per labelled box (test sheets): how much of its ink the model would delete, the gain of deleting exactly the labelled
shape, and the gain and overlap of the shape the step fitted there.

    uv run python scripts/spikes/issue71_box_evidence.py
"""
import json
import math
import subprocess

import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

from sfzmaps.frame import blocks as blk
from sfzmaps.frame import data, layout, rules, shape
from sfzmaps.frame import run as fr
from sfzmaps.frame.decide import cut_level

SRC = data.DATA_DIR / "frame-look39"
OUT = data.DATA_DIR / "frame-box-evidence"
TEST = data.LABELS.parent / "frame-labels-test"
T = 1.540
PARAMS = shape.Params(lam_notch=1e4, beta=3, lam_block=1e3)
# The box step as it was when this evidence was gathered (before f2c9dd8 made rectangles square to the frame):
MOMENT_ANGLES = (-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3)


def _wrap90(deg):
    return (deg + 45.0) % 90.0 - 45.0


PAGES = [("1921", "20180102094348509"), ("1929", "20180102100804566-p14"), ("1935", "20180102102853571-p12"),
         ("1948", "6a"), ("1980-1990-su", "1980-1990-su-p05"), ("1990-1996-zoning", "1990-1996-zoning-p30"),
         ("1948", "4")]


def pieces(p, D, kept, frame, bounds):
    """`blocks.candidates`, piece by piece, keeping every family's best shape."""
    h, w = D.shape
    L, S = bounds.rect_max
    amax, bmax = bounds.ellipse_max
    lab, n = ndi.label((p[2] > cut_level()) & kept, structure=blk._EIGHT)
    sizes = np.bincount(lab.ravel(), minlength=n + 1)
    out = []
    for k, sl in enumerate(ndi.find_objects(lab), start=1):
        if sl is None or sizes[k] < blk.BLOCK_MIN_PX:
            continue
        ys, xs = np.nonzero(lab[sl] == k)
        ys, xs = ys + sl[0].start, xs + sl[1].start
        bh, bw = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        mg = max(blk.CROP_MARGIN[0], int(blk.CROP_MARGIN[1] * max(bh, bw)))
        y0, y1 = max(0, sl[0].start - mg), min(h, sl[0].stop + mg)
        x0, x1 = max(0, sl[1].start - mg), min(w, sl[1].stop + mg)
        Gc = np.where(kept[y0:y1, x0:x1], -D[y0:y1, x0:x1], np.float32(0))
        cx, cy = xs.mean() + 0.5, ys.mean() + 0.5
        cov = np.cov(np.stack([xs, ys])) if len(xs) > 2 else np.eye(2)
        ev, evec = np.linalg.eigh(cov)
        thm = math.degrees(math.atan2(evec[1, 1], evec[0, 1]))
        fam = {}
        for name, angs in (("rect at the piece's angle", [_wrap90(thm) + d for d in MOMENT_ANGLES]),
                           ("rect at the frame's angle", [frame.phi + d for d in blk.FRAME_ANGLES])):
            best = None
            for th in angs:
                t = math.radians(th)
                c, s = math.cos(t), math.sin(t)
                pu = c * (xs + 0.5 - cx) + s * (ys + 0.5 - cy)
                pv = -s * (xs + 0.5 - cx) + c * (ys + 0.5 - cy)
                init = (pu.min(), pu.max() + 1, pv.min(), pv.max() + 1)
                for lu, lv in ((L, S), (S, L)):
                    g, b = blk._rect_on(Gc, x0, y0, cx, cy, th, init, lu, lv)
                    if best is None or g > best.gain:
                        best = b
            fam[name] = best
        a, b = 2 * math.sqrt(max(ev[1], 1.0)), 2 * math.sqrt(max(ev[0], 1.0))
        fam["ellipse"] = blk._ellipse_on(Gc, x0, y0, cx, cy, a, b, thm, amax, bmax)[1]
        out.append({"mask_px": int(sizes[k]), "ys": ys, "xs": xs, "box": (y0, y1, x0, x1), "moment_angle": thm,
                    "fam": fam})
    return out


def shape_mask(b, h, w):
    m = np.zeros((h, w), bool)
    y0, y1, x0, x1 = blk.block_box(b, h, w)
    if y1 > y0 and x1 > x0:
        m[y0:y1, x0:x1] = blk.block_crop(b, y0, y1, x0, x1)
    return m


def page(sid, heads, bounds, maps):
    lum = data.load_luminance(data.page_path(*sid))
    h, w = lum.shape
    ink = 1.0 - lum
    p0 = fr.chances([layout.head_logits(hd, fr.load_features(data.OUT, sid)) for hd in heads], lum.shape, T)
    dec, frm, ns, bs = shape.decide(p0, lum, PARAMS, bounds, maps)
    p = rules.share(p0, lum)
    D = shape.cost_field(p, lum)
    K = shape._shape_mask(frm, ns, lum.shape)
    labelled = (TEST / sid[0] / f"{sid[1]}.json").exists()
    polys = data.load_label(*sid, TEST) if labelled else []
    rem = data.load_remove(*sid, TEST) if labelled else []
    lab_masks = [data.class_mask(polys, [poly], lum.shape) == 2 for poly in rem]
    print(f"\n=== {sid[0]}/{sid[1]}  page {h}x{w}, frame angle {frm.phi:+.2f}, boxes kept by the step: "
          f"{[(b.kind, round(b.theta, 1), round(b.gain)) for b in bs]}", flush=True)
    kept_masks = [shape_mask(b, h, w) for b in bs]
    rows = []
    for i, pc in enumerate(pieces(p, D, K, frm, bounds)):
        ys, xs = pc["ys"], pc["xs"]
        wi = ink[ys, xs]
        gains = {k: v.gain for k, v in pc["fam"].items()}
        chosen = max(gains, key=gains.get)
        b = pc["fam"][chosen]
        bm = shape_mask(b, h, w)
        on = [float((bm & lm).sum() / max((bm | lm).sum(), 1)) for lm in lab_masks]
        row = {"piece": i, "px": pc["mask_px"], "ink": float(wi.sum()),
               "p_removed_on_ink": float((p[2][ys, xs] * wi).sum() / max(wi.sum(), 1e-9)),
               "p_map_on_ink": float((p[0][ys, xs] * wi).sum() / max(wi.sum(), 1e-9)),
               "moment_angle": round(pc["moment_angle"], 1), "gains": {k: round(v) for k, v in gains.items()},
               "chosen": chosen, "chosen_angle": round(b.theta, 1), "accepted": gains[chosen] > PARAMS.lam_block,
               "map_ink_it_deletes": float((ink * p[0])[bm & K].sum()),
               "ink_it_deletes": float(ink[bm & K].sum()),
               "best_overlap_with_a_labelled_box": round(max(on), 2) if on else None}
        rows.append(row)
        if gains[chosen] > 200:
            print(f"  piece {i}: {row['px']} px, removed-chance on its ink {row['p_removed_on_ink']:.2f} (map "
                  f"{row['p_map_on_ink']:.2f}), own angle {row['moment_angle']}; gains {row['gains']}; chosen "
                  f"{chosen} at {row['chosen_angle']}; deletes ink {row['ink_it_deletes']:.0f} of which expected map "
                  f"{row['map_ink_it_deletes']:.0f}; overlap with a labelled box "
                  f"{row['best_overlap_with_a_labelled_box']}", flush=True)
    for j, (poly, lm) in enumerate(zip(rem, lab_masks)):
        kind, a, b_ = blk.label_kind(poly)
        wi = ink[lm]
        g_label = -float(D[lm & K].sum(dtype=np.float64))
        best = max(((float((km & lm).sum() / max((km | lm).sum(), 1)), b) for km, b in zip(kept_masks, bs)),
                   key=lambda t: t[0], default=(0.0, None))
        print(f"  LABELLED box {j} ({kind} {a:.0f}x{b_:.0f}): removed-chance on its ink "
              f"{float((p[2][lm] * wi).sum() / max(wi.sum(), 1e-9)):.2f}, outside-chance "
              f"{float((p[1][lm] * wi).sum() / max(wi.sum(), 1e-9)):.2f}; gain of deleting exactly the label "
              f"{g_label:.0f}; the step's box there: " + (f"{best[1].kind} at {best[1].theta:.1f}, gain "
              f"{best[1].gain:.0f}, overlap {best[0]:.2f}" if best[1] is not None and best[0] > 0 else "none"),
              flush=True)
    # picture: the "removed" chance over the scan, pieces' chosen shapes in orange, labels in green
    s = 1400 / max(h, w)
    g = np.asarray(Image.fromarray((lum * 255).astype(np.uint8)).resize((int(w * s), int(h * s)), Image.BOX), np.float32)
    heat = np.asarray(Image.fromarray((p[2] * 255).astype(np.uint8)).resize((int(w * s), int(h * s)), Image.BILINEAR), np.float32) / 255
    rgb = np.stack([g, g * (1 - 0.7 * heat), g * (1 - 0.7 * heat)], -1)
    im = Image.fromarray(rgb.clip(0, 255).astype(np.uint8))
    dr = ImageDraw.Draw(im)
    for poly in polys + rem:
        pts = [(x * s, y * s) for x, y in poly]
        dr.line(pts + pts[:1], fill=(0, 170, 0), width=2)
    for km in kept_masks:
        edge = km & ~ndi.binary_erosion(km, iterations=3)
        ys, xs = np.nonzero(edge)
        for x, y in zip(xs[::7], ys[::7]):
            dr.point((x * s, y * s), fill=(255, 130, 0))
    OUT.mkdir(parents=True, exist_ok=True)
    im.save(OUT / f"{sid[0]}_{sid[1][-6:]}.png")
    return {"sheet": f"{sid[0]}/{sid[1]}", "pieces": rows}


def main():
    torch.set_num_threads(4)
    train = sorted({tuple(f.split("/")[-2:]) for f in subprocess.run(
        ["git", "ls-files", "inputs/frame-labels"], capture_output=True, text=True).stdout.split() if f.endswith(".json")})
    train = [(s, h[:-5]) for s, h in train]
    heads = []
    for sd in torch.load(SRC / "heads.pt"):
        hd = layout.new_head(); hd.load_state_dict(sd); heads.append(hd.eval())
    bounds = blk.block_bounds(train, root=data.LABELS)
    print("bounds", bounds, "lam_block", PARAMS.lam_block, flush=True)
    counts = data.map_counts(data.PAGE_KINDS)
    res = [page(s, heads, bounds, counts[s]) for s in PAGES]
    (OUT / "evidence.json").write_text(json.dumps(res, indent=1))
    print("done", flush=True)


if __name__ == "__main__":
    main()
