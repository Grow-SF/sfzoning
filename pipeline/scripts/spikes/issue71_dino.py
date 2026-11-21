"""Issue #71 spike 2: frozen DINOv2 patch features + a small trained head for the page layout.  Throwaway.

Backbone: DINOv2 ViT-B/14 with registers (`facebook/dinov2-with-registers-base`, Apache-2.0) loaded through Hugging
Face `transformers` (safetensors weights; torch.hub was not used: it runs code fetched from GitHub).  Run with
`uv run --with transformers --with safetensors python scripts/spikes/issue71_dino.py <cmd>`.

  extract   features per page at long side 1036 and 1554 px (74 / 111 patches), the last 4 layers' patch tokens,
            each through the final layer norm, concatenated (3072 ch; the last 768 are the last layer), float16,
            cached under data/frame-dino/feat/<scale>/
  eval      fold 0 and/or leave-one-sheet-out for variants (scale-layers-head) -> data/frame-dino/results/*.json
  loso_parts / loso_merge   leave-one-sheet-out for one variant, shared between processes a sheet at a time
  held_look a leave-one-sheet-out picture of one labelled sheet (label outline in green)
  table     markdown tables of every result
  ours      the region network's fold-0 numbers from data/frame-shape/cache, same split
  look      a head trained on all 21 (the best variant) on every unlabelled 1948 page + one page of 1964, 1968,
            1972, 1980, 1990-1996-zoning -> data/frame-dino/look/, data/frame-dino/compare.png
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage

from sfzmaps.frame import data, decide, folds, look, scores

Image.MAX_IMAGE_PIXELS = None
torch.set_num_threads(4)
log = logging.getLogger("dino")

OUT = data.DATA_DIR / "frame-dino"
FEAT = OUT / "feat"
NEAR = OUT / "near"
SCALES = (1036, 1554)
PATCH = 14
MODEL = "facebook/dinov2-with-registers-base"
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
EDGE_PX = 100.0
HIDDEN = 32
MAX_STEPS = 3000
LR = {"linear": 1e-2, "conv3": 1e-3}   # linear at 1e-3 had not flattened after 3000 steps (220 s)
OTHERS = [("1964", "20180102105639444-p10"), ("1968", "20180102110513967-p10"), ("1972", "20180102113004574-p10"),
          ("1980", "20180102115319417-p10"), ("1990-1996-zoning", "1990-1996-zoning-p30")]
OURS_CACHE = data.DATA_DIR / "frame-shape" / "cache"


def look_pages():
    labelled = set(data.labelled_sheets())
    return [s for s in data.map_pages() if s[0] == "1948" and s not in labelled] + OTHERS


def all_pages():
    return data.labelled_sheets() + look_pages()


def grid(h, w, long_side):
    s = long_side / max(h, w)
    gh, gw = max(1, round(h * s / PATCH)), max(1, round(w * s / PATCH))
    return gh, gw


def feat_path(scale, sid):
    return FEAT / str(scale) / f"{sid[0]}__{sid[1]}.npy"


# ---------------------------------------------------------------------------------------------------------------
# extraction
# ---------------------------------------------------------------------------------------------------------------

def cmd_extract(device="cpu"):
    from transformers import Dinov2WithRegistersModel
    from sfzmaps import gpu
    m = Dinov2WithRegistersModel.from_pretrained(MODEL, use_safetensors=True).eval()
    nreg = m.config.num_register_tokens
    times = {}
    with gpu.gpu_turn(True, device):
        m = m.to(device)
        for scale in SCALES:
            (FEAT / str(scale)).mkdir(parents=True, exist_ok=True)
            for sid in all_pages():
                path = feat_path(scale, sid)
                if path.exists():
                    continue
                t0 = time.perf_counter()
                lum = data.load_luminance(data.page_path(*sid))
                h, w = lum.shape
                gh, gw = grid(h, w, scale)
                small = np.asarray(Image.fromarray(lum).resize((gw * PATCH, gh * PATCH), Image.BOX), np.float32)
                x = (np.repeat(small[..., None], 3, -1) - MEAN) / STD
                x = torch.from_numpy(x.transpose(2, 0, 1)[None].copy()).to(device)
                with torch.no_grad():
                    hs = m(pixel_values=x, output_hidden_states=True).hidden_states[-4:]
                    toks = [m.layernorm(t)[0, 1 + nreg:] for t in hs]
                    f = torch.cat(toks, -1).T.reshape(-1, gh, gw)
                np.save(path, f.cpu().numpy().astype(np.float16))
                dt = time.perf_counter() - t0
                times.setdefault(str(scale), []).append(dt)
                log.info("%s %s/%s %dx%d %.1f s", scale, *sid, gh, gw, dt)
    if times:
        (OUT / f"extract_times_{device}.json").write_text(json.dumps(times))


# ---------------------------------------------------------------------------------------------------------------
# targets and heads
# ---------------------------------------------------------------------------------------------------------------

_TGT = {}


def targets(scale, sid):
    """(class shares (3, gh, gw), weight (gh, gw)) of a labelled sheet on its feature grid."""
    key = (scale, sid)
    path = OUT / "targets" / str(scale) / f"{sid[0]}__{sid[1]}.npz"
    if key not in _TGT and path.exists():
        with np.load(path) as z:
            _TGT[key] = (z["t"], z["w"])
    if key not in _TGT:
        lum = data.load_luminance(data.page_path(*sid))
        h, w = lum.shape
        gh, gw = grid(h, w, scale)
        cls = data.class_mask(data.load_label(*sid), data.load_remove(*sid), lum.shape)
        t = np.stack([np.asarray(Image.fromarray((cls == k).astype(np.float32)).resize((gw, gh), Image.BOX))
                      for k in range(3)])
        t = t / t.sum(0, keepdims=True)
        ink = np.asarray(Image.fromarray(1.0 - lum).resize((gw, gh), Image.BOX), np.float32)
        _TGT[key] = (t.astype(np.float32), (0.1 + ink).astype(np.float32))
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, t=_TGT[key][0], w=_TGT[key][1])
    return _TGT[key]


_FEATS = {}


def feats(scale, sid, layers):
    key = (scale, sid)
    if key not in _FEATS:
        _FEATS[key] = torch.from_numpy(np.load(feat_path(scale, sid)).astype(np.float32))
    f = _FEATS[key]
    return f if layers == "last4" else f[-768:]


PCA_DIM = 128   # conv3 heads see the fold's training-patch PCA of the standardised features (3x3 conv on 3072 ch is
                # ~7 s a step on 2 cpu threads; on 128 it is ~0.1 s)


class Head(torch.nn.Module):
    """`linear`: a 1x1 conv (as a matmul on the flat patches).  `conv3`: PCA -> 3x3 conv (HIDDEN) -> ReLU -> 1x1."""

    def __init__(self, c, kind):
        super().__init__()
        self.kind = kind
        if kind == "linear":
            self.net = torch.nn.Linear(c, 3)
        else:
            self.net = torch.nn.Sequential(torch.nn.Conv2d(c, HIDDEN, 3, padding=1), torch.nn.ReLU(),
                                           torch.nn.Conv2d(HIDDEN, 3, 1))

    def forward(self, x):  # x (C, gh, gw) -> logits (3, gh, gw)
        if self.kind == "linear":
            c, gh, gw = x.shape
            return self.net(x.reshape(c, -1).T).T.reshape(3, gh, gw)
        return self.net(x[None])[0]


class Fitted:
    def __init__(self, head, mu, sd, proj, scale, layers):
        self.head, self.mu, self.sd, self.proj, self.scale, self.layers = head, mu, sd, proj, scale, layers

    def prep(self, sid):
        x = feats(self.scale, sid, self.layers)
        c, gh, gw = x.shape
        z = (x.reshape(c, -1) - self.mu[:, None]) / self.sd[:, None]
        if self.proj is not None:
            z = self.proj.T @ z
        return z.reshape(-1, gh, gw)


def train_head(scale, layers, kind, train, seed=0):
    torch.manual_seed(seed)
    xs = [feats(scale, s, layers) for s in train]
    flat = torch.cat([x.reshape(x.shape[0], -1) for x in xs], 1)
    mu, sd = flat.mean(1), flat.std(1) + 1e-6
    proj = None
    if kind == "conv3":
        z = (flat - mu[:, None]) / sd[:, None]
        evals, evecs = torch.linalg.eigh(z @ z.T / z.shape[1])
        proj = evecs[:, -PCA_DIM:].flip(1).contiguous()
    del flat
    fit = Fitted(None, mu, sd, proj, scale, layers)
    xs = [fit.prep(s) for s in train]
    ts = [torch.from_numpy(targets(scale, s)[0]) for s in train]
    ws = [torch.from_numpy(targets(scale, s)[1]) for s in train]
    wsum = sum(float(w.sum()) for w in ws)
    if kind == "linear":   # one batch of every patch
        xs = [torch.cat([x.reshape(x.shape[0], -1) for x in xs], 1)[:, :, None]]
        ts = [torch.cat([t.reshape(3, -1) for t in ts], 1)[:, :, None]]
        ws = [torch.cat([w.reshape(-1) for w in ws])[:, None]]
    head = Head(xs[0].shape[0], kind)
    opt = torch.optim.Adam(head.parameters(), lr=LR[kind])
    hist = []
    for step in range(MAX_STEPS):
        opt.zero_grad()
        loss = 0.0
        for x, t, w in zip(xs, ts, ws):
            z = head(x)
            loss = loss + (-(t * F.log_softmax(z, 0)).sum(0) * w).sum() / wsum
        loss.backward()
        opt.step()
        hist.append(float(loss.detach()))
        # flat: the mean of the last 100 steps within 0.2% of the 100 before
        if step >= 300 and step % 50 == 0:
            a, b = np.mean(hist[-200:-100]), np.mean(hist[-100:])
            if a - b < 0.002 * a:
                break
    head.eval()
    fit.head = head
    return fit, hist


def predict(fit, sid, shape):
    with torch.no_grad():
        p = torch.softmax(fit.head(fit.prep(sid)), 0)[None]
        p = F.interpolate(p, size=shape, mode="bilinear", align_corners=False)[0]
    return p.numpy()


# ---------------------------------------------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------------------------------------------

def near_mask(sid, cls):
    path = NEAR / f"{sid[0]}__{sid[1]}.npz"
    if path.exists():
        with np.load(path) as z:
            return np.unpackbits(z["near"], count=cls.size).reshape(cls.shape).astype(bool)
    edge = np.zeros(cls.shape, bool)
    edge[:-1] |= cls[:-1] != cls[1:]
    edge[1:] |= cls[:-1] != cls[1:]
    edge[:, :-1] |= cls[:, :-1] != cls[:, 1:]
    edge[:, 1:] |= cls[:, :-1] != cls[:, 1:]
    near = ndimage.distance_transform_edt(~edge) <= EDGE_PX if edge.any() else np.zeros(cls.shape, bool)
    NEAR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, near=np.packbits(near.ravel()))
    return near


def score_sheet(p, sid, lum=None):
    if lum is None:
        lum = data.load_luminance(data.page_path(*sid))
    cls = data.class_mask(data.load_label(*sid), data.load_remove(*sid), lum.shape)
    d = decide.decide(p, lum, data.map_counts().get(sid, 1))
    sc = scores.sheet_scores(p, d.deleted, cls, lum)
    near = near_mask(sid, cls)
    ink = 1.0 - lum.astype(np.float64)
    c = np.where((cls != 0) & ~d.deleted, decide.COST_FRAME * ink, 0.0) + np.where((cls == 0) & d.deleted, ink, 0.0)
    sc["cost_edge"] = float(c[near].sum())
    sc["cost_layout"] = float(c[~near].sum())
    fk = np.where((cls != 0) & ~d.deleted, decide.COST_FRAME * ink, 0.0)
    sc["layout_frame_kept_cost"] = float(fk[~near].sum())
    sc["layout_map_cut_cost"] = sc["cost_layout"] - sc["layout_frame_kept_cost"]
    return sc


VARIANTS = [(s, l, k) for k in ("linear", "conv3") for s in SCALES for l in ("last", "last4")]


def vname(v):
    return f"{v[0]}-{v[1]}-{v[2]}"


def cmd_eval(which=None, which_plans="fold0,loso", seed="0"):
    """Each variant x plan's held-out scores to data/frame-dino/results/<variant>-<plan>.json (skipped if there)."""
    sheets = data.labelled_sheets()
    variants = [v for v in VARIANTS if which is None or vname(v) in which.split(",")]
    plans = {"fold0": [folds.split(sheets, 4)[0]],
             "loso": [folds.Fold(tuple(s for s in sheets if s != h), (h,)) for h in sheets]}
    lums = {}
    for v in variants:
        for plan in which_plans.split(","):
            path = OUT / "results" / (f"{vname(v)}-{plan}.json" if seed == "0" else f"{vname(v)}-{plan}-seed{seed}.json")
            if path.exists():
                continue
            rows, t_train = [], []
            for f in plans[plan]:
                t0 = time.perf_counter()
                fit, hist = train_head(*v, list(f.train), seed=int(seed))
                t_train.append(time.perf_counter() - t0)
                for sid in f.held:
                    if sid not in lums:
                        lums[sid] = data.load_luminance(data.page_path(*sid))
                    lum = lums[sid]
                    p = predict(fit, sid, lum.shape)
                    sc = score_sheet(p, sid, lum)
                    sc.update(series=sid[0], sheet=sid[1], steps=len(hist), train_loss=hist[-1])
                    rows.append(sc)
                    log.info("%s %s %s/%s cost %.0f layout %.0f (steps %d, loss %.4f, %.0f s)", vname(v), plan, *sid,
                             sc["cost"], sc["cost_layout"], len(hist), hist[-1], t_train[-1])
            tot = sum(r["cost"] for r in rows)
            lay = sum(r["cost_layout"] for r in rows)
            log.info("== %s %s pooled cost %.0f layout %.0f", vname(v), plan, tot, lay)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"sheets": rows, "train_s": t_train}, indent=1))


def cmd_loso_parts(variant, order="fwd"):
    """Leave-one-sheet-out for one variant, a sheet at a time, so several processes can share it: each claims a
    sheet (a `.claim` file) before training; `loso_merge` writes results/<variant>-loso.json when all are done."""
    s, l, k = variant.split("-")
    v = (int(s), l, k)
    sheets = data.labelled_sheets()
    parts = OUT / "results" / "parts" / variant
    parts.mkdir(parents=True, exist_ok=True)
    for h in (sheets if order == "fwd" else sheets[::-1]):
        claim = parts / f"{h[0]}__{h[1]}.claim"
        try:
            claim.open("x").close()
        except FileExistsError:
            continue
        t0 = time.perf_counter()
        fit, hist = train_head(*v, [x for x in sheets if x != h])
        dt = time.perf_counter() - t0
        lum = data.load_luminance(data.page_path(*h))
        sc = score_sheet(predict(fit, h, lum.shape), h, lum)
        sc.update(series=h[0], sheet=h[1], steps=len(hist), train_loss=hist[-1], train_s=dt)
        (parts / f"{h[0]}__{h[1]}.json").write_text(json.dumps(sc))
        log.info("%s loso %s/%s cost %.0f layout %.0f (steps %d, %.0f s)", variant, *h, sc["cost"],
                 sc["cost_layout"], len(hist), dt)


def cmd_loso_merge(variant):
    parts = OUT / "results" / "parts" / variant
    rows = [json.loads((parts / f"{h[0]}__{h[1]}.json").read_text()) for h in data.labelled_sheets()]
    (OUT / "results" / f"{variant}-loso.json").write_text(
        json.dumps({"sheets": rows, "train_s": [r["train_s"] for r in rows]}, indent=1))
    log.info("== %s loso pooled cost %.0f layout %.0f", variant, sum(r["cost"] for r in rows),
             sum(r["cost_layout"] for r in rows))


def cmd_held_look(variant, series, sheet):
    """A held-out (leave-one-sheet-out) picture of one labelled sheet: look/held/<variant>/<series>__<sheet>.png,
    with the label's outline drawn in green."""
    from PIL import ImageDraw
    s, l, k = variant.split("-")
    v = (int(s), l, k)
    h = (series, sheet)
    fit, hist = train_head(*v, [x for x in data.labelled_sheets() if x != h])
    lum = data.load_luminance(data.page_path(*h))
    p = predict(fit, h, lum.shape)
    d = decide.decide(p, lum, data.map_counts().get(h, 1))
    img = look.sheet_picture(lum, p[1] + p[2], d.deleted, look.LONG_SIDE)
    sc = img.width / lum.shape[1]
    draw = ImageDraw.Draw(img)
    for poly in data.load_label(*h) + data.load_remove(*h):
        draw.line([(x * sc, y * sc) for x, y in poly + poly[:1]], fill=(0, 160, 0), width=2)
    path = OUT / "look" / "held" / variant / f"{series}__{sheet}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    log.info("held look %s", path)


def cmd_ours():
    out = {}
    for path in sorted(OURS_CACHE.glob("*.npy")):
        series, sheet = path.stem.split("__", 1)
        sid = (series, sheet)
        p = np.load(path)
        sc = score_sheet(p, sid)
        sc.update(series=series, sheet=sheet)
        out[f"{series}/{sheet}"] = sc
        log.info("ours %s/%s cost %.0f layout %.0f", *sid, sc["cost"], sc["cost_layout"])
    (OUT / "ours.json").write_text(json.dumps(out, indent=1))


# ---------------------------------------------------------------------------------------------------------------
# pictures
# ---------------------------------------------------------------------------------------------------------------

def cmd_look(variant):
    s, l, k = variant.split("-")
    v = (int(s), l, k)
    fit, hist = train_head(*v, data.labelled_sheets())
    log.info("all-21 head %s: %d steps, loss %.4f", variant, len(hist), hist[-1])
    by_series = {}
    for sid in look_pages():
        lum = data.load_luminance(data.page_path(*sid))
        p = predict(fit, sid, lum.shape)
        d = decide.decide(p, lum, data.map_counts().get(sid, 1))
        q = p[1] + p[2]
        full = OUT / "look" / sid[0] / f"{sid[1]}.png"
        full.parent.mkdir(parents=True, exist_ok=True)
        look.sheet_picture(lum, q, d.deleted, look.LONG_SIDE).save(full)
        tile = look.sheet_picture(lum, q, d.deleted, look.MOSAIC_SIDE)
        by_series.setdefault(sid[0], []).append((tile, (sid[1], f"dino {variant}")))
        log.info("look %s/%s", *sid)
    for series, tiles in by_series.items():
        look._mosaic(tiles, look.MOSAIC_SIDE, look.MOSAIC_COLUMNS, OUT / "look" / f"{series}.png")
    others = [t for ser, ts in by_series.items() if ser != "1948" for t in ts]
    look._mosaic(others, look.MOSAIC_SIDE, look.MOSAIC_COLUMNS, OUT / "look" / "unseen-editions.png")
    look._mosaic(by_series["1948"], look.MOSAIC_SIDE, look.MOSAIC_COLUMNS, OUT / "compare.png")


def cmd_table():
    """Markdown tables of every result file, and ours on fold 0."""
    ours = json.loads((OUT / "ours.json").read_text())
    res = {p.stem: json.loads(p.read_text()) for p in sorted((OUT / "results").glob("*.json"))}
    print("| variant | plan | pooled cost | edge (<=100 px) | layout (>100 px) | layout frame kept x3 | layout map cut |"
          " log loss (mean) | train s (mean) |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    o = list(ours.values())
    print(f"| ours (region net) | fold0 | {sum(r['cost'] for r in o):,.0f} | {sum(r['cost_edge'] for r in o):,.0f} | "
          f"{sum(r['cost_layout'] for r in o):,.0f} | {sum(r['layout_frame_kept_cost'] for r in o):,.0f} | "
          f"{sum(r['layout_map_cut_cost'] for r in o):,.0f} | {np.mean([r['log_loss'] for r in o]):.3f} | |")
    for k, v in res.items():
        r = v["sheets"]
        name, plan = k.split("-fold0")[0].split("-loso")[0], k[len(k.split("-fold0")[0].split("-loso")[0]) + 1:]
        print(f"| {name} | {plan} | {sum(x['cost'] for x in r):,.0f} | {sum(x['cost_edge'] for x in r):,.0f} | "
              f"{sum(x['cost_layout'] for x in r):,.0f} | {sum(x['layout_frame_kept_cost'] for x in r):,.0f} | "
              f"{sum(x['layout_map_cut_cost'] for x in r):,.0f} | {np.mean([x['log_loss'] for x in r]):.3f} | "
              f"{np.mean(v['train_s']):.0f} |")
    for k, v in res.items():
        print(f"\n### {k}\n\n| sheet | cost | edge | layout | frame_kept | map_cut | log_loss | ours cost | ours layout |")
        print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for x in v["sheets"]:
            o = ours.get(f"{x['series']}/{x['sheet']}")
            oc = f"{o['cost']:,.0f} | {o['cost_layout']:,.0f}" if o else " | "
            print(f"| {x['series']}/{x['sheet']} | {x['cost']:,.0f} | {x['cost_edge']:,.0f} | {x['cost_layout']:,.0f} | "
                  f"{x['frame_kept']:,.0f} | {x['map_cut']:,.0f} | {x['log_loss']:.3f} | {oc} |")
        if "fold0" in k:
            a = {(x["series"], x["sheet"]): x["cost"] for x in v["sheets"]}
            al = {(x["series"], x["sheet"]): x["cost_layout"] for x in v["sheets"]}
            b = {(x["series"], x["sheet"]): x["cost"] for x in ours.values()}
            bl = {(x["series"], x["sheet"]): x["cost_layout"] for x in ours.values()}
            print("\npaired (dino - ours) mean per sheet, 5-95%%: cost %s, layout %s" % (
                tuple(round(t) for t in scores.paired_difference(a, b)),
                tuple(round(t) for t in scores.paired_difference(al, bl))))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cmd, *args = sys.argv[1:]
    {"extract": cmd_extract, "eval": cmd_eval, "ours": cmd_ours, "look": cmd_look, "table": cmd_table, "loso_parts": cmd_loso_parts, "loso_merge": cmd_loso_merge, "held_look": cmd_held_look}[cmd](*args)
