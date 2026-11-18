"""learned-links (Task 8): a learned "same word" decision between neighbouring core px, trained where merges happen.

For a fold holding quarter q out, the pairs come from the other three quarters only, outside the answered blocks;
each px's inputs from the detector that held ITS quarter out (a-<p>-s0 for a px of quarter p), so the inputs look
held out, as at inference.  A pair is two 8-neighbouring px both with P(core) > 1/2 (so inside one crude-map
component); it is "same" when both px are in one Terra word (`run_detector.word_ids`: the words' own rectangles,
later over earlier), "different" when they are in two words or one is in a word and the other in none; two px in no
word are not taught.  Every "different" pair is kept (they are the words' boundaries, the merges' cuts); the "same"
pairs are drawn uniformly at the one rate that makes their expected number the different pairs' (over every
quarter), so each class's inputs keep their own distribution (`counts` records the whole populations).

Each px's inputs (`FEATURES`, the detector's 16 outputs): the text logit, the four kind probabilities, the axis (cos 2t,
sin 2t), the core logit, the centre offset over the height (x, y), the log length and log height, the four tags;
standardised by the training px' means and sds.  A pair's: [f_a, f_b, f_a - f_b, f_a * f_b] and the direction from a
to b (one of 8, one-hot); the logit is the mean of the MLP's on (a, b, d) and (b, a, -d), so it is the same whichever
px is a.  The MLP: two hidden layers of HIDDEN (ReLU), trained with the balanced binary cross-entropy (each class by
the inverse of its sampled number), Adam, early stopping on the pairs of a held-out share of page cells (CELL px a
side, `VAL_SHARE` of them by a fixed hash): the epoch with the least held-out loss is kept.

At inference an edge joins two neighbouring core px of the held-out quarter when logit + log(n_same / n_different) >
0 (the Bayes correction by the measured counts: the even-odds call); words are the components."""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1] / "textlines")]
import run_detector as RD  # noqa: E402

from sfzmaps.solver import paths  # noqa: E402
from sfzmaps.solver.textnet import reader_stage  # noqa: E402

log = logging.getLogger(__name__)
OUT = RD.ROOT / "links"
FEATURES = ("text", "kind_code", "kind_block", "kind_street", "kind_other", "axis_cos", "axis_sin", "core", "dx", "dy",
            "log_length", "log_height", "tag0", "tag1", "tag2", "tag3")
STEPS = ((0, 1), (1, 0), (1, 1), (1, -1))    # (dy, dx) to the right, down, down-right, down-left; their opposites
#                                              are directions 4-7
HIDDEN = 64
CELL = 512                # px: the side of the page cells the held-out pairs are drawn by (whole cells)
VAL_SHARE = 5             # one cell in this many is held out for early stopping
BATCH = 4096              # numerical: pairs a step
LR = 1e-3                 # numerical: Adam's step size
EPOCHS = 40               # numerical: at most this many passes ...
PATIENCE = 3              # ... stopping once the held-out loss has not fallen for this many
LOGIT_CLIP = 30.0         # numerical: the core and text logits read within +-this (a px not scored is -1e4)
SEED = 0


def features_at(o, ys: np.ndarray, xs: np.ndarray) -> np.ndarray:
    """The detector's 16 outputs (FEATURES) at page px (ys, xs), float32 (n, 16), from its saved word outputs."""
    eps = 0.5 / 255
    c = np.clip(o.chance[ys, xs].astype(np.float64) / 255, eps, 1 - eps)
    geo = o.geo[:, ys, xs].astype(np.float64)
    height = np.maximum(geo[3], 1e-3)
    two = np.radians(2 * geo[4])
    f = np.stack([np.log(c / (1 - c)), *o.kind_p[:, ys, xs].astype(np.float64), np.cos(two), np.sin(two),
                  np.clip(o.core[ys, xs].astype(np.float64), -LOGIT_CLIP, LOGIT_CLIP), geo[0] / height,
                  geo[1] / height, np.log(np.maximum(geo[2], 1e-3)), np.log(height),
                  *o.tag[:, ys, xs].astype(np.float64)], 1)
    return f.astype(np.float32)


def pairs_of(on: np.ndarray):
    """Every pair of 8-neighbouring px both `on`: (ya, xa, yb, xb, direction 0-3), each pair once."""
    h, w = on.shape
    out = []
    for d, (dy, dx) in enumerate(STEPS):
        x0, x1 = max(0, -dx), w - max(0, dx)             # the columns px a may be in
        a = on[0:h - dy, x0:x1]
        b = on[dy:h, x0 + dx:x1 + dx]
        ys, xs = np.nonzero(a & b)
        xs = xs + max(0, -dx)
        out.append((ys, xs, ys + dy, xs + dx, np.full(len(ys), d, np.int8)))
    return tuple(np.concatenate(k) for k in zip(*out))


def labels_of(wid: np.ndarray, ya, xa, yb, xb) -> np.ndarray:
    """Each pair's class: 1 same word, 0 different (two words, or a word and none), -1 not taught (no word)."""
    a, b = wid[ya, xa], wid[yb, xb]
    return np.where((a == 0) & (b == 0), -1, np.where((a > 0) & (a == b), 1, 0)).astype(np.int8)


def held_cell(sheet_index: int, ys: np.ndarray, xs: np.ndarray) -> np.ndarray:
    """Whether each px's page cell (CELL px a side) is held out for early stopping (one in VAL_SHARE, a fixed hash)."""
    h = (sheet_index * 1000003 + (ys // CELL).astype(np.int64) * 1009 + (xs // CELL).astype(np.int64)) * 2654435761
    return (h % (2 ** 32)) % VAL_SHARE == 0


def _area(sd, p, blocks):
    return RD.fit_area(sd, (p,), blocks)


def _core(run: Path, sheet: str) -> np.ndarray:
    with np.load(Path(run) / "words" / f"{sheet}.npz") as z:
        return z["core"]


def _count_job(args):
    p, sheet = args
    sd = RD.load_sheet(sheet)
    area = _area(sd, p, RD._SHARED["blocks"])
    if not area.any():
        return p, sheet, {"same": 0, "different": 0, "word_word": 0, "word_none": 0}
    on = area & (_core(RD.run_dir("a", paths.QUARTERS[p], 0), sheet) > 0)
    ya, xa, yb, xb, _ = pairs_of(on)
    wid = RD.word_ids(sd)
    lab = labels_of(wid, ya, xa, yb, xb)
    a, b = wid[ya, xa], wid[yb, xb]
    return p, sheet, {"same": int((lab == 1).sum()), "different": int((lab == 0).sum()),
                      "word_word": int(((lab == 0) & (a > 0) & (b > 0)).sum()),
                      "word_none": int(((lab == 0) & ((a == 0) | (b == 0))).sum())}


def _extract_job(args):
    p, sheet, rate = args
    sd = RD.load_sheet(sheet)
    area = _area(sd, p, RD._SHARED["blocks"])
    empty = {"fa": np.zeros((0, 16), np.float32), "fb": np.zeros((0, 16), np.float32), "d": np.zeros(0, np.int8),
             "y": np.zeros(0, np.int8), "val": np.zeros(0, bool), "word_word": np.zeros(0, bool)}
    if not area.any():
        return p, sheet, empty
    o = RD.outputs(RD.run_dir("a", paths.QUARTERS[p], 0), sheet)
    on = area & (o.core > 0)
    ya, xa, yb, xb, d = pairs_of(on)
    wid = RD.word_ids(sd)
    lab = labels_of(wid, ya, xa, yb, xb)
    rng = np.random.default_rng([SEED, p, paths.SHEETS.index(sheet)])
    keep = (lab == 0) | ((lab == 1) & (rng.random(len(lab)) < rate))
    ya, xa, yb, xb, d, lab = ya[keep], xa[keep], yb[keep], xb[keep], d[keep], lab[keep]
    return p, sheet, {"fa": features_at(o, ya, xa), "fb": features_at(o, yb, xb), "d": d, "y": lab,
                      "val": held_cell(paths.SHEETS.index(sheet), ya, xa),
                      "word_word": (wid[ya, xa] > 0) & (wid[yb, xb] > 0) & (lab == 0)}


def build_pairs(workers: int = RD.WORKERS) -> dict:
    """The pairs of every quarter (from its own held-out detector, outside the answered blocks): the counts first
    (OUT/counts.json), then every different pair and the same pairs drawn at one rate (OUT/pairs-<quarter>.npz)."""
    RD._SHARED["blocks"] = reader_stage.load_block_area(reader_stage.held_blocks())
    jobs = [(p, s) for p in range(len(paths.QUARTERS)) for s in paths.SHEETS]
    counted = RD._pool_map(_count_job, jobs, workers)
    per = {qn: {"same": 0, "different": 0, "word_word": 0, "word_none": 0} for qn in paths.QUARTERS}
    for p, _, c in counted:
        for k, v in c.items():
            per[paths.QUARTERS[p]][k] += v
    same, diff = sum(v["same"] for v in per.values()), sum(v["different"] for v in per.values())
    rate = diff / same
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "counts.json").write_text(json.dumps({"by_quarter": per, "same": same, "different": diff,
                                                 "same_rate": rate}, indent=1) + "\n")
    log.info("pairs: %d same, %d different; the same drawn at %.5f", same, diff, rate)
    got = RD._pool_map(_extract_job, [(p, s, rate) for p, s in jobs], workers)
    for p in range(len(paths.QUARTERS)):
        parts = [g for q_, _, g in got if q_ == p]
        np.savez(OUT / f"pairs-{paths.QUARTERS[p]}.npz",
                 **{k: np.concatenate([g[k] for g in parts]) for k in parts[0]})
    return {"by_quarter": per, "same_rate": rate}


# --- the model -------------------------------------------------------------------------------------------------------

def _net(torch):
    return torch.nn.Sequential(torch.nn.Linear(4 * len(FEATURES) + 8, HIDDEN), torch.nn.ReLU(),
                               torch.nn.Linear(HIDDEN, HIDDEN), torch.nn.ReLU(), torch.nn.Linear(HIDDEN, 1))


def pair_logit(net, fa, fb, d, torch):
    """The symmetric logit of pairs (standardised inputs fa, fb (n, 16) tensors; d (n,) long, 0-3 from a to b)."""
    eye = torch.eye(8, dtype=fa.dtype)

    def one(x, y, k):
        return net(torch.cat([x, y, x - y, x * y, eye[k]], 1))[:, 0]
    return (one(fa, fb, d) + one(fb, fa, d + 4)) / 2


def train_fold(quarter: str) -> dict:
    """The fold's MLP from the other three quarters' pairs (OUT/pairs-<quarter>.npz): OUT/model-<quarter>.pt with the
    standardisation, the measured counts (the Bayes correction) and the held-out pairs' accuracies."""
    import torch
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    counts = json.loads((OUT / "counts.json").read_text())
    others = [qn for qn in paths.QUARTERS if qn != quarter]
    z = [dict(np.load(OUT / f"pairs-{qn}.npz")) for qn in others]
    cat = {k: np.concatenate([p[k] for p in z]) for k in z[0]}
    n_same = sum(counts["by_quarter"][qn]["same"] for qn in others)
    n_diff = sum(counts["by_quarter"][qn]["different"] for qn in others)
    fit_px = np.concatenate([cat["fa"][~cat["val"]], cat["fb"][~cat["val"]]])
    mean, sd = fit_px.mean(0), np.maximum(fit_px.std(0), 1e-6)
    t = {k: torch.from_numpy(((cat[k] - mean) / sd).astype(np.float32)) for k in ("fa", "fb")}
    d = torch.from_numpy(cat["d"].astype(np.int64))
    y = torch.from_numpy(cat["y"].astype(np.float32))
    val = torch.from_numpy(cat["val"])
    tr = torch.nonzero(~val)[:, 0]
    va = torch.nonzero(val)[:, 0]
    pos = float(y[tr].sum())
    w_pos, w_neg = 0.5 / pos, 0.5 / (len(tr) - pos)       # balanced: each class half the loss

    def weights(idx):
        return torch.where(y[idx] > 0, w_pos, w_neg) * len(idx)

    net = _net(torch)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    bce = torch.nn.functional.binary_cross_entropy_with_logits

    def held_loss():
        with torch.no_grad():
            out = []
            for k in range(0, len(va), 65536):
                i = va[k:k + 65536]
                lg = pair_logit(net, t["fa"][i], t["fb"][i], d[i], torch)
                wv = torch.where(y[i] > 0, 0.5 / float(y[va].sum()), 0.5 / float((1 - y[va]).sum()))
                out.append((bce(lg, y[i], reduction="none") * wv).sum())
            return float(sum(out))

    best, best_state, since, history = np.inf, None, 0, []
    g = torch.Generator().manual_seed(SEED)
    t0 = time.perf_counter()
    for epoch in range(EPOCHS):
        perm = tr[torch.randperm(len(tr), generator=g)]
        for k in range(0, len(perm), BATCH):
            i = perm[k:k + BATCH]
            lg = pair_logit(net, t["fa"][i], t["fb"][i], d[i], torch)
            loss = (bce(lg, y[i], reduction="none") * weights(i)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        h = held_loss()
        history.append(round(h, 5))
        log.info("links %s: epoch %d, held-out balanced loss %.5f (%.0f s)", quarter, epoch + 1, h,
                 time.perf_counter() - t0)
        if h < best - 1e-6:
            best, best_state, since = h, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            since += 1
            if since >= PATIENCE:
                break
    net.load_state_dict(best_state)
    prior = float(np.log(n_same / n_diff))
    with torch.no_grad():
        lg = torch.cat([pair_logit(net, t["fa"][va[k:k + 65536]], t["fb"][va[k:k + 65536]], d[va[k:k + 65536]],
                                   torch) for k in range(0, len(va), 65536)]).numpy()
    yv, ww = cat["y"][cat["val"]], cat["word_word"][cat["val"]]
    call = lg + prior > 0
    acc = {"same": float(call[yv == 1].mean()), "different": float((~call[yv == 0]).mean()),
           "different_word_word": float((~call[(yv == 0) & ww]).mean()),
           "different_word_none": float((~call[(yv == 0) & ~ww]).mean()),
           "balanced_call_same": float((lg[yv == 1] > 0).mean()),
           "balanced_call_different": float((lg[yv == 0] <= 0).mean()),
           "held_out_pairs": {"same": int((yv == 1).sum()), "different_word_word": int(((yv == 0) & ww).sum()),
                              "different_word_none": int(((yv == 0) & ~ww).sum())}}
    info = {"quarter": quarter, "trained_on": others, "n_same": n_same, "n_different": n_diff, "log_prior": prior,
            "sampled": {"same": int((cat["y"] == 1).sum()), "different": int((cat["y"] == 0).sum())},
            "epochs": len(history), "held_out_loss": history, "best_held_out_loss": best, "accuracy": acc}
    torch.save({"state": net.state_dict(), "mean": mean, "sd": sd, "info": info}, OUT / f"model-{quarter}.pt")
    (OUT / f"model-{quarter}.json").write_text(json.dumps(info, indent=1) + "\n")
    return info


_MODEL: dict = {}


def link_labels(on: np.ndarray, o, quarter: str) -> np.ndarray:
    """learned-links' words on a page: the components of the `on` px (core px of the held-out quarter) joined where
    the fold's model gives logit + log(n_same / n_different) > 0; a label image (int32, 0 off them)."""
    import torch
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    if quarter not in _MODEL:
        m = torch.load(OUT / f"model-{quarter}.pt", weights_only=False)
        net = _net(torch)
        net.load_state_dict(m["state"])
        net.eval()
        _MODEL[quarter] = (net, m["mean"], m["sd"], m["info"]["log_prior"])
    net, mean, sd, prior = _MODEL[quarter]
    torch.set_num_threads(1)
    ya, xa, yb, xb, d = pairs_of(on)
    ys, xs = np.nonzero(on)
    idx = np.full(on.shape, -1, np.int64)
    idx[ys, xs] = np.arange(len(ys))
    f = torch.from_numpy(((features_at(o, ys, xs) - mean) / sd).astype(np.float32))
    a, b = idx[ya, xa], idx[yb, xb]
    keep = np.zeros(len(a), bool)
    dd = torch.from_numpy(d.astype(np.int64))
    with torch.no_grad():
        for k in range(0, len(a), 262144):
            s = slice(k, k + 262144)
            lg = pair_logit(net, f[a[s]], f[b[s]], dd[s], torch).numpy()
            keep[s] = lg + prior > 0
    n = len(ys)
    _, comp = connected_components(coo_matrix((np.ones(int(keep.sum()), np.int8), (a[keep], b[keep])), shape=(n, n)),
                                   directed=False)
    out = np.zeros(on.shape, np.int32)
    out[ys, xs] = comp + 1
    return out


# --- the boundary diagnostic: the links' evidence summed over a whole boundary ---------------------------------------

KIND_SHORT = {0: "code", 1: "block", 2: "street", 3: "other"}


def pair_logits(on: np.ndarray, o, quarter: str):
    """Every 8-neighbour pair of the `on` px (`pairs_of`) and the fold's balanced logit for it (before the prior)."""
    import torch
    if quarter not in _MODEL:
        m = torch.load(OUT / f"model-{quarter}.pt", weights_only=False)
        net = _net(torch)
        net.load_state_dict(m["state"])
        net.eval()
        _MODEL[quarter] = (net, m["mean"], m["sd"], m["info"]["log_prior"])
    net, mean, sd, _ = _MODEL[quarter]
    ya, xa, yb, xb, d = pairs_of(on)
    ys, xs = np.nonzero(on)
    idx = np.full(on.shape, -1, np.int64)
    idx[ys, xs] = np.arange(len(ys))
    f = torch.from_numpy(((features_at(o, ys, xs) - mean) / sd).astype(np.float32))
    a, b = idx[ya, xa], idx[yb, xb]
    dd = torch.from_numpy(d.astype(np.int64))
    out = np.zeros(len(a), np.float32)
    with torch.no_grad():
        for k in range(0, len(a), 262144):
            s_ = slice(k, k + 262144)
            out[s_] = pair_logit(net, f[a[s_]], f[b[s_]], dd[s_], torch).numpy()
    return (ya, xa, yb, xb), out


def _boundary_job(args):
    """One page's boundaries in the held-out quarter: every two Terra words with core px touching (8 px around) and
    every word's two halves (split at its rectangle's centre along its reading axis), each with the number of its
    crossing pairs and the sum of their logits; and the word pairs crude-map merges with no touching px."""
    import cv2
    import torch
    torch.set_num_threads(1)
    quarter, sheet = args
    q = paths.QUARTERS.index(quarter)
    sd = RD.load_sheet(sheet)
    inq = reader_stage.page_quarters(sd) == q
    o = RD.outputs(RD.run_dir("a", quarter, 0), sheet)
    on = inq & (o.core > 0)
    if not on.any():
        return []
    (ya, xa, yb, xb), lg = pair_logits(on, o, quarter)
    wid = RD.word_ids(sd)
    wa, wb = wid[ya, xa], wid[yb, xb]
    rows = []
    cross = (wa > 0) & (wb > 0) & (wa != wb)
    lo, hi = np.minimum(wa[cross], wb[cross]), np.maximum(wa[cross], wb[cross])
    keys = lo.astype(np.int64) * 1_000_000 + hi
    uk, inv = np.unique(keys, return_inverse=True)
    sums = np.bincount(inv, lg[cross].astype(np.float64), len(uk))
    cnt = np.bincount(inv, minlength=len(uk))
    for k, key in enumerate(uk):
        a_, b_ = int(key // 1_000_000) - 1, int(key % 1_000_000) - 1
        ka, kb = sorted((KIND_SHORT[int(sd.kinds[a_])], KIND_SHORT[int(sd.kinds[b_])]))
        rows.append({"sheet": sheet, "quarter": quarter, "what": "different", "pair": f"{ka}+{kb}",
                     "stacked": None, "n": int(cnt[k]), "sum": float(sums[k])})
    # the merged word pairs with no touching core px (joined through px of no word)
    _, comp = cv2.connectedComponents(on.astype(np.uint8), connectivity=8)
    m = on & (wid > 0)
    words_of = {}
    for c_, w_ in zip(comp[m], wid[m]):
        words_of.setdefault(int(c_), set()).add(int(w_))
    touching = set(int(k) for k in uk)
    apart = 0
    for ws in words_of.values():
        ws = sorted(ws)
        for i in range(len(ws)):
            for j in range(i + 1, len(ws)):
                if ws[i] * 1_000_000 + ws[j] not in touching:
                    apart += 1
    rows.append({"sheet": sheet, "quarter": quarter, "what": "merged_without_touching", "pair": None,
                 "stacked": None, "n": apart, "sum": 0.0})
    # each word's two halves
    same = (wa > 0) & (wa == wb)
    j = wa[same] - 1
    theta = np.radians(sd.angles[j].astype(np.float64))
    u = np.stack([np.cos(theta), -np.sin(theta)], 1)
    centre = sd.words[j].astype(np.float64).mean(1)
    side_a = (((np.stack([xa[same], ya[same]], 1) + 0.5 - centre) * u).sum(1) >= 0)
    side_b = (((np.stack([xb[same], yb[same]], 1) + 0.5 - centre) * u).sum(1) >= 0)
    split = side_a != side_b
    jj = j[split]
    uw, inv = np.unique(jj, return_inverse=True)
    sums = np.bincount(inv, lg[same][split].astype(np.float64), len(uw))
    cnt = np.bincount(inv, minlength=len(uw))
    for k, w_ in enumerate(uw):
        kk = KIND_SHORT[int(sd.kinds[w_])]
        rows.append({"sheet": sheet, "quarter": quarter, "what": "same", "pair": f"{kk} halves", "stacked": None,
                     "n": int(cnt[k]), "sum": float(sums[k])})
    return rows


def _auc(neg: np.ndarray, pos: np.ndarray) -> float:
    """P(a different word's score < a same word's), ties half (Mann-Whitney)."""
    from scipy.stats import rankdata
    if not len(neg) or not len(pos):
        return float("nan")
    r = rankdata(np.r_[neg, pos])
    return float((r[len(neg):].sum() - len(pos) * (len(pos) + 1) / 2) / (len(neg) * len(pos)))


def boundary_diagnostic(workers: int = RD.WORKERS) -> dict:
    """The boundary sums on every held-out quarter (`_boundary_job`): their distributions for different and same
    words, how well the sum and the mean separate them (AUC), by the partner kinds.  Writes OUT/boundaries.json and
    boundaries.parquet."""
    import pandas as pd
    got = RD._pool_map(_boundary_job, [(qn, s) for qn in paths.QUARTERS for s in paths.SHEETS], workers)
    t = pd.DataFrame([r for rows in got for r in rows])
    t.to_parquet(OUT / "boundaries.parquet", index=False)
    apart = int(t.loc[t["what"] == "merged_without_touching", "n"].sum())
    t = t[t["what"] != "merged_without_touching"].copy()
    t["mean"] = t["sum"] / t["n"]
    diff, same = t[t["what"] == "different"], t[t["what"] == "same"]

    def q(v):
        return np.quantile(v, [0.1, 0.25, 0.5, 0.75, 0.9]).round(2).tolist() if len(v) else None

    out = {"different": {"boundaries": len(diff), "sum_quantiles": q(diff["sum"]), "mean_quantiles": q(diff["mean"]),
                         "pairs_quantiles": q(diff["n"]), "sum_below_0": float((diff["sum"] < 0).mean())},
           "same": {"boundaries": len(same), "sum_quantiles": q(same["sum"]), "mean_quantiles": q(same["mean"]),
                    "pairs_quantiles": q(same["n"]), "sum_below_0": float((same["sum"] < 0).mean())},
           "auc_sum": _auc(diff["sum"].to_numpy(), same["sum"].to_numpy()),
           "auc_mean": _auc(diff["mean"].to_numpy(), same["mean"].to_numpy()),
           "merged_without_touching_pairs": apart, "by_pair": {}}
    for pair, d in diff.groupby("pair"):
        out["by_pair"][pair] = {"boundaries": len(d), "sum_median": round(float(d["sum"].median()), 2),
                                "sum_below_0": round(float((d["sum"] < 0).mean()), 3),
                                "auc_sum_vs_same": round(_auc(d["sum"].to_numpy(), same["sum"].to_numpy()), 3),
                                "auc_mean_vs_same": round(_auc(d["mean"].to_numpy(), same["mean"].to_numpy()), 3)}
    # same words matched to each different boundary's number of crossing pairs (the nearest-length same boundary)
    s_n, s_sum = same["n"].to_numpy(), same["sum"].to_numpy()
    order = np.argsort(s_n, kind="stable")
    pos = np.searchsorted(s_n[order], diff["n"].to_numpy())
    pos = np.clip(pos, 0, len(order) - 1)
    matched = s_sum[order][pos]
    out["auc_sum_length_matched"] = _auc(diff["sum"].to_numpy(), matched)
    out["by_quarter"] = {qn: {"different": int((diff["quarter"] == qn).sum()),
                              "same": int((same["quarter"] == qn).sum()),
                              "auc_sum": round(_auc(diff.loc[diff["quarter"] == qn, "sum"].to_numpy(),
                                                    same.loc[same["quarter"] == qn, "sum"].to_numpy()), 3)}
                         for qn in paths.QUARTERS}
    (OUT / "boundaries.json").write_text(json.dumps(out, indent=1) + "\n")
    return out


def main() -> None:
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("pairs", "train", "boundaries"))
    ap.add_argument("--quarter", default="all", choices=(*paths.QUARTERS, "all"))
    ap.add_argument("--workers", type=int, default=RD.WORKERS)
    a = ap.parse_args()
    if a.cmd == "pairs":
        print(json.dumps(build_pairs(a.workers), indent=1))
    elif a.cmd == "boundaries":
        print(json.dumps(boundary_diagnostic(a.workers), indent=1))
    else:
        for qn in (paths.QUARTERS if a.quarter == "all" else (a.quarter,)):
            print(json.dumps(train_fold(qn), indent=1))


if __name__ == "__main__":
    main()
