"""Spike S2, step 3: every reader scored with the same measures, beside the generative reader's records.

    uv run --with python-doctr --with torch==2.15.0 python scripts/spikes/s2_ctc/score.py

Reads infer.py's frames (frames_<R>.npz, direction.npz), crops.py's metadata, and the generative reader's records
(inputs/full_fix1held.records: df61853, every answered label at its four temperatures; inputs/full_fix2held.records:
c626e46, partial) with the prior they were scored with (inputs/prior3.json) and its label list (inputs/labs.pkl).

The code posterior of a CTC reader (survey 3.1): for each printed form s of the code list (prior3 `forms`, the
generative reader's own hypotheses), p(s | x) by CTC's exact forward algorithm on the crop's frames; a code's mass is
the sum of its forms'; "none" is the empty string (the all-blank path); "other" is the remainder 1 - the rest (every
other string, exactly).  The reading direction is summed out: p(s | x) = (1 - q) p(s | crop) + q p(s | crop turned a
half turn), q = p(upside down | crop, Terra's direction) (columns "+dir"; without: Terra's direction alone).  D was
trained on balanced classes, so its output's odds are a likelihood ratio; q puts it on the half-turn rate the
generative reader measured on its training labels for Terra's direction (prior3 `pose.half_turn`: left-to-right
0.003, bottom-to-top 0.02, slanted 0.22, top-to-bottom 0.50; `half_turn_other` 0.05 elsewhere), as that reader's own
half-turn prior does: q = r LR / (r LR + 1 - r).  No code prior: CTC's own
string prior (from Terra's training strings) stands (survey 3.2, beta = 0).  One temperature T divides the logits.

The generative reader's posterior: `posterior.posterior_of` at its temperature and its two masses (p(none), p(other)).

Measures on the owner's answered labels with Terra's box (box-found; crops.py `answered`), the truth the owner's code;
answers outside the code list are counted apart:
  top-1, top-3, mean -log p(truth) (p floored at 1e-15, so one label adds at most 34.5 nats), and a reliability table
  (top-1 probability bins: the labels, their mean probability, the share right); 95% bootstrap intervals (BOOT
  resamples of labels, seeded), and paired intervals for each reader's difference from the generative reader.
  Subsets: every label; the unflagged (the owner's fair sample); the flagged (the scoring set's own doubt).
Calibration constants are cross-fitted two-fold on the labels' folds (`labels.assign_folds`): each fold's constants
are chosen by mean -log p on the other fold.  CTC "cross-fitted": T on GRID_T alone.  CTC "cross-fitted +w": T and
the survey's prior weight (sections 0 and 4.2: "calibrate with a temperature and a prior weight, cross-fitted"):
p(h | x) oc p_ctc(h | x) w_h with w = 1 for a code, W_OTHER for "other", W_NONE for "none" (on a box Terra called a
code, the kind prior; Weinman's mixed vocabulary is this with w_other = 0.001), so the CTC reader has the generative
reader's three constants.  Generative: T on its four temperatures x the two masses on GRID_NONE x GRID_OTHER (the
three constants the spec cross-fits).  Also each at its defaults (CTC T = 1; the generative reader's measured
temperature and masses 0.02 / 0.03).
The box-found answered labels are all codes (no "no label" answer has Terra's box), so the masses of "none" and
"other" are not identifiable on them: cross-fitting drives both readers' to the grid's floor, i.e. to the closed list.
So the like-for-like measure is the closed list itself ("closed"): p(code | x, the box holds a code), "none" and
"other" left out and the codes renormalised (computed in log space), T cross-fitted (the generative reader's four
temperatures; the CTC's GRID_T).  "none" / "other" are judged where they are identifiable: on the held-out quarter's
street, block, other and non-text crops (explaining away, below).
Exploratory (survey 4.3: "a product of the two likelihoods is possible ... decide by the measures"): "<R> x gen", the
product of a CTC reader's and the generative reader's closed cross-fitted posteriors, renormalised (both carry a
string prior, so it is counted twice; reported, not part of the decision).
On the held-out quarter (2000 scan, never trained on): best-path character and word error against Terra's strings by
kind (95% bootstrap intervals over words); and the mean p(code), p(other), p(none) on code, street, block, other and
non-text crops (explaining away), and code top-1 against Terra's own string where it names a code.
Writes results.json, reliability.png, disagreements.png and .json (report.py writes results.md).

CHOSEN AFTER SEEING RESULTS (review S2, I4).  The first scoring (v1 crops) had the CTC posterior at T alone, open
(none / other live), and D's output as q.  These were added after its numbers and the answered misses were seen:
  - the closed-list framing ("closed") and the CTC prior weights ("+w"): B's first numbers showed "other" taking the
    mass of near-miss reads ("C-3R"); the survey had named the prior weight, the closed framing is this script's;
  - the direction rate prior (q on the measured half-turn rate): D alone flipped 19 of 574 upright answered crops;
  - (in crops.py) the slanted-height floor, found by looking at C's answered misses.
v1, scored with this same code, is kept in v1/ (a tie on both measures).
The two closed posteriors differ in their prior (review M4): the CTC's has no code prior and sums a code's forms
unweighted (CTC's own string prior stands, beta = 0), so a code with more printed forms gets their summed mass;
the generative reader's applies prior3's p(code) and p(form | code), as that reader is built.
The generative reader's closed temperature is at its grid's edge (T = 16 chosen on both folds, -log p still falling).
Its records hold each form's tempered log evidence at the four temperatures only, and the temperature acts inside the
sums over placements and the other-text term, so its evidence at a lower T can't be computed from them (rescoring is
the 3.6 h run).  As a sensitivity check only, "gen-fix1 closed, T extrapolated (approx)" fits each form's log
evidence as a + b / T over the four recorded temperatures (the fit's residuals are reported) and extrapolates it to
EXTRAP_T, cross-fitted with the recorded temperatures; it is not the reader's evidence at those temperatures.
Seed replicates (review I3): frames of "<R>s<seed>" models (train_seeds.sh) are scored like their reader.
"""
import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

from sfzmaps.solver import codes as codes_mod  # noqa: E402
from sfzmaps.solver.codereader import labels as L  # noqa: E402
from sfzmaps.solver.codereader import posterior as P  # noqa: E402

READERS = ("A", "B", "C")
SEED_REPLICATES = ("Bs1", "Bs2", "Cs1", "Cs2")      # train_seeds.sh: B and C again with seeds 1 and 2
EXTRAP_T = (6.0, 8.0, 11.0)
CITED = (241, 1810, 1594, 1651, 1249, 1630, 1220, 1401, 326, 232, 387, 699,     # C wrong, generative right
         91, 602, 1359, 1428, 1558, 1820)                                     # C right on clean prints                          # the generative reader's extrapolated temperatures (approx)
GRID_T = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0)
W_OTHER = (1.0, 0.3, 0.1, 0.03, 0.01, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 1e-6)
W_NONE = (1.0, 0.1, 0.01, 1e-3, 1e-4)
GRID_NONE = (1e-4, 1e-3, 0.005, 0.02, 0.05)
GRID_OTHER = (1e-5, 1e-4, 1e-3, 3e-3, 0.01, 0.03, 0.1, 0.3)
FLOOR = 1e-15
BOOT = 2000
LLM_YARDSTICK = {"J7 combined reading, owner's 717 answers": 0.967, "J7, fair unflagged sample": 0.981}
BINS = (0.0, 0.5, 0.8, 0.95, 0.99, 1.0000001)


# --- posteriors -------------------------------------------------------------------------------------------------------

class Frames:
    def __init__(self, reader: str):
        with np.load(C.OUT / f"frames_{reader}.npz") as z:
            self.index = z["index"]
            self.cut, self.cut_off, self.turned, self.turned_off = z["cut"], z["cut_off"], z["turned"], z["turned_off"]
        self.pos = {int(i): k for k, i in enumerate(self.index)}

    def get(self, i: int):
        k = self.pos[i]
        return (self.cut[self.cut_off[k]:self.cut_off[k + 1]], self.turned[self.turned_off[k]:self.turned_off[k + 1]])


def string_probs(logits: np.ndarray, strings: list[str], temp: float) -> np.ndarray:
    lp = F.log_softmax(torch.from_numpy(logits).double() / temp, -1)
    return np.exp(C.string_log_probs(lp, strings))


def ctc_posterior(fr: Frames, i: int, forms: dict, temp: float, q: float | None) -> dict:
    """The code posterior of crop i (see the module); q: p(upside down), None for Terra's direction alone."""
    strings = sorted(forms) + [""]
    cut, turned = fr.get(i)
    p = string_probs(cut, strings, temp)
    if q is not None:
        p = (1 - q) * p + q * string_probs(turned, strings, temp)
    out: dict[str, float] = {}
    for s, v in zip(strings[:-1], p[:-1]):
        out[forms[s]] = out.get(forms[s], 0.0) + float(v)
    out["none"] = float(p[-1])
    out["other"] = max(0.0, 1.0 - sum(out.values()))
    return out


def weighted(post: dict, w_other: float, w_none: float) -> dict:
    """A posterior times the prior weights of "other" and "none" (codes 1), normalised."""
    out = dict(post)
    out["other"] *= w_other
    out["none"] *= w_none
    z = sum(out.values())
    return {h: v / z for h, v in out.items()}


def gen_posterior(ev, prior, temp, none_mass=P.NONE_MASS, other_mass=P.OTHER_MASS) -> dict:
    return P.posterior_of(ev, prior, P.Constants(temperature=temp, none_mass=none_mass, other_mass=other_mass))


def _softmax(logs: dict) -> dict:
    m = max(logs.values())
    if not math.isfinite(m):
        return {h: 1.0 / len(logs) for h in logs}
    z = m + math.log(sum(math.exp(v - m) for v in logs.values()))
    return {h: math.exp(v - z) for h, v in logs.items()}


def ctc_closed(fr: Frames, i: int, forms: dict, temp: float, q: float | None) -> dict:
    """p(code | x, a code): each code's forms' CTC probabilities summed, the codes renormalised, in log space."""
    strings = sorted(forms)
    cut, turned = fr.get(i)
    lp = C.string_log_probs(F.log_softmax(torch.from_numpy(cut).double() / temp, -1), strings)
    if q is not None:
        lt = C.string_log_probs(F.log_softmax(torch.from_numpy(turned).double() / temp, -1), strings)
        lp = np.logaddexp(math.log(max(1 - q, 1e-300)) + lp, math.log(max(q, 1e-300)) + lt)
    by: dict[str, list[float]] = {}
    for st, v in zip(strings, lp):
        by.setdefault(forms[st], []).append(float(v))
    return _softmax({c: float(np.logaddexp.reduce(v)) for c, v in by.items()})


def extrapolated(ev) -> tuple[dict, float]:
    """Each form's log evidence fitted as a + b / T over the recorded temperatures (least squares): {form: (a, b)}
    (None for a form not finite at every temperature), and the largest absolute residual of the fit."""
    x = 1 / np.array(ev.temperatures, float)
    a_ = np.c_[np.ones_like(x), x]
    out, worst = {}, 0.0
    for f, v in ev.forms.items():
        v = np.asarray(v, float)
        if not np.isfinite(v).all():
            out[f] = None
            continue
        coef, *_ = np.linalg.lstsq(a_, v, rcond=None)
        out[f] = (float(coef[0]), float(coef[1]))
        worst = max(worst, float(np.abs(a_ @ coef - v).max()))
    return out, worst


def gen_closed_extrapolated(fits: dict, prior, temp: float) -> dict:
    """`gen_closed` with each form's log evidence from its a + b / T fit at `temp` (approximate; see the module)."""
    by_code: dict[str, list[tuple[float, float]]] = {}
    for f, ab in fits.items():
        if ab is not None:
            by_code.setdefault(prior.forms[f], []).append((ab[0] + ab[1] / temp, prior.form_weights.get(f, 1.0)))
    logs = {}
    for code, p in prior.codes.items():
        vals = by_code.get(code)
        if not vals or p <= 0:
            logs[code] = float("-inf")
            continue
        w = np.array([x for _, x in vals], float)
        logs[code] = math.log(p) + float(np.logaddexp.reduce([v + math.log(x / w.sum()) for (v, _), x in zip(vals, w)]))
    return _softmax(logs)


def gen_closed(ev, prior, temp) -> dict:
    """The generative reader's p(code | ink, a code): `posterior.posterior_of` without "none" and "other"."""
    temp = prior.pose.temperature if temp is None else temp
    i = int(np.argmin([abs(t - temp) for t in ev.temperatures]))
    by_code: dict[str, list[tuple[float, float]]] = {}
    for f, v in ev.forms.items():
        by_code.setdefault(prior.forms[f], []).append((float(v[i]), prior.form_weights.get(f, 1.0)))
    logs = {}
    for code, p in prior.codes.items():
        vals = by_code.get(code)
        if not vals or p <= 0:
            logs[code] = float("-inf")
            continue
        w = np.array([x for _, x in vals], float)
        logs[code] = math.log(p) + float(np.logaddexp.reduce([v + math.log(x / w.sum()) for (v, _), x in zip(vals, w)]))
    return _softmax(logs)


# --- measures ---------------------------------------------------------------------------------------------------------

def per_label(posts: list[dict], truths: list[str]) -> dict[str, np.ndarray]:
    """Per label: right at top-1, within top-3, -log p(truth) (floored), the top-1 probability and hypothesis."""
    top1, top3, nll, conf, best = [], [], [], [], []
    for post, t in zip(posts, truths):
        key = t if t else None
        ranked = sorted(post.items(), key=lambda kv: (-kv[1], kv[0]))
        if key is None:      # the owner says no code here: "none" or "other" is right
            p = post.get("none", 0.0) + post.get("other", 0.0)
            hit1 = ranked[0][0] in ("none", "other")
            hit3 = any(h in ("none", "other") for h, _ in ranked[:3])
        else:
            p = post.get(key, 0.0)
            hit1 = ranked[0][0] == key
            hit3 = any(h == key for h, _ in ranked[:3])
        top1.append(hit1)
        top3.append(hit3)
        nll.append(-math.log(max(p, FLOOR)))
        conf.append(ranked[0][1])
        best.append(ranked[0][0])
    return {"top1": np.array(top1, float), "top3": np.array(top3, float), "nll": np.array(nll),
            "conf": np.array(conf), "best": np.array(best, object)}


def boot_idx(n: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, (BOOT, n))


def interval(x: np.ndarray, idx: np.ndarray) -> list[float]:
    s = x[idx].mean(axis=1)
    return [round(float(np.quantile(s, 0.025)), 4), round(float(np.quantile(s, 0.975)), 4)]


def summary(m: dict, mask: np.ndarray, idx_seed: int = 0) -> dict:
    sel = np.flatnonzero(mask)
    if not len(sel):
        return {"n": 0}
    idx = boot_idx(len(sel), idx_seed)
    out = {"n": int(len(sel))}
    for k in ("top1", "top3", "nll"):
        x = m[k][sel]
        out[k] = round(float(x.mean()), 4)
        out[k + "_ci"] = interval(x, idx)
    out["nll_median"] = round(float(np.median(m["nll"][sel])), 4)
    return out


def paired(a: dict, b: dict, mask: np.ndarray) -> dict:
    """a - b on the same labels, with paired bootstrap intervals."""
    sel = np.flatnonzero(mask)
    idx = boot_idx(len(sel), 1)
    out = {}
    for k in ("top1", "nll"):
        d = a[k][sel] - b[k][sel]
        out[k] = round(float(d.mean()), 4)
        out[k + "_ci"] = interval(d, idx)
    out["a_right_b_wrong"] = int(((a["top1"][sel] == 1) & (b["top1"][sel] == 0)).sum())
    out["a_wrong_b_right"] = int(((a["top1"][sel] == 0) & (b["top1"][sel] == 1)).sum())
    return out


def reliability(m: dict, mask: np.ndarray) -> list[dict]:
    rows = []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        sel = mask & (m["conf"] >= lo) & (m["conf"] < hi)
        if sel.any():
            rows.append({"bin": f"[{lo:.2f}, {min(hi, 1):.2f}{']' if hi > 1 else ')'}", "n": int(sel.sum()),
                         "mean_p": round(float(m["conf"][sel].mean()), 4),
                         "right": round(float(m["top1"][sel].mean()), 4)})
    return rows


def crossfit(score_fn, grid, folds: np.ndarray, truths) -> tuple[dict, dict]:
    """Each fold's constants chosen by mean -log p on the other fold; the per-label measures with them."""
    chosen, merged = {}, None
    cache = {g: per_label(score_fn(g), truths) for g in grid}
    for f in (0, 1):
        fit = folds == 1 - f
        best = min(grid, key=lambda g: (cache[g]["nll"][fit].mean(), str(g)))
        chosen[f] = best
        part = cache[best]
        if merged is None:
            merged = {k: v.copy() for k, v in part.items()}
        sel = folds == f
        for k in merged:
            merged[k][sel] = part[k][sel]
    return merged, {str(k): v for k, v in chosen.items()}


# --- character and word error -------------------------------------------------------------------------------------

def edit(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def best_path(logits: np.ndarray) -> str:
    return C.best_path(torch.from_numpy(logits))


# --- the run ----------------------------------------------------------------------------------------------------------

def records(name: str) -> dict:
    out = {}
    with open(C.INPUTS / name, "rb") as fh:
        while True:
            try:
                r = pickle.load(fh)
            except EOFError:
                break
            out[r[0]] = r
    return out


def main():
    crops = C.load_crops()
    meta = crops.meta
    prior = P.Prior.load(C.INPUTS / "prior3.json")
    forms = prior.forms
    codes = set(prior.codes)
    labs = {(lab.index, tuple(lab.point)): lab for lab in pickle.load(open(C.INPUTS / "labs.pkl", "rb"))}
    q_of = None
    if (C.OUT / "direction.npz").exists():
        with np.load(C.OUT / "direction.npz") as z:
            raw_q = dict(zip(z["index"].tolist(), z["p_flip"].tolist()))
        q_of = {}
        for i, d_raw in raw_q.items():      # D's odds on the measured half-turn rate of Terra's direction
            pd_ = min(max(d_raw, 1e-9), 1 - 1e-9)
            rate = prior.pose.half_turn.get(meta.at[i, "direction"], prior.pose.half_turn_other)
            lr = pd_ / (1 - pd_)
            q_of[i] = rate * lr / (rate * lr + 1 - rate)
    frames = {r: Frames(r) for r in READERS + SEED_REPLICATES if (C.OUT / f"frames_{r}.npz").exists()}
    results: dict = {"readers": sorted(frames), "llm_yardstick": LLM_YARDSTICK}

    # --- the owner's answered labels, box-found ---
    ans = meta[meta["split"] == "answered"]
    keys = [(int(i), tuple(p)) for i, p in zip(ans["index"], ans["point"])]
    truths_all = [a if isinstance(a, str) else None for a in ans["answer"]]     # None: no answer (NaN in pandas)
    in_list = np.array([t is not None and (t == "" or t in codes) for t in truths_all])
    off_list = sorted({t for t, ok in zip(truths_all, in_list) if t is not None and not ok})
    sel_rows = np.flatnonzero(in_list)
    rows = ans.iloc[sel_rows]
    keys = [keys[k] for k in sel_rows]
    truths = [truths_all[k] for k in sel_rows]
    idx = rows.index.to_numpy()
    flagged = rows["flagged"].to_numpy(bool)
    folds = rows["fold"].to_numpy(int)
    every = np.ones(len(rows), bool)
    subsets = {"all": every, "unflagged": ~flagged, "flagged": flagged}
    results["answered"] = {"box_found": int(len(ans)), "scored": int(len(rows)),
                           "outside_code_list": {"n": int(sum(1 for t in truths_all if t is not None) - len(rows)),
                                                 "codes": off_list},
                           "no_answer": int(sum(t is None for t in truths_all)), "flagged": int(flagged.sum()),
                           "folds": {str(f): int((folds == f).sum()) for f in (0, 1, -1)}}

    measures: dict[str, dict] = {}
    chosen: dict[str, dict] = {}
    closed_posts: dict[str, list[dict]] = {}     # each reader's closed cross-fitted posterior per label
    # the generative reader
    gen_recs = {"fix1": records("full_fix1held.records"), "fix2": records("full_fix2held.records")}
    have = {g: np.array([k in r for k in keys]) for g, r in gen_recs.items()}
    results["answered"]["generative_records"] = {g: int(h.sum()) for g, h in have.items()}
    for g, recs in gen_recs.items():
        if not have[g].all():
            continue
        evs = [recs[k][3] for k in keys]
        temps = evs[0].temperatures
        measures[f"gen-{g} default"] = per_label([gen_posterior(e, prior, None) for e in evs], truths)
        for t in temps:
            measures[f"gen-{g} T={t:.1f}"] = per_label([gen_posterior(e, prior, t) for e in evs], truths)
        grid = [(t, n, o) for t in temps for n in GRID_NONE for o in GRID_OTHER]
        m, ch = crossfit(lambda g3: [gen_posterior(e, prior, *g3) for e in evs], grid, folds, truths)
        measures[f"gen-{g} cross-fitted"] = m
        chosen[f"gen-{g} cross-fitted"] = ch
        closed = {t: [gen_closed(e, prior, t) for e in evs] for t in temps}
        m, ch = crossfit(lambda t: closed[t], temps, folds, truths)
        measures[f"gen-{g} closed"] = m
        chosen[f"gen-{g} closed"] = ch
        closed_posts[f"gen-{g}"] = [closed[ch[str(f)]][k] for k, f in enumerate(folds)]
        if g == "fix1":     # sensitivity only: the closed posterior at extrapolated lower temperatures (see module)
            fitted = [extrapolated(e) for e in evs]
            resid = np.array([r for _, r in fitted])
            ext = dict(closed)
            for t in EXTRAP_T:
                ext[t] = [gen_closed_extrapolated(fx, prior, t) for fx, _ in fitted]
            m, ch = crossfit(lambda t: ext[t], tuple(EXTRAP_T) + tuple(temps), folds, truths)
            measures["gen-fix1 closed, T extrapolated (approx)"] = m
            chosen["gen-fix1 closed, T extrapolated (approx)"] = ch
            results["answered"]["extrapolation_residual_nats"] = {
                "median": round(float(np.median(resid)), 3), "q90": round(float(np.quantile(resid, 0.9)), 3),
                "max": round(float(resid.max()), 3),
                "per_temperature": {str(t): {"top1": round(float(per_label(ext[t], truths)["top1"].mean()), 4),
                                             "nll": round(float(per_label(ext[t], truths)["nll"].mean()), 4)}
                                    for t in tuple(EXTRAP_T) + tuple(temps)}}
    # the CTC readers
    for r, fr in frames.items():
        for use_dir in (False, True) if q_of is not None else (False,):
            name = f"{r}{'+dir' if use_dir else ''}"
            q = [q_of[int(i)] if use_dir else None for i in idx]
            measures[f"{name} T=1"] = per_label([ctc_posterior(fr, int(i), forms, 1.0, qq) for i, qq in zip(idx, q)],
                                                truths)
            raw = {t: [ctc_posterior(fr, int(i), forms, t, qq) for i, qq in zip(idx, q)] for t in GRID_T}
            m, ch = crossfit(lambda t: raw[t], GRID_T, folds, truths)
            measures[f"{name} cross-fitted"] = m
            chosen[f"{name} cross-fitted"] = ch
            grid = [(t, wo, wn) for t in GRID_T for wo in W_OTHER for wn in W_NONE]
            m, ch = crossfit(lambda g3: [weighted(p, g3[1], g3[2]) for p in raw[g3[0]]], grid, folds, truths)
            measures[f"{name} cross-fitted +w"] = m
            chosen[f"{name} cross-fitted +w"] = ch
            closed = {t: [ctc_closed(fr, int(i), forms, t, qq) for i, qq in zip(idx, q)] for t in GRID_T}
            m, ch = crossfit(lambda t: closed[t], GRID_T, folds, truths)
            measures[f"{name} closed"] = m
            chosen[f"{name} closed"] = ch
            closed_posts[name] = [closed[ch[str(f)]][k] for k, f in enumerate(folds)]
    for name, posts in list(closed_posts.items()):
        if name.startswith("gen") or "gen-fix1" not in closed_posts:
            continue
        prod = []
        for a_, b_ in zip(posts, closed_posts["gen-fix1"]):
            logs = {c: math.log(max(a_.get(c, 0.0), 1e-300)) + math.log(max(b_.get(c, 0.0), 1e-300)) for c in b_}
            prod.append(_softmax(logs))
        measures[f"{name} x gen (exploratory)"] = per_label(prod, truths)
    table = {}
    for name, m in measures.items():
        table[name] = {s: summary(m, mask) for s, mask in subsets.items()}
        table[name]["reliability"] = reliability(m, every)
    # the saved LLM reads on the same labels, each on the labels it read (the yardstick on this very set)
    llm = {}
    for reader in L.READERS:
        reads = [labs[k].reads.get(reader, "") for k in keys]
        read = np.array([bool(x) for x in reads])
        right = np.array([codes_mod.canon(x) == t for x, t in zip(reads, truths)], float)
        m = {"top1": right, "top3": right, "nll": np.zeros(len(right))}
        llm[reader] = {s: {k: v for k, v in summary(m, mask & read).items() if not k.startswith(("nll", "top3"))}
                       for s, mask in subsets.items()}
    results["answered"]["llm_reads"] = llm
    # the labels the first report cited (the generative reader right, C wrong on a digit crossed by a line; and C's
    # clean wins): each closed reader's top-1 there, seeds included (review I3)
    where = {int(i): k for k, i in enumerate(rows["index"])}
    results["answered"]["cited_labels"] = {
        str(i): {"owner": truths[where[i]], "flagged": bool(flagged[where[i]]),
                 **{n: str(measures[n]["best"][where[i]]) for n in measures if n.endswith("closed")}}
        for i in CITED if i in where}
    results["answered"]["measures"] = table
    results["answered"]["constants"] = chosen
    # each CTC variant against the generative reader's like variant: closed with closed, open cross-fitted (+w or T
    # alone) with its cross-fitted three constants, defaults with defaults
    def like(name: str) -> str:
        if name.endswith(("closed", "(exploratory)")):
            return "gen-fix1 closed"
        if name.endswith("T=1"):
            return "gen-fix1 default"
        return "gen-fix1 cross-fitted"
    results["answered"]["paired_vs_generative"] = {
        name: {"against": like(name), **{s: paired(m, measures[like(name)], mask) for s, mask in subsets.items()}}
        for name, m in measures.items() if not name.startswith("gen") and like(name) in measures}
    ext_name = "gen-fix1 closed, T extrapolated (approx)"
    results["answered"]["paired_vs_generative_extrapolated"] = {
        name: {"against": ext_name, **{s: paired(m, measures[ext_name], mask) for s, mask in subsets.items()}}
        for name, m in measures.items() if name.endswith(" closed") and not name.startswith("gen")
        and ext_name in measures}
    results["answered"]["paired_c_vs_a"] = {
        f"{c} vs {a}": {"against": a, **{s: paired(measures[c], measures[a], mask) for s, mask in subsets.items()}}
        for c, a in (("C+dir closed", "A+dir closed"), ("C closed", "A closed"), ("Cs1+dir closed", "A+dir closed"),
                     ("Cs2+dir closed", "A+dir closed")) if c in measures and a in measures}
    if "fix2" in gen_recs:      # the partial c626e46 run: its labels only, beside fix1 and the CTC readers there
        part = have["fix2"]
        recs2 = gen_recs["fix2"]
        sub = np.flatnonzero(part)
        if len(sub):
            evs2 = [recs2[keys[k]][3] for k in sub]
            tr2 = [truths[k] for k in sub]
            m2 = per_label([gen_posterior(e, prior, None) for e in evs2], tr2)
            res2 = {"n": int(len(sub)), "gen-fix2 default": summary(m2, np.ones(len(sub), bool)),
                    "gen-fix1 default": summary({k: v[sub] for k, v in measures["gen-fix1 default"].items()},
                                                np.ones(len(sub), bool))}
            for name, m in measures.items():
                if not name.startswith("gen"):
                    res2[name] = summary({k: v[sub] for k, v in m.items()}, np.ones(len(sub), bool))
            results["answered"]["fix2_subset"] = res2

    # --- the held-out quarter ---
    qm = meta[meta["split"] == "quarter"]
    qres: dict = {"crops": {k: int(v) for k, v in qm["kind"].value_counts().items()}}
    for r, fr in frames.items():
        errs: dict[str, dict] = {}
        for kind, part in qm.groupby("kind"):
            if kind == "none":
                continue
            e, n, wrong = [], [], []
            for i, text in zip(part.index, part["text"]):
                got = best_path(fr.get(int(i))[0])
                e.append(edit(got, text))
                n.append(len(text))
                wrong.append(got != text)
            e, n, wrong = np.array(e, float), np.array(n, float), np.array(wrong, float)
            bi = boot_idx(len(e), 2)
            cer = e.sum() / n.sum()
            cer_s = e[bi].sum(axis=1) / n[bi].sum(axis=1)
            errs[kind] = {"words": int(len(e)), "cer": round(float(cer), 4),
                          "cer_ci": [round(float(np.quantile(cer_s, 0.025)), 4),
                                     round(float(np.quantile(cer_s, 0.975)), 4)],
                          "wer": round(float(wrong.mean()), 4), "wer_ci": interval(wrong, bi)}
        masses: dict[str, dict] = {}
        temp = 1.0
        for kind, part in qm.groupby("kind"):
            ps = [ctc_posterior(fr, int(i), forms, temp, q_of[int(i)] if q_of else None) for i in part.index]
            code_mass = np.array([sum(v for h, v in p.items() if h not in ("none", "other")) for p in ps])
            p_none = np.array([p["none"] for p in ps])
            p_other = np.array([p["other"] for p in ps])
            top = [max(p, key=p.get) for p in ps]
            masses[kind] = {"crops": len(ps), "p_code": round(float(code_mass.mean()), 4),
                            "p_other": round(float(p_other.mean()), 4), "p_none": round(float(p_none.mean()), 4),
                            "top_is_code": round(float(np.mean([t not in ("none", "other") for t in top])), 4),
                            "top_is_none": round(float(np.mean([t == "none" for t in top])), 4),
                            "top_is_other": round(float(np.mean([t == "other" for t in top])), 4)}
            if kind == "zoning code":
                want = [L.code_of(t, codes) or "" for t in part["text"]]
                named = np.array([bool(w) for w in want])
                hit = np.array([t == w for t, w in zip(top, want)], float)
                masses[kind]["terra_names_code"] = int(named.sum())
                masses[kind]["top1_vs_terra"] = round(float(hit[named].mean()), 4)
                masses[kind]["top1_vs_terra_ci"] = interval(hit[named], boot_idx(int(named.sum()), 3))
        qres[r] = {"errors": errs, "masses_dir_T1": masses}
    results["quarter"] = qres

    # --- direction classifier on the held-out quarter (Terra's direction as truth; Terra errs too) ---
    qi = qm[qm["kind"] != "none"].index
    pf = np.array([raw_q[int(i)] for i in qi]) if q_of else np.zeros(0)
    pq = np.array([q_of[int(i)] for i in qi]) if q_of else np.zeros(0)
    ai = meta.index[meta["split"] == "answered"]
    results["direction"] = {} if not q_of else {
        "answered_by_terra_direction": {
            d: {"n": int((meta.loc[ai, "direction"] == d).sum()),
                "D_alone_upside_down": int(sum(raw_q[int(i)] > 0.5 for i in ai if meta.at[i, "direction"] == d)),
                "with_rate_upside_down": int(sum(q_of[int(i)] > 0.5 for i in ai if meta.at[i, "direction"] == d))}
            for d in sorted(set(meta.loc[ai, "direction"]))},
        "quarter_with_rate_upside_down": int((pq > 0.5).sum()),
        "quarter_words": int(len(qi)), "said_upside_down": int((pf > 0.5).sum()),
                            "mean_p_flip": round(float(pf.mean()), 4),
                            "said_upside_down_by_terra_direction": {
                                d: int((pf[(qm.loc[qi, 'direction'] == d).to_numpy()] > 0.5).sum())
                                for d in sorted(set(qm.loc[qi, "direction"]))}}

    (C.OUT / "results.json").write_text(json.dumps(results, indent=1, default=str) + "\n")
    pictures(crops, rows, idx, truths, measures, flagged)
    print(json.dumps({k: v["all"] for k, v in table.items()}, indent=0, default=str)[:6000])


def pictures(crops, rows, idx, truths, measures, flagged):
    import cv2
    # the reliability diagram: every reader with intervals' worth of labels per bin
    names = [n for n in measures if n.endswith("closed") and (n.startswith("gen") or n[1] in "+ ")]
    h, w = 420, 520
    img = np.full((h, w, 3), 255, np.uint8)
    x0, y0, s = 50, 380, 330
    cv2.line(img, (x0, y0), (x0 + s, y0 - s), (180, 180, 180), 1)
    cv2.rectangle(img, (x0, y0 - s), (x0 + s, y0), (0, 0, 0), 1)
    colours = [(31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40), (148, 103, 189), (140, 86, 75),
               (227, 119, 194), (127, 127, 127)]
    for k, n in enumerate(names):
        col = colours[k % len(colours)][::-1]
        pts = [(x0 + int(r["mean_p"] * s), y0 - int(r["right"] * s)) for r in reliability(measures[n],
                                                                                           np.ones(len(truths), bool))]
        for a, b in zip(pts[:-1], pts[1:]):
            cv2.line(img, a, b, col, 2)
        for p in pts:
            cv2.circle(img, p, 3, col, -1)
        cv2.putText(img, n, (x0 + s + 8, 60 + 18 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1, cv2.LINE_AA)
    cv2.putText(img, "reliability on the answered labels (x: top-1 p, y: share right)", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(img, "bins 0-.5, .5-.8, .8-.95, .95-.99, .99-1", (x0, y0 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(C.OUT / "reliability.png"), img)
    # the disagreements: the best CTC reader (cross-fitted, by mean -log p on all answered) vs the generative reader
    ctc = [n for n in measures if n.endswith("closed") and n[0] in "BC" and n[1] in "+ "]
    if not ctc or "gen-fix1 closed" not in measures:
        return
    best = min(ctc, key=lambda n: measures[n]["nll"].mean())
    a, g = measures[best], measures["gen-fix1 closed"]
    diff = np.flatnonzero(a["best"] != g["best"])
    order = sorted(diff, key=lambda k: (-(a["top1"][k] - g["top1"][k]), k))
    crops_, caps = [], []
    for k in order:
        i = int(idx[k])
        crops_.append(crops[i])
        tag = "CTC right" if a["top1"][k] > g["top1"][k] else ("gen right" if g["top1"][k] > a["top1"][k]
                                                                 else "both wrong")
        flag = " FLAG" if flagged[k] else ""
        caps.append(f"{tag}: owner {truths[k] or '(none)'}{flag} #{int(rows['index'].iat[k])}\n"
                    f"{best.split()[0]} {a['best'][k]} {a['conf'][k]:.2f} | gen {g['best'][k]} {g['conf'][k]:.2f}")
    title = (f"spike S2: answered labels where {best} and the generative reader (fix1, closed) disagree on "
             f"top-1 ({len(order)}); judge by eye")
    C.contact_sheet(crops_, caps, C.OUT / "disagreements.png", cols=5, scale=2, cell_w=260, title=title)
    (C.OUT / "disagreements.json").write_text(json.dumps(
        {"reader": best, "labels": [{"index": int(rows["index"].iat[k]), "owner": truths[k],
                                     "flagged": bool(flagged[k]), "ctc": a["best"][k], "ctc_p": float(a["conf"][k]),
                                     "gen": g["best"][k], "gen_p": float(g["conf"][k])} for k in order]},
        indent=1) + "\n")


if __name__ == "__main__":
    main()
