"""Spike S0: the 2000 solver's map when the only reading is our own template reader's posterior (no LLM reads).

A measurement, not shipped code (brief: .superpowers/sdd/2026-11-16-search-move-and-reader-spikes/task-S01-brief.md;
the shared pieces and how a posterior enters the energy: reader_spike.py).

    uv run python scripts/spikes/s0_reader_only.py solve     # control, s0 and s0-mix, four quarters each
    uv run python scripts/spikes/s0_reader_only.py report    # results.json, results.md and the pictures

Writes into data/solver/experiments/spike-s0/: runs/ (the solves), results.json, results.md, pictures/.
  - The control (the stage's readings through this path) must give the stage's `final` lot for lot and its six
    measures exactly; the report stops otherwise.
  - s0 and s0-mix against final: the six measures (every lot), and paired block-bootstrap intervals (`reader_spike.
    Blocks`: blocks resampled, the same resample for both runs, BOOT_DRAWS draws, fixed seed) for wrong lots,
    incidents and labels right.
  - The reader's own top-1 on the owner's answered labels (labs.pkl's `answer`), beside the stage's readings' top-1
    on the same labels; its misread and miss rates there (the extra S1 point); its reliability.
  - Where s0's map is worse: the five changed sets (connected lots whose code differs between s0 and final) that
    add the most wrong lots, each drawn (`picture`)."""
import os

for _k, _v in {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1",
               "MKL_NUM_THREADS": "1", "OPENCV_FOR_THREADS_NUM": "1", "NUMEXPR_NUM_THREADS": "1"}.items():
    os.environ.setdefault(_k, _v)

import json  # noqa: E402
import math  # noqa: E402
import pickle  # noqa: E402
import sys  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from scipy.sparse import coo_matrix  # noqa: E402
from scipy.sparse.csgraph import connected_components  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reader_spike as R  # noqa: E402

from sfzmaps.solver import grid, runs  # noqa: E402
from sfzmaps.solver import labels as labels_mod  # noqa: E402
from sfzmaps.solver.city import city, load_run, run_file  # noqa: E402
from sfzmaps.solver.codes import canon  # noqa: E402
from sfzmaps.solver.measures import data_error_lots  # noqa: E402
from sfzmaps.solver import report as report_mod  # noqa: E402
from sfzmaps.solver.trust import load_prints  # noqa: E402

cv2.setNumThreads(1)
OUT = R.S0_DIR
RUNS = ("control", "s0", "s0-mix")
KEY = ("lines_drawn", "drawn_are_table_lines", "labels_right", "holes", "incidents", "wrong_lots")
PICTURES = 5
S = 2               # image px per grid px
MARGIN = 50         # grid px around a changed set


# --- the control -----------------------------------------------------------------------------------------------------

def check_control() -> dict:
    """The control against final: every quarter's solve lot for lot, the join, the six measures (every lot, and with
    final's own data-error list left out, against report.json's two columns)."""
    out = {"quarters_equal": {}}
    for q in R.QUARTERS:
        a = load_run_q("control", q, R.S0_DIR)
        b = load_run_q(runs.FINAL_RUN, q, runs.SOLVE_DIR)
        out["quarters_equal"][q] = bool(np.array_equal(a, b))
    out["join_equal"] = bool(np.array_equal(R.codes_of("control"), R.codes_of(runs.FINAL_RUN)))
    rep = json.loads((runs.SOLVE_DIR / "report.json").read_text())["measures"]
    every = R.six("control")
    left = report_mod.six_measures(R.codes_of("control"), data_error_lots(runs.SOLVE_DIR))
    out["every_lot_equal"] = every == rep["every_lot"]
    out["left_out_equal"] = left == rep["data_errors_left_out"]
    out["every_lot"] = {k: every[k] for k in KEY}
    if not (all(out["quarters_equal"].values()) and out["join_equal"] and out["every_lot_equal"]
            and out["left_out_equal"]):
        raise SystemExit(f"the control does not reproduce final: {out}")
    return out


def load_run_q(run: str, q: str, folder) -> np.ndarray:
    Z = np.load(run_file(run, f"-{q}", folder), allow_pickle=True)
    return np.array([canon(x) for x in np.asarray(Z["names"]).astype(str)])[Z["pred"]]


def solve_logs(run: str) -> dict:
    """Each quarter's log record (reader_spike.solve_one's json): energies, info, the solver files' hashes."""
    _, folder = R.kind_of(run)
    return {q: json.loads(run_file(run, f"-{q}", folder).with_suffix(".json").read_text()) for q in R.QUARTERS}


# --- the reader on the owner's answers -------------------------------------------------------------------------------

def answered() -> list:
    """The owner's answered labels (labs.pkl, the records' labels): answer not None."""
    labs = pickle.load(open(R.INPUTS / "labs.pkl", "rb"))
    return [lab for lab in labs if lab.answer is not None]


def stage_tops() -> dict:
    """prints row -> the stage's top reading of a charged label (its largest trust among the kept, own-lot
    readings), from the readings of the fit that held its lot's quarter out."""
    C = city()
    voc = R.vocabulary()
    out = {}
    for k, q in enumerate(R.QUARTERS):
        rd = labels_mod.readings(q, voc)
        best = {}
        for r in np.flatnonzero(~rd.far):
            lab = int(rd.label[r])
            if C.quarter[rd.lot[r]] != k:
                continue
            if rd.trust[r] > best.get(lab, (None, -1.0))[1]:
                best[lab] = (rd.names[rd.code[r]], float(rd.trust[r]))
        out.update(best)
    return out


def reader_on_answers(post: dict) -> dict:
    """Top-1 of the reader (its posterior's top over codes, "none" and "other") and of the stage's readings on the
    answered labels whose answer is a code; the "no code printed" answers apart; the misread and miss rates on the
    answered labels the solver charges; the reader's reliability by its top's mass; its confusions."""
    labs = answered()
    tops = stage_tops()
    charged = R.charged_labels()
    voc = R.vocabulary()
    rows = []
    skipped = 0
    for lab in labs:
        truth = canon(lab.answer) if lab.answer else ""
        p = post.get(lab.index)
        if p is None:                                   # a case on no print (row -1): no solver label
            skipped += 1
            continue
        top = max(p, key=p.get)
        rt = canon(top) if top not in ("none", "other") else top
        st, trust = tops.get(lab.index, (None, 0.0))
        t = min(max(trust, 1e-3), 1 - 1e-3)
        a, b = R.E.TOP_READING_FIT
        sq = 1 / (1 + math.exp(-(a + b * math.log(t / (1 - t)))))     # the chance the energy gives that top
        rows.append({"index": lab.index, "truth": truth, "reader": rt, "p": p[top], "stage": st, "stage_q": sq,
                     "charged": lab.index in charged, "flagged": lab.flagged, "kind": lab.kind,
                     "box": lab.box is not None, "in_vocab": truth in voc})
    code = [r for r in rows if r["truth"]]
    n = len(code)

    def share(sel, f):
        sel = list(sel)
        return {"right": int(sum(f(r) for r in sel)), "of": len(sel),
                "share": (sum(f(r) for r in sel) / len(sel)) if sel else None}

    reader_right = lambda r: r["reader"] == r["truth"]           # noqa: E731
    stage_right = lambda r: r["stage"] == r["truth"]             # noqa: E731
    out = {"answered_labels": len(rows) + skipped, "on_no_print": skipped, "with_code": n, "no_code_printed": len(rows) - n,
           "answers_not_table_codes": sum(not r["in_vocab"] for r in code),
           "reader_top1": share(code, reader_right), "stage_top1": share(code, stage_right),
           "reader_top1_unflagged": share((r for r in code if not r["flagged"]), reader_right),
           "stage_top1_unflagged": share((r for r in code if not r["flagged"]), stage_right),
           "reader_top1_box": share((r for r in code if r["box"]), reader_right),
           "reader_top1_point": share((r for r in code if not r["box"]), reader_right),
           "no_code_printed_reader_says_none_or_other": sum(r["reader"] in ("none", "other")
                                                            for r in rows if not r["truth"])}
    ch = [r for r in code if r["charged"]]
    rmiss = sum(r["reader"] in ("none", "other") for r in ch)
    rmis = sum(r["reader"] not in ("none", "other") and r["reader"] != r["truth"] for r in ch)
    smis = sum(r["stage"] is not None and r["stage"] != r["truth"] for r in ch)
    smiss = sum(r["stage"] is None for r in ch)
    out["rates_on_charged_answered"] = {
        "labels": len(ch), "reader_miss": rmiss / len(ch), "reader_misread": rmis / len(ch),
        "stage_miss": smiss / len(ch), "stage_misread": smis / len(ch)}
    # how sure each was when it misread: the reader's mass on its wrong top; the chance the energy gives the stage's
    # wrong top reading (TOP_READING_FIT of its trust)
    rm = [r["p"] for r in ch if r["reader"] not in ("none", "other") and r["reader"] != r["truth"]]
    sm = [r["stage_q"] for r in ch if r["stage"] is not None and r["stage"] != r["truth"]]
    out["misread_confidence"] = {
        "reader": {"misreads": len(rm), "mean": float(np.mean(rm)), "at_least_0.9": int(sum(x >= 0.9 for x in rm))},
        "stage": {"misreads": len(sm), "mean": float(np.mean(sm)), "at_least_0.9": int(sum(x >= 0.9 for x in sm))},
        "labels": len(ch)}
    bins = [0, 0.5, 0.8, 0.95, 0.99, 1.0001]
    rel = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        sel = [r for r in code if lo <= r["p"] < hi]
        if sel:
            rel.append({"from": lo, "to": min(hi, 1.0), "labels": len(sel),
                        "mean_p": float(np.mean([r["p"] for r in sel])),
                        "right": float(np.mean([reader_right(r) for r in sel]))})
    out["reader_reliability"] = rel
    conf = Counter((r["truth"], r["reader"]) for r in code if not reader_right(r))
    out["reader_confusions"] = [[t, g, n_] for (t, g), n_ in conf.most_common(15)]
    sconf = Counter((r["truth"], r["stage"]) for r in code if not stage_right(r))
    out["stage_confusions"] = [[t, g, n_] for (t, g), n_ in sconf.most_common(10)]
    model = R.confusion()
    covered = sum(1 for (t, g), n_ in conf.items() if t in model and g in model[t][0])
    out["reader_confusions_in_s1_model"] = {"pairs": len(conf), "in_model": covered,
                                            "misreads": sum(n_ for (t, g), n_ in conf.items()
                                                            if g not in ("none", "other")),
                                            "misreads_in_model": sum(n_ for (t, g), n_ in conf.items()
                                                                     if t in model and g in model[t][0])}
    return out


def none_other_by_size(post: dict) -> list[dict]:
    """Over every charged label (answered or not): how often the reader's top is "none" or "other", by the height
    of Terra's word (px; box labels), and for the point labels."""
    labs = [lab for lab in pickle.load(open(R.INPUTS / "labs.pkl", "rb")) if lab.index in R.charged_labels()]
    out = []
    for lo, hi in ((0, 20), (20, 26), (26, 32), (32, 40), (40, 46), (46, 1000)):
        sel = [lab for lab in labs if lab.box is not None and lo <= lab.word[1] < hi]
        n = sum(max(post[lab.index], key=post[lab.index].get) in ("none", "other") for lab in sel)
        out.append({"from": lo, "to": hi, "labels": len(sel), "none_or_other": n})
    sel = [lab for lab in labs if lab.box is None]
    out.append({"from": None, "to": None, "labels": len(sel),
                "none_or_other": sum(max(post[lab.index], key=post[lab.index].get) in ("none", "other")
                                     for lab in sel)})
    return out


class Energies:
    """Each quarter's energy as solve_joint minimises it, for final (the stage's readings) and for s0 (the reader's
    P), with each run's quarter solve as code indices: to ask, for a changed set, which energy prefers which answer
    there (`energy_test`)."""

    def __init__(self):
        self._cache = {}

    def get(self, run: str, q: str):
        if (run, q) in self._cache:
            return self._cache[(run, q)]
        from sfzmaps.solver import energy as E
        from sfzmaps.solver.city import wall_odds
        from sfzmaps.solver.placement import load_options
        C = city()
        odds = wall_odds(q)
        options = load_options(q, runs.SOLVE_DIR)
        if run == runs.FINAL_RUN:
            rd = labels_mod.readings(q, R.vocabulary())
            U, _, _ = E.label_costs(C.n, len(rd.names), rd, options, 1.0)
            folder = runs.SOLVE_DIR
        else:
            charges, rd, _ = R.s0_problem(q, mixed=(run == "s0-mix"))
            U, _, _ = R.label_costs_from(C.n, len(rd.names), charges, options, 1.0)
            folder = R.S0_DIR
        U = U + 1e-4 * E.path_costs(C.n, len(rd.names), C.a, C.b, odds, rd)    # solve_joint's tie-break
        U -= U.min(1, keepdims=True)
        Z = np.load(run_file(run, f"-{q}", folder), allow_pickle=True)
        first = {}
        for i, n in enumerate(rd.names):
            first.setdefault(n, i)
        x = np.array([first[canon(n)] for n in np.asarray(Z["names"]).astype(str)[Z["pred"]]], int)
        self._cache[(run, q)] = (U, odds, x)
        return self._cache[(run, q)]


def energy_test(cs: dict, E_: Energies, run: str = "s0") -> dict:
    """For a changed set (its lots in its quarter q): s0's energy of s0's quarter-q answer with the set's lots given
    final's codes, minus that of s0's own answer (below 0: s0's energy prefers final's codes there, so s0's search
    missed them: the search, not the reading); and final's energy of final's answer with s0's codes there, minus
    final's (below 0: final's search missed s0's answer)."""
    from sfzmaps.solver import energy as E
    C = city()
    q = cs["quarter"]
    lots = cs["lots"][C.quarter[cs["lots"]] == R.QUARTERS.index(q)]
    Us, odds, xs = E_.get(run, q)
    Uf, _, xf = E_.get(runs.FINAL_RUN, q)
    ys, yf = xs.copy(), xf.copy()
    ys[lots], yf[lots] = xf[lots], xs[lots]
    return {"quarter": q, "lots_in_quarter": int(len(lots)),
            "s0_energy_of_final_codes_minus_own": round(E.energy(Us, C.a, C.b, odds, ys, 1.0)
                                                        - E.energy(Us, C.a, C.b, odds, xs, 1.0), 3),
            "final_energy_of_s0_codes_minus_own": round(E.energy(Uf, C.a, C.b, odds, yf, 1.0)
                                                        - E.energy(Uf, C.a, C.b, odds, xf, 1.0), 3)}


# --- where s0 is worse -----------------------------------------------------------------------------------------------

def changed_sets(run: str, base: str = runs.FINAL_RUN) -> list[dict]:
    """The lots whose code differs between `run` and `base`, joined through any border into connected sets; per
    set its lots, how many scored lots each run gets wrong, and its commonest (table, base, run) codes; the sets
    sorted by the wrong lots `run` adds, most first."""
    C = city()
    x, y = R.codes_of(run), R.codes_of(base)
    diff = x != y
    m = diff[C.a] & diff[C.b]
    _, comp = connected_components(coo_matrix((np.ones(int(m.sum())), (C.a[m], C.b[m])), shape=(C.n, C.n)),
                                   directed=False)
    wrong_x, wrong_y = C.scored & (x != C.table), C.scored & (y != C.table)
    ix = np.flatnonzero(diff)
    order = np.argsort(comp[ix], kind="stable")
    groups = np.split(ix[order], np.flatnonzero(np.diff(comp[ix][order])) + 1)
    blk = C.lots["blklot"].astype(str).str[:4].values
    out = []
    for g in groups:
        if not len(g):
            continue
        add = int(wrong_x[g].sum()) - int(wrong_y[g].sum())
        trio = Counter(zip(C.table[g], y[g], x[g])).most_common(1)[0][0]
        out.append({"lots": g, "n": int(len(g)), "added_wrong": add, "wrong_run": int(wrong_x[g].sum()),
                    "wrong_base": int(wrong_y[g].sum()), "table": trio[0], "base_code": trio[1],
                    "run_code": trio[2], "quarter": R.QUARTERS[int(Counter(C.quarter[g]).most_common(1)[0][0])],
                    "blocks": [b for b, _ in Counter(blk[g]).most_common(3)]})
    out.sort(key=lambda s: (-s["added_wrong"], int(s["lots"][0])))
    return out


def picture(cs: dict, rank: int, post: dict, tops: dict, out: Path, run: str = "s0", tag: str = "worse") -> tuple:
    """One changed set drawn at 2x the grid: grey page ink; the set's lots tinted orange; the table's district
    borders (green, widest), final's (blue, middle) and the run's (red, thin, on top); every label in the crop as a
    dot with the stage's top reading and trust and the reader's top and its mass (red text where they differ)."""
    C = city()
    a, b = C.a, C.b
    codes_r, codes_f = R.codes_of(run), R.codes_of(runs.FINAL_RUN)
    both = C.known[a] & C.known[b]
    tline = both & (C.table[a] != C.table[b])
    d_run, d_fin = codes_r[a] != codes_r[b], codes_f[a] != codes_f[b]
    lot, grown = C.lot_raster, C.grown_raster
    inD = np.zeros(C.n + 1, bool)
    inD[cs["lots"]] = True
    rr, cc = np.nonzero(inD[np.where(lot >= 0, lot, C.n)])
    r0, r1 = max(rr.min() - MARGIN, 0), min(rr.max() + MARGIN, grid.HEIGHT)
    c0, c1 = max(cc.min() - MARGIN, 0), min(cc.max() + MARGIN, grid.WIDTH)
    ink = C.scan()[r0:r1, c0:c1]
    L, G = lot[r0:r1, c0:c1], grown[r0:r1, c0:c1]
    H, W = L.shape
    keys = a.astype(np.int64) * C.n + b
    pair_px = np.full((H, W), -1, np.int64)
    for gA, gB, s1, s2 in ((G[:, :-1], G[:, 1:], (slice(None), slice(0, W - 1)), (slice(None), slice(1, W))),
                           (G[:-1], G[1:], (slice(0, H - 1), slice(None)), (slice(1, H), slice(None)))):
        lo_, hi_ = np.minimum(gA, gB).astype(np.int64), np.maximum(gA, gB).astype(np.int64)
        k = np.minimum(np.searchsorted(keys, lo_ * C.n + hi_), len(keys) - 1)
        ok = (gA != gB) & (keys[k] == lo_ * C.n + hi_)
        for sl in (s1, s2):
            sub = pair_px[sl]
            sub[ok] = k[ok]
            pair_px[sl] = sub
    has, pk = pair_px >= 0, np.maximum(pair_px, 0)

    def up(mk):
        return np.kron(mk.astype(np.uint8), np.ones((S, S), np.uint8))

    def wide(mk, w):
        u = up(mk)
        return cv2.dilate(u, np.ones((w, w), np.uint8)) > 0 if w > 1 else u > 0

    img = np.full((H * S, W * S, 3), 255, np.uint8)
    Ls = np.where(L >= 0, L, C.n)
    img[up(inD[Ls]) > 0] = (150, 200, 255)
    g = up(ink) > 0
    img[g] = (img[g] * 0.35 + np.array([110, 110, 110]) * 0.65).astype(np.uint8)
    TABLE, FINAL, RUN = (60, 180, 60), (220, 120, 20), (40, 30, 220)
    img[wide(has & tline[pk], 9)] = TABLE
    img[wide(has & d_fin[pk], 5)] = FINAL
    img[wide(has & d_run[pk], 2)] = RUN
    P = load_prints()
    cr = grid.to_grid(P[["x", "y"]].to_numpy())
    differ = []
    for j, (c, r) in enumerate(cr):
        if not (c0 <= c < c1 and r0 <= r < r1) or j not in post:
            continue
        x, y = int((c - c0) * S), int((r - r0) * S)
        st = tops.get(j)
        p = post[j]
        top = max(p, key=p.get)
        rt = canon(top) if top not in ("none", "other") else top
        txt = f"{st[0] if st else '-'} {st[1]:.2f} | {rt} {p[top]:.2f}" if st else f"- | {rt} {p[top]:.2f}"
        colour = (0, 0, 0) if (st and st[0] == rt) else (0, 0, 200)
        if not (st and st[0] == rt):
            differ.append({"print": j, "stage": st[0] if st else None, "trust": round(st[1], 3) if st else None,
                           "reader": rt, "mass": round(float(p[top]), 3)})
        cv2.circle(img, (x, y), 4, colour, -1)
        for col, th in (((255, 255, 255), 3), (colour, 1)):
            cv2.putText(img, txt, (x + 6, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, th, cv2.LINE_AA)
    lines = [(f"S0 {tag} #{rank}: {cs['n']} changed lots (quarter {cs['quarter']}, blocks "
              f"{', '.join(cs['blocks'])}); wrong: final {cs['wrong_base']}, S0 {cs['wrong_run']} "
              f"({cs['added_wrong']:+}); mostly table {cs['table']}, final {cs['base_code']}, S0 {cs['run_code']}",
              None),
             (f"2x the grid (1 image px = 2 ft); crop {W * 4} x {H * 4} ft; grey: page ink; orange tint: the changed "
              "lots", None),
             ("table district border (widest)", TABLE), ("final's border (middle)", FINAL),
             ("S0's border (thin, on top)", RUN),
             ("dot + text: a label: the stage's top reading and trust | the template reader's top and its mass; "
              "red where they differ", None)]
    strip = np.full((22 * len(lines) + 8, img.shape[1], 3), 255, np.uint8)
    for i, (text, col) in enumerate(lines):
        yy, xx = 20 + 22 * i, 6
        if col is not None:
            cv2.rectangle(strip, (xx, yy - 12), (xx + 22, yy + 2), col, -1)
            xx += 30
        cv2.putText(strip, text, (xx, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    img = np.vstack([strip, img])
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"s0_{tag}_{rank}.png"
    cv2.imwrite(str(f), img)
    return f, differ


# --- the report ------------------------------------------------------------------------------------------------------

def fmt(k, v):
    return f"{v:.2%}" if k in ("lines_drawn", "drawn_are_table_lines", "labels_right") else f"{v:,}"


def build() -> dict:
    ctl = check_control()
    post, temperature = R.posteriors()
    B = R.Blocks()
    W = B.weights()
    s_final = B.stats(R.codes_of(runs.FINAL_RUN))
    res = {"control": ctl, "temperature": temperature, "floor": R.FLOOR, "records": list(R.RECORDS),
           "prior": R.PRIOR, "bootstrap": {"draws": R.BOOT_DRAWS, "seed": R.BOOT_SEED, "blocks": B.nb},
           "final": {k: R.six(runs.FINAL_RUN)[k] for k in KEY}
           | {"wrong_lots_outside_1": R.wrong_outside_1(runs.FINAL_RUN)}, "runs": {}}
    hashes = {q: v["solver_sha256"] for q, v in solve_logs("control").items()}
    for run in RUNS:
        logs = solve_logs(run)
        m = R.six(run)
        entry = {"measures": {k: m[k] for k in KEY} | {"wrong_lots_outside_1": R.wrong_outside_1(run)},
                 "companions": {k: m[k] for k in m if k not in KEY},
                 "info": {q: v["info"] for q, v in logs.items()},
                 "solved_energy": {q: next(s for s in v["log"] if "solved" in s).split("solved ")[-1]
                                   for q, v in logs.items()},
                 "search_gap": {q: json.loads(run_file(run, f"-{q}", R.S0_DIR).with_suffix(".gap.json").read_text())
                                ["final_answer_minus_own"] for q in R.QUARTERS},
                 # the solver files as loaded (hashed right after the imports; the solve path imports nothing later)
                 "same_solver_files_as_control": all(v["solver_sha256"] == hashes[q] for q, v in logs.items()),
                 # another agent edited src/ while these ran: the files on disk after the solve (not what it ran)
                 "files_on_disk_changed_during_solve": sorted(q for q, v in logs.items()
                                                              if v["hashes_after"] != v["solver_sha256"])}
        if run != "control":
            entry["vs_final"] = R.paired(W, B.stats(R.codes_of(run)), s_final)
        res["runs"][run] = entry
    res["reader"] = reader_on_answers(post)
    res["reader_none_other_by_size"] = none_other_by_size(post)
    sets = changed_sets("s0")
    res["changed_sets"] = {"sets": len(sets), "worse": sum(s["added_wrong"] > 0 for s in sets),
                           "better": sum(s["added_wrong"] < 0 for s in sets),
                           "added_by_worse": sum(s["added_wrong"] for s in sets if s["added_wrong"] > 0),
                           "removed_by_better": -sum(s["added_wrong"] for s in sets if s["added_wrong"] < 0)}
    tops = stage_tops()
    pics = []
    E_ = Energies()
    for i, cs in enumerate(sets[:PICTURES], 1):
        f, differ = picture(cs, i, post, tops, OUT / "pictures")
        pics.append({k: v for k, v in cs.items() if k != "lots"} | {"file": str(f.relative_to(OUT)),
                                                                     "labels_differing": differ,
                                                                     "energy_test": energy_test(cs, E_)})
    res["worse_sets"] = pics
    better = [cs for cs in sets[::-1] if cs["added_wrong"] < 0][:5]
    res["better_sets"] = [{k: v for k, v in cs.items() if k != "lots"} for cs in better]
    f, differ = picture(better[0], 1, post, tops, OUT / "pictures", tag="better")
    res["better_sets"][0].update(file=str(f.relative_to(OUT)), labels_differing=differ,
                                 energy_test=energy_test(better[0], E_))
    att = {}
    for cs in sets:
        if cs["added_wrong"] == 0:
            continue
        et = energy_test(cs, E_)
        side = "worse" if cs["added_wrong"] > 0 else "better"
        cause = "search" if et["s0_energy_of_final_codes_minus_own"] < 0 else "readings"
        a = att.setdefault(side, {}).setdefault(cause, {"sets": 0, "wrong_lots": 0})
        a["sets"] += 1
        a["wrong_lots"] += abs(cs["added_wrong"])
    res["attribution"] = att
    res["changed_sets"]["net_without_largest_better"] = (res["changed_sets"]["added_by_worse"]
                                                         - res["changed_sets"]["removed_by_better"]
                                                         - better[0]["added_wrong"])
    return res


def markdown(res: dict) -> str:
    f, runs_ = res["final"], res["runs"]
    out = ["# Spike S0: the template reader's posterior as the only reading", "",
           "The 2000 solver's map solved with the stage's walls, placement and search, the label readings replaced "
           "by the template code reader's posterior (commit df61853's reader, `full_fix1*.records`, T = "
           f"{res['temperature']:.1f}, default none/other masses). No LLM reading enters the energy. "
           "**Caveat: the reader's letter shapes and priors were learned from training labels across the whole city "
           "(the owner's answered labels and their blocks held out), not held out by quarter as the trust fit is.** "
           "The labels themselves (where the prints are) still come from the saved reads' boxes and points.", "",
           "## The control", "",
           f"The stage's readings through this script's path give `final` exactly: each quarter's solve lot for lot "
           f"({', '.join(f'{q} {v}' for q, v in res['control']['quarters_equal'].items())}), the join "
           f"({res['control']['join_equal']}), the six measures every lot ({res['control']['every_lot_equal']}) and "
           f"with final's data errors left out ({res['control']['left_out_equal']}).", "",
           "## The six measures (every lot scored)", "",
           "| Measure | final | control | S0 (posterior direct) | S0-mix (function's mixing) |", "|---|---|---|---|---|"]
    for k in KEY:
        out.append(f"| {report_mod.MEASURES[k]} | {fmt(k, f[k])} | " + " | ".join(
            fmt(k, runs_[r]["measures"][k]) for r in RUNS) + " |")
    out.append("| Companion: wrong lots outside final's largest incident | " + f"{f['wrong_lots_outside_1']:,} | "
               + " | ".join(f"{runs_[r]['measures']['wrong_lots_outside_1']:,}" for r in RUNS) + " |")
    out += ["", "Paired block bootstrap (blocks resampled, the same draw for both runs; "
            f"{res['bootstrap']['draws']:,} draws over {res['bootstrap']['blocks']:,} blocks, seed "
            f"{res['bootstrap']['seed']}); 95% intervals:", "",
            "| Measure | run | run − final | interval of the difference | run's own interval |", "|---|---|---|---|---|"]
    names = {"wrong_lots": "wrong lots", "wrong_lots_outside_1": "wrong lots outside final's largest incident",
             "incidents": "incidents", "labels_right": "labels right"}
    for r in ("s0", "s0-mix"):
        for k, nm in names.items():
            v = runs_[r]["vs_final"][k]
            d = runs_[r]["measures"][k] - f[k]
            if k == "labels_right":
                out.append(f"| {nm} | {r} | {d * 100:+.2f} pt | [{v['diff_interval'][0] * 100:+.2f}, "
                           f"{v['diff_interval'][1] * 100:+.2f}] pt | [{v['run_interval'][0]:.2%}, "
                           f"{v['run_interval'][1]:.2%}] |")
            else:
                out.append(f"| {nm} | {r} | {d:+,} | [{v['diff_interval'][0]:+,.0f}, {v['diff_interval'][1]:+,.0f}]"
                           f" | [{v['run_interval'][0]:,.0f}, {v['run_interval'][1]:,.0f}] |")
    rd = res["reader"]
    rt = rd["rates_on_charged_answered"]
    base_w = runs_["s0"]["vs_final"]["wrong_lots"]["base_interval"]
    out += ["", f"Final's own block-bootstrap interval: wrong lots [{base_w[0]:,.0f}, {base_w[1]:,.0f}], labels right "
            f"[{runs_['s0']['vs_final']['labels_right']['base_interval'][0]:.2%}, "
            f"{runs_['s0']['vs_final']['labels_right']['base_interval'][1]:.2%}].", "",
            "The companion measure leaves out the 1,838 lots of final's largest incident (RH-1 coded RH-1(S), the "
            "Mission-strip district behind Mission St), a fixed lot set taken from final for every run. That incident "
            "is a search trap under final's readings (final's energy prefers the table's map there: big-incidents "
            "note, #1); whether a run falls into it moves wrong lots by about 1,800 at once, so the companion shows "
            "the rest.", "",
            "Search check: each run's own energy at final's quarter answer minus at its own (below 0: its search "
            "stopped above final's map): " + "; ".join(
                f"{r} " + ", ".join(f"{q} {v:+.1f}" for q, v in runs_[r]["search_gap"].items()) for r in RUNS) + ".",
            "",
            "## How the posterior enters the energy", "",
            "- Each charged label of the stage's energy (the same 1,881 labels and 2,325 charges, own lot and "
            "far-side copies, in the same order) gets P(code) from the reader's posterior instead of from the trust "
            f"fit. Labels with no record: {runs_['s0']['info']['nw']['no_record']} (every print was scored).",
            "- S0: P(code) = the posterior's mass on the code; the mass of \"none\" and \"other\" goes to no code: it "
            "is spread evenly over the K names (K = 118, the readings' name list, as the function's 1/K). Then P is "
            f"mixed with 1/K at weight {res['floor']:.2e}, the uniform share of the function's surest P (a reading at "
            "its trust clip 0.999 is right with q = 0.99951), so no code is impossible and no posterior is surer "
            "than the function's surest reading. No recalibration (`TOP_READING_FIT` is not applied).",
            "- Far-side copies (a label printed on an ordinary street also speaks for the lot across): the code masses "
            "times the quarter's ratio min(rB/rA, 1) (as `labels.readings` scales a far copy's trust); the mass that "
            "takes off is spread evenly too; then the same floor.",
            "- The placement's options (where the label may stand) and the -log(pi P + (1 - pi)/K) charge are the "
            "function's own lines (`reader_spike.label_costs_from`, checked bit for bit against "
            "`energy.label_costs` with the function's P).",
            "- The tie-break's cheapest paths (1e-4 x path cost) start from each charge's top code at its lot.",
            "- S0-mix (a check on that choice): q = the posterior's top code mass, and the other codes' masses "
            "through the function's own mixing, (rest + 1/K) / (rest.sum() + 1).", "",
            "## The reader on the owner's answers", "",
            f"Answered labels: {rd['answered_labels']} ({rd['with_code']} with a code, {rd['no_code_printed']} "
            "\"no code printed\", one case on no print left out). Top-1 on the ones with a code:", "",
            "| | all | unflagged |", "|---|---|---|",
            f"| template reader (T 32.7) | {rd['reader_top1']['right']}/{rd['reader_top1']['of']} = "
            f"{rd['reader_top1']['share']:.1%} | {rd['reader_top1_unflagged']['share']:.1%} |",
            f"| the stage's readings (trust fit, held out by quarter) | {rd['stage_top1']['right']}/"
            f"{rd['stage_top1']['of']} = {rd['stage_top1']['share']:.1%} | {rd['stage_top1_unflagged']['share']:.1%} |",
            "| J7's combined LLM reading (spec, on 717 cases) | 96.7% | 98.1% |", "",
            f"Reader: box labels {rd['reader_top1_box']['share']:.1%} of {rd['reader_top1_box']['of']}, point labels "
            f"{rd['reader_top1_point']['share']:.1%} of {rd['reader_top1_point']['of']}. On the {rt['labels']} "
            f"answered labels the solver charges: reader misread {rt['reader_misread']:.1%}, miss (top is \"none\" "
            f"or \"other\") {rt['reader_miss']:.1%}; the stage's readings misread {rt['stage_misread']:.1%}, miss "
            f"{rt['stage_miss']:.1%}.", "",
            "How sure each was when it misread (the charged answered labels): the reader's mass on its wrong top, "
            f"mean {rd['misread_confidence']['reader']['mean']:.2f}, {rd['misread_confidence']['reader']['at_least_0.9']} "
            f"of its {rd['misread_confidence']['reader']['misreads']} misreads at 0.9 or more; the chance the energy gives "
            f"the stage's wrong top reading, mean {rd['misread_confidence']['stage']['mean']:.2f}, "
            f"{rd['misread_confidence']['stage']['at_least_0.9']} of its {rd['misread_confidence']['stage']['misreads']} "
            "at 0.9 or more.", "",
            "Reliability (labels with a code answer, by the reader's top mass):", "",
            "| top mass | labels | mean mass | right |", "|---|---|---|---|"]
    out += [f"| {b['from']}–{b['to']:.2f} | {b['labels']} | {b['mean_p']:.2f} | {b['right']:.2f} |"
            for b in rd["reader_reliability"]]
    out += ["", "The reader's commonest confusions (answer → read): " + ", ".join(
        f"{t} → {g} ({n})" for t, g, n in rd["reader_confusions"][:10]) + ".", "",
        "The reader's top is \"none\" or \"other\" by the size of the print (every charged label; box labels by "
        "the height of Terra's word, the training words' median 26 px): " + "; ".join(
            (f"{b['from']}–{b['to']} px {b['none_or_other']}/{b['labels']}" if b["from"] is not None and b["to"] < 1000
             else f"{b['from']}+ px {b['none_or_other']}/{b['labels']}" if b["from"] is not None
             else f"point labels {b['none_or_other']}/{b['labels']}") for b in res["reader_none_other_by_size"]) + ".",
        "",
        "## Where S0's map is worse", "",
        f"{res['changed_sets']['sets']:,} changed sets (connected lots whose code differs between S0 and final): "
        f"{res['changed_sets']['worse']:,} add wrong lots ({res['changed_sets']['added_by_worse']:,} in all), "
        f"{res['changed_sets']['better']:,} remove some ({res['changed_sets']['removed_by_better']:,}).", "",
        "| # | lots | added wrong | table | final | S0 | quarter | blocks | S0's energy, final's codes there | "
        "final's energy, S0's codes there | picture |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for i, s in enumerate(res["worse_sets"], 1):
        et = s["energy_test"]
        out.append(f"| {i} | {s['n']:,} | +{s['added_wrong']:,} | {s['table']} | {s['base_code']} | {s['run_code']} "
                   f"| {s['quarter']} | {', '.join(s['blocks'])} | {et['s0_energy_of_final_codes_minus_own']:+.1f} | "
                   f"{et['final_energy_of_s0_codes_minus_own']:+.1f} | `{s['file']}` |")
    eb = res["better_sets"][0]["energy_test"]
    out += ["", "The energy columns (each run's own energy for its quarter, solve_joint's terms): S0's energy of S0's "
            "answer with the set given final's codes, minus S0's own; and final's energy of final's answer with S0's "
            "codes there, minus final's own. A positive S0 column means S0's readings really prefer S0's worse map "
            "there (the reader's doing); a negative one means S0's search missed a lower-energy answer (the "
            "search's doing). For the largest better set: S0's column "
            f"{eb['s0_energy_of_final_codes_minus_own']:+.1f}, final's {eb['final_energy_of_s0_codes_minus_own']:+.1f}.",
            "", "Over every changed set that moves wrong lots (the same test on each, its lots in its main quarter): "
            + "; ".join(f"S0 {side}: " + ", ".join(f"{c} {v['sets']} sets / {v['wrong_lots']:,} lots"
                                                  for c, v in sorted(d.items()))
                        for side, d in sorted(res["attribution"].items()))
            + ". \"search\": S0's own energy is lower with final's codes there (S0's search missed it); \"readings\": "
            "S0's energy prefers S0's codes there."]
    out += ["", "Each picture shows one changed set, to see why S0 coded it differently from final: grey page ink, "
            "the set's lots tinted orange, the table's district borders green (widest), final's blue, S0's red "
            "(thin, on top), and each label with the stage's top reading and trust | the reader's top and its mass "
            "(red where they differ).", "",
            "Where S0 is better (the five sets removing the most wrong lots): " + "; ".join(
                f"{s['n']} lots, {s['added_wrong']:+,} (table {s['table']}, final {s['base_code']}, S0 {s['run_code']})"
                for s in res["better_sets"]) + ". The first is drawn the same way in "
            f"`{res['better_sets'][0]['file']}`, to see which reading made final's largest incident. Without that "
            f"one set S0's changed sets add {res['changed_sets']['net_without_largest_better']:+,} wrong lots net.", "",
            "Labels in each picture whose stage top reading and reader top differ (print: stage → reader):", ""]
    for s in res["worse_sets"] + res["better_sets"][:1]:
        out.append(f"- `{s['file']}`: " + (", ".join(
            f"{d['print']}: {d['stage']} {d['trust']} → {d['reader']} {d['mass']}" for d in s["labels_differing"])
            or "none") + ".")
    out += [""]
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "solve":
        R.run_jobs([(r, q) for r in RUNS for q in R.QUARTERS])
    elif cmd == "report":
        res = build()
        (OUT / "results.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
        md = markdown(res)
        f = OUT / "results.md"
        findings = ""
        if f.is_file() and "## Findings" in f.read_text():         # the findings are written by hand: kept
            findings = f.read_text()[f.read_text().index("## Findings"):]
        f.write_text(md + ("\n" + findings if findings else ""))
        print(md)
    else:
        raise SystemExit(__doc__)
