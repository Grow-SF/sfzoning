"""The compound fusion move's acceptance criteria, fixed with the owner before it was measured (2026-11-16), checked
on an experiment run made with it (`sfzmaps solver solve RUN --compound-fusion`) against the stage's run `final`:

  1. the solver's energy goes down in every quarter (each quarter log's "solved" energy against final's).  This one
     compares the energy of two runs of the SAME labels and walls: once the labels or the walls differ (another label
     source, another walls fit) the energies are of different functions and the comparison means nothing.  Pass
     --across-label-sources for such a comparison: criterion 1 is then reported as "not comparable", left out of the
     verdict, and only criterion 2 gates;
  2. wrong-lot incidents and labels right are not worse beyond the measured noise: a paired bootstrap over blocks
     (blocks drawn with replacement, the same draw for both runs, 1,000 draws, seed 0) gives a 95% interval on
     (run - final), with report.six_measures' definitions on every lot; beside it the scoring table's known error
     (its by-eye corrections, the corrected table against the raw one).  "Worse beyond noise": the whole interval
     on the worse side AND the difference larger than the table's known error;
  3. the same on the 1996-2000-zoning scan: not measurable (Ruling R7: the solver exists only for
     2000-digital-zoning).

Reported, not gated: the six measures; the lots right in final and wrong in the run ("broken"), by patch, with the
energy of both answers in their quarter; the five largest changed sets with pictures (render_incident.py); the
census's search incidents, the share of each right; each quarter's time (given with --times, as the solve prints
it: the run's folder holds no times).

    uv run python scripts/incidents/criteria_check.py RUN [--times nw=305,ne=327,sw=329,se=407] [--times-note TEXT]
                                                          [--no-pictures] [--across-label-sources] [--baseline FOLDER]

The labels right and the per-block split count the one fixed scoring set of labels (`measures.scoring_labels`: the
saved-reads chain's 1,881), whatever label source the run solved with.

Writes data/solver/experiments/RUN/criteria.json and criteria.md (and pictures/ there).  Reads only otherwise."""
import argparse
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from common import CENSUS_DIR
from sfzmaps.solver import energy as E
from sfzmaps.solver import paths, runs
from sfzmaps.solver.city import city, load_run, run_file
from sfzmaps.solver.measures import Table, check_block_split
from sfzmaps.solver.report import six_measures

DRAWS, SEED = 1000, 0
SEARCH_INCIDENTS = (1, 7, 34, 4, 23, 9, 44, 18, 14, 21, 12, 13, 15, 42, 22)   # the search report's table


def solved_energy(run, q, folder=None):
    log = (run_file(run, f"-{q}", folder or runs.solve_dir(run)).with_suffix(".log")).read_text()
    return float(re.search(r"solved ([\d,\.]+)", log).group(1).replace(",", ""))


def per_block(codes, blocks, nb):
    """per block: the incidents (each incident split over its lots' blocks by lot), the zoning labels counted, the
    zoning labels right, the wrong lots; and the totals as report.six_measures counts them"""
    C = city()
    a, b = C.a, C.b
    wrong = C.scored & (codes != C.table)
    key = np.char.add(np.char.add(C.table, "|"), codes)
    m = wrong[a] & wrong[b] & (key[a] == key[b])
    _, comp = connected_components(coo_matrix((np.ones(int(m.sum())), (a[m], b[m])), shape=(C.n, C.n)),
                                   directed=False)
    size = np.bincount(comp[wrong], minlength=comp.max() + 1)
    inc = np.bincount(blocks[wrong], 1.0 / size[comp[wrong]], minlength=nb)
    T = Table(np.zeros(C.n, bool))                       # measures.score's labels, label by label
    nd = T.district.max() + 1
    lots_in = np.bincount(T.district[C.scored], minlength=nd)
    right_in = np.bincount(T.district[(codes == C.table) & C.scored], minlength=nd)
    d = np.maximum(T.label_district, 0)
    counted = (T.label_district < 0) | (lots_in[d] > 0)
    right = (T.label_district >= 0) & (lots_in[d] > 0) & (right_in[d] > lots_in[d] / 2)
    zoning = counted & T.label_in_vocab
    label_block = blocks[T.label_lot]                    # the labels are Table's: one fixed scoring set, one source
    return {"incidents": inc, "zoning": np.bincount(label_block[zoning], minlength=nb).astype(float),
            "right": np.bincount(label_block[zoning & right], minlength=nb).astype(float),
            "wrong_lots": np.bincount(blocks[wrong], minlength=nb).astype(float)}


def bootstrap(fin, new, nb):
    rng = np.random.default_rng(SEED)
    out = {"incidents": [], "labels_right": [], "wrong_lots": []}
    for _ in range(DRAWS):
        w = np.bincount(rng.integers(0, nb, nb), minlength=nb).astype(float)
        out["incidents"].append(w @ new["incidents"] - w @ fin["incidents"])
        out["labels_right"].append(w @ new["right"] / (w @ new["zoning"]) - w @ fin["right"] / (w @ fin["zoning"]))
        out["wrong_lots"].append(w @ new["wrong_lots"] - w @ fin["wrong_lots"])
    return {k: [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))] for k, v in out.items()}


def known_error(fin_codes):
    """the scoring table's known error: its by-eye corrections (groups applied, lots changed), and what they change in
    final's two gated measures (final scored on the corrected table against the raw one)"""
    import csv
    C = city()
    with open(paths.TABLE_DIR / "table-corrections.csv", newline="") as f:
        groups = sum(r["verdict"] == "A" for r in csv.DictReader(f))
    lots = int((C.known & (C.table != C.table_raw)).sum())
    corrected = six_measures(fin_codes, np.zeros(C.n, bool))
    saved = C.table
    try:
        C.table = C.table_raw
        raw = six_measures(fin_codes, np.zeros(C.n, bool))
    finally:
        C.table = saved
    return {"corrections_applied": groups, "lots_corrected": lots,
            "final_incidents_corrected_vs_raw": [corrected["incidents"], raw["incidents"]],
            "final_labels_right_corrected_vs_raw": [corrected["labels_right"], raw["labels_right"]],
            "incidents": float(groups),
            "labels_right": abs(corrected["labels_right"] - raw["labels_right"])}


def patches(mask, key):
    """connected patches of the lots of mask with the same key"""
    C = city()
    a, b = C.a, C.b
    j = mask[a] & mask[b] & (key[a] == key[b])
    _, comp = connected_components(coo_matrix((np.ones(int(j.sum())), (a[j], b[j])), shape=(C.n, C.n)),
                                   directed=False)
    sizes = np.bincount(comp[mask], minlength=comp.max() + 1)
    return comp, sizes


def broken(run, fin_codes, new_codes, baseline=None):
    """the lots right in final and wrong in the run, by patch (same table, final and run codes), each with the energy
    of both answers in its quarter's energy: the run's quarter answer, and it with final's quarter answer put back
    on the changed lots joined to the patch"""
    C = city()
    blk = C.lots["blklot"].astype(str).str[:4].values
    m = C.scored & (fin_codes == C.table) & (new_codes != C.table)
    key = np.char.add(np.char.add(np.char.add(C.table, "|"), fin_codes), np.char.add("|", new_codes))
    comp, sizes = patches(m, key)
    rows, cache = [], {}
    for c in np.argsort(-sizes):
        if sizes[c] == 0:
            break
        ix = np.flatnonzero(m & (comp == c))
        q = paths.QUARTERS[Counter(C.quarter[ix].tolist()).most_common(1)[0][0]]
        if q not in cache:
            z = np.load(run_file(run, f"-{q}-costs", runs.solve_dir(run)))
            names = [str(n) for n in z["names"]]
            F = np.load(run_file("final", f"-{q}", baseline), allow_pickle=True)
            fq = np.array([names.index(n) for n in np.asarray(F["names"]).astype(str)])[F["pred"]]
            cache[q] = (z["U"].astype(float), z["odds"].astype(float), z["codes"], fq)
        U, odds, xq, fq = cache[q]
        diff = xq != fq                                   # the quarter's answers' changed lots joined to the patch
        j = diff[C.a] & diff[C.b]
        _, cc = connected_components(coo_matrix((np.ones(int(j.sum())), (C.a[j], C.b[j])), shape=(C.n, C.n)),
                                     directed=False)
        around = diff & np.isin(cc, np.unique(cc[ix[diff[ix]]]))
        back = xq.copy()
        back[around] = fq[around]
        alone = xq.copy()
        alone[ix] = fq[ix]
        e_run = E.energy(U, C.a, C.b, odds, xq, 1.0)
        rows.append({"lots": int(len(ix)), "table": str(C.table[ix[0]]), "final": str(fin_codes[ix[0]]),
                     "run": str(new_codes[ix[0]]), "quarter": q,
                     "blocks": [b_ for b_, _ in Counter(blk[ix]).most_common(3)],
                     "changed_set_lots": int(around.sum()),
                     "energy_final_answer_minus_run": round(E.energy(U, C.a, C.b, odds, back, 1.0) - e_run, 3),
                     "energy_patch_alone_back_minus_run": round(E.energy(U, C.a, C.b, odds, alone, 1.0) - e_run, 3)})
    return {"lots": int(m.sum()), "patches": rows}


def changed_sets(fin_codes, new_codes, top=5):
    """the largest connected sets of lots whose code differs between final and the run (the join)"""
    C = city()
    m = fin_codes != new_codes
    comp, sizes = patches(m, np.full(C.n, "", dtype="<U1"))
    blk = C.lots["blklot"].astype(str).str[:4].values
    out = []
    for c in np.argsort(-sizes)[:top]:
        if not sizes[c]:            # fewer than `top` changed sets (none when the run is final's answer)
            break
        ix = np.flatnonzero(m & (comp == c))
        xy = C.centroid[ix]
        out.append({"lots": int(len(ix)),
                    "transitions": [[f, t, n] for (f, t), n in
                                    Counter(zip(fin_codes[ix].tolist(), new_codes[ix].tolist())).most_common(4)],
                    "right_in_final": int((fin_codes[ix] == C.table[ix]).sum()),
                    "right_in_run": int((new_codes[ix] == C.table[ix]).sum()),
                    "quarter": paths.QUARTERS[Counter(C.quarter[ix].tolist()).most_common(1)[0][0]],
                    "blocks": [b_ for b_, _ in Counter(blk[ix]).most_common(3)],
                    "window": [float(np.nanmin(xy[:, 0]) - 300), float(np.nanmin(xy[:, 1]) - 300),
                               float(np.nanmax(xy[:, 0]) + 300), float(np.nanmax(xy[:, 1]) + 300)]})
    return out


def picture(run, k, s, out):
    """render_incident.py's plain window around a changed set: the table's borders, final's and the run's"""
    name = f"changed-{k + 1}"
    cmd = [sys.executable, str(Path(__file__).with_name("render_incident.py")), "0", "--other", run, "--window",
           *[str(v) for v in s["window"]], "--quarter", s["quarter"], "--out", str(out), "--name", name]
    subprocess.run(cmd, check=True, capture_output=True)
    return f"pictures/{name}_lines_window.png"


def search_incidents(new_codes, fin_codes):
    C = city()
    doc = json.loads((CENSUS_DIR / "incidents.json").read_text())
    rows = []
    for i in SEARCH_INCIDENTS:
        rec = next(t for t in doc["list"] if t["id"] == i)
        lots = np.array(rec["lots"], int)
        rows.append({"id": i, "lots": int(len(lots)), "table": rec["table"], "final_code": rec["solved"],
                     "quarter": rec["quarter"], "final": float(np.mean(fin_codes[lots] == C.table[lots])),
                     "run": float(np.mean(new_codes[lots] == C.table[lots]))})
    return rows


def main(run, times, times_note, pictures, across_label_sources=False, baseline=None):
    folder = runs.solve_dir(run)
    baseline = Path(baseline) if baseline else None
    C = city()
    fin_codes, new_codes = load_run("final", baseline), load_run(run, folder)
    blocks_name = C.lots["blklot"].astype(str).str[:4].values
    _, blocks = np.unique(blocks_name, return_inverse=True)
    nb = blocks.max() + 1
    # 1: the energy per quarter
    energy = {q: {"final": solved_energy("final", q, baseline), "run": solved_energy(run, q)} for q in paths.QUARTERS}
    for v in energy.values():
        v["change"] = round(v["run"] - v["final"], 1)
    # across label sources (or walls) the two energies are of different functions: not comparable, not gated
    c1 = None if across_label_sources else all(v["run"] < v["final"] for v in energy.values())
    # 2: the paired block bootstrap, with the table's known error
    m_fin, m_new = six_measures(fin_codes, np.zeros(C.n, bool)), six_measures(new_codes, np.zeros(C.n, bool))
    fin, new = per_block(fin_codes, blocks, nb), per_block(new_codes, blocks, nb)
    for name, pb, m, codes in (("final", fin, m_fin, fin_codes), (run, new, m_new, new_codes)):     # the split per block adds up to the measures
        check_block_split(pb, blocks, codes, np.zeros(C.n, bool), m, name)
    ci = bootstrap(fin, new, nb)
    err = known_error(fin_codes)
    diff = {"incidents": m_new["incidents"] - m_fin["incidents"],
            "labels_right": m_new["labels_right"] - m_fin["labels_right"],
            "wrong_lots": m_new["wrong_lots"] - m_fin["wrong_lots"]}
    worse = {"incidents": ci["incidents"][0] > 0 and diff["incidents"] > err["incidents"],
             "labels_right": ci["labels_right"][1] < 0 and -diff["labels_right"] > err["labels_right"]}
    c2 = not any(worse.values())
    out = {"run": run, "compared_with": "final",
           "measure": "report.six_measures on every lot (no data errors left out)",
           "criteria": {
               "1_energy_lower_every_quarter": {
                   "verdict": "not comparable" if c1 is None else "pass" if c1 else "fail", "energy": energy,
                   **({"why": "run and final differ in labels or walls: their energies are of different functions"}
                      if c1 is None else {})},
               "2_not_worse_beyond_noise": {
                   "verdict": "pass" if c2 else "fail", "difference_run_minus_final": diff,
                   "bootstrap_95": ci, "draws": DRAWS, "seed": SEED, "blocks": int(nb),
                   "table_known_error": err, "worse_beyond_noise": worse},
               "3_same_on_1996_2000_scan": {
                   "verdict": "not measurable",
                   "why": "Ruling R7: the solver exists only for 2000-digital-zoning (paths.SERIES); measuring it on "
                          "1996-2000-zoning needs the whole solver generalised to a second series"}},
           "measures_every_lot": {"final": m_fin, run: m_new},
           "broken": broken(run, fin_codes, new_codes, baseline),
           "changed_sets": changed_sets(fin_codes, new_codes),
           "search_incidents": search_incidents(new_codes, fin_codes),
           "times_s": times, "times_note": times_note}
    if pictures:
        for k, s in enumerate(out["changed_sets"]):
            s["picture"] = picture(run, k, s, folder / "pictures")
    (folder / "criteria.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    (folder / "criteria.md").write_text(markdown(out))
    print(markdown(out))


def markdown(o):
    c = o["criteria"]
    e1, b2 = c["1_energy_lower_every_quarter"], c["2_not_worse_beyond_noise"]
    err, ci, d = b2["table_known_error"], b2["bootstrap_95"], b2["difference_run_minus_final"]
    run = o["run"]
    L = [f"# Acceptance criteria: the compound fusion move ({run} against final)", "",
         "Fixed with the owner before the build (2026-11-16). Measures: `report.six_measures` on every lot.", "",
         "| Criterion | Verdict |", "|---|---|",
         f"| 1. The solver's energy goes down in every quarter | **{e1['verdict']}** |",
         f"| 2. Incidents and labels right not worse beyond the measured noise | **{b2['verdict']}** |",
         f"| 3. The same on the 1996-2000-zoning scan | **{c['3_same_on_1996_2000_scan']['verdict']}** (Ruling R7) |",
         "", "## 1. Energy per quarter (each quarter log's solved energy)", "",
         "| Quarter | final | run | change |", "|---|---|---|---|"]
    L += [f"| {q} | {v['final']:,.1f} | {v['run']:,.1f} | {v['change']:+.1f} |" for q, v in e1["energy"].items()]
    L += ["", "## 2. Paired block bootstrap (run - final)", "",
          f"{b2['draws']:,} draws over {b2['blocks']:,} blocks (assessor blocks, blklot's first four characters), "
          f"seed {b2['seed']}; incidents split over their lots' blocks; labels by the block of the lot under the "
          "label.", "",
          "| Measure | final | run | difference | 95% interval | table's known error | worse beyond noise |",
          "|---|---|---|---|---|---|---|",
          f"| Incidents | {o['measures_every_lot']['final']['incidents']:,} | "
          f"{o['measures_every_lot'][run]['incidents']:,} | {d['incidents']:+,} | "
          f"[{ci['incidents'][0]:+.1f}, {ci['incidents'][1]:+.1f}] | {err['incidents']:.0f} corrections | "
          f"{'yes' if b2['worse_beyond_noise']['incidents'] else 'no'} |",
          f"| Labels right | {o['measures_every_lot']['final']['labels_right']:.2%} | "
          f"{o['measures_every_lot'][run]['labels_right']:.2%} | {d['labels_right'] * 100:+.2f} pts | "
          f"[{ci['labels_right'][0] * 100:+.2f}, {ci['labels_right'][1] * 100:+.2f}] pts | "
          f"{err['labels_right'] * 100:.2f} pts | {'yes' if b2['worse_beyond_noise']['labels_right'] else 'no'} |",
          f"| Wrong lots (reported) | {o['measures_every_lot']['final']['wrong_lots']:,} | "
          f"{o['measures_every_lot'][run]['wrong_lots']:,} | {d['wrong_lots']:+,} | "
          f"[{ci['wrong_lots'][0]:+,.0f}, {ci['wrong_lots'][1]:+,.0f}] | – | – |", "",
          f"The table's known error: {err['corrections_applied']} by-eye corrections applied "
          f"({err['lots_corrected']:,} lots). They change final's incidents "
          f"{err['final_incidents_corrected_vs_raw'][1]:,} (raw) to {err['final_incidents_corrected_vs_raw'][0]:,} "
          f"(corrected) and its labels right {err['final_labels_right_corrected_vs_raw'][1]:.2%} to "
          f"{err['final_labels_right_corrected_vs_raw'][0]:.2%}. For incidents the error counted is the "
          "corrections' number; for labels right, what they change in final's share. \"Worse beyond noise\": the "
          "whole interval on the worse side and the difference larger than that error.", "",
          "## The six measures (every lot; reported)", "", "| Measure | final | run |", "|---|---|---|"]
    for k, lab, fmt in (("lines_drawn", "Lines drawn", "{:.2%}"),
                        ("drawn_are_table_lines", "Drawn that are table lines", "{:.2%}"),
                        ("labels_right", "Labels right", "{:.2%}"), ("holes", "Holes", "{:,}"),
                        ("incidents", "Incidents", "{:,}"), ("wrong_lots", "Wrong lots", "{:,}")):
        L.append(f"| {lab} | {fmt.format(o['measures_every_lot']['final'][k])} | "
                 f"{fmt.format(o['measures_every_lot'][run][k])} |")
    br = o["broken"]
    L += ["", f"## Broken lots: right in final, wrong in the run ({br['lots']:,})", "",
          "Each patch with the energy of both answers in its quarter: the run's answer, and it with final's put back "
          "on the changed lots joined to the patch (the whole move it was part of), or on the patch alone (positive: "
          "the energy prefers the run's, wrong, answer there).", "",
          "| Lots | Table | final | run | Quarter | Blocks | Changed set | E(final's on the set) - E(run's) "
          "| E(final's on the patch) - E(run's) |",
          "|---|---|---|---|---|---|---|---|---|"]
    L += [f"| {p['lots']} | {p['table']} | {p['final']} | {p['run']} | {p['quarter']} | {', '.join(p['blocks'])} | "
          f"{p['changed_set_lots']:,} | {p['energy_final_answer_minus_run']:+.2f} | "
          f"{p['energy_patch_alone_back_minus_run']:+.2f} |" for p in br["patches"]]
    L += ["", "## The five largest changed sets (final's code against the run's)", "",
          "Each picture (render_incident.py, a window around the set, 2x the grid) shows the page's ink (grey), the "
          "table's district borders (green, widest), final's (blue, middle) and the run's (red, thin, on top), and "
          "the printed labels with their top reading and trust. It is there to show where the run's lines move "
          "against the table's; no action is needed unless a set below is mostly wrong in the run.", "",
          "| # | Lots | final → run | Right in final | Right in run | Quarter | Blocks | Picture |",
          "|---|---|---|---|---|---|---|---|"]
    for k, s in enumerate(o["changed_sets"]):
        tr = "; ".join(f"{f} → {t} {n:,}" for f, t, n in s["transitions"])
        L.append(f"| {k + 1} | {s['lots']:,} | {tr} | {s['right_in_final']:,} | {s['right_in_run']:,} | "
                 f"{s['quarter']} | {', '.join(s['blocks'])} | {s.get('picture', '–')} |")
    L += ["", "## The census's search incidents: share of each right (join)", "",
          "| # | Lots | Table → final | Q | final | run |", "|---|---|---|---|---|---|"]
    L += [f"| {r['id']} | {r['lots']:,} | {r['table']} → {r['final_code']} | {r['quarter']} | {r['final']:.0%} | "
          f"{r['run']:.0%} |" for r in o["search_incidents"]]
    if o["times_s"]:
        L += ["", "## Times", "", "Each quarter's solve, seconds (as `sfzmaps solver solve` printed them): "
              + ", ".join(f"{q} {s}" for q, s in o["times_s"].items()) + ".", "", o["times_note"] or ""]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run")
    ap.add_argument("--times", default="", help="each quarter's solve time in seconds, e.g. nw=305,ne=327")
    ap.add_argument("--times-note", default="")
    ap.add_argument("--no-pictures", action="store_true")
    ap.add_argument("--across-label-sources", action="store_true",
                    help="the run solved with other labels (or walls) than final: leave out criterion 1 (energy "
                         "lower in every quarter), which compares the energies of two different functions")
    ap.add_argument("--baseline", help="a folder holding a copy of an earlier stage run `final` (its runs/final*.npz and "
                                        "logs) to compare against, in place of data/solver/solve; the pictures still "
                                        "draw the stage's own final, so give --no-pictures with it")
    args = ap.parse_args()
    t = dict(kv.split("=") for kv in args.times.split(",") if kv)
    main(args.run, {k: float(v) for k, v in t.items()}, args.times_note, not args.no_pictures, args.across_label_sources,
         args.baseline)
