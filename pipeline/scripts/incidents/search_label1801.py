"""Does label 1801 explain why the solver's moves cannot leave the wrong answer on the Mission strip (#1 sw, #7 se)?
(The placement agent's lead: label_fixes.csv deletes it, "nothing printed here"; dropping it alone flips the
confined search there.)  Measure only: the stage's inputs are not changed; 1801 is left out of a copy of the energy
(search_lab.problem(drop_labels=[1801])).

    uv run python scripts/incidents/search_label1801.py QUARTER

Per quarter (sw for #1/#34, se for #7):
  * 1801's charges: the lots it is charged on (its placement options), the chance on each, and its cost of NC-2
    against RH-1 there;
  * with 1801 and without it: the energy of each single-code part of the census's lower answer from the stage's answer
    (the tip -> NC-2 part, the district -> RH-1 part), and of the whole lower answer;
  * without 1801: the solver's own moves (region, compound, expansion) from the stage's answer, and the solver's whole
    sequence from scratch: their share of the incidents right and the energy;
  * with 1801: which search experiments get the incidents right (incidents-search-<variant> runs, this quarter's
    solve).
Writes incidents-search-diagnosis/label1801-<q>.json."""
import json
import sys

import numpy as np

import search_lab as L
from common import CENSUS_DIR
from sfzmaps.solver import energy as E
from sfzmaps.solver.city import city, run_file
from sfzmaps.solver.codes import table_vocabulary
from sfzmaps.solver.labels import Readings, readings
from sfzmaps.solver.placement import load_options

LABEL = 1801
INCIDENTS = {"sw": (1, 34), "se": (7,)}
VARIANTS = ("qpbo", "regions", "fusion", "compound", "compound-qpbo", "starts", "all", "starts-compound")


def right(P, x, ids, recs):
    return {i: round(float(np.mean(x[recs[i]["lots"]] == P.table_codes[recs[i]["lots"]])), 3) for i in ids}


def main(q):
    C = city()
    doc = json.loads((CENSUS_DIR / "incidents.json").read_text())
    recs = {r["id"]: r for r in doc["list"]}
    ids = INCIDENTS[q]
    P = L.problem(q)
    P0 = L.problem(q, drop_labels=[LABEL])
    names = list(P.names)
    blk = C.lots["blklot"].astype(str).values
    # 1801's own charges (label_costs on its readings alone, no tie-break)
    rd = readings(q, table_vocabulary(C.lots))
    k = rd.label == LABEL
    rd1 = Readings(rd.lot[k], rd.code[k], rd.trust[k], rd.label[k], rd.far[k], rd.names)
    U1, _, _ = E.label_costs(C.n, len(names), rd1, load_options(q, L.runs.SOLVE_DIR))
    lots = np.flatnonzero(np.abs(U1).sum(1) > 0)
    nc2, rh1 = names.index("NC-2"), names.index("RH-1")
    out = {"quarter": q, "label": LABEL,
           "readings": [[names[c], round(float(t), 3), blk[l], bool(f)] for c, t, l, f in
                        zip(rd1.code, rd1.trust, rd1.lot, rd1.far)],
           "charged_lots": [{"lot": blk[l], "table": str(C.table[l]), "final": names[P.final[l]],
                             "cost_NC2_minus_RH1": round(float(U1[l, nc2] - U1[l, rh1]), 3)} for l in lots]}
    # the parts of the lower answer, with and without 1801
    i0 = ids[0]
    z = np.load(CENSUS_DIR / "answers" / f"cheapest-right-{i0}.npz")
    y = P.final.copy()
    y[z["lots"]] = P.to_index(z["codes"].astype(str))
    D = np.flatnonzero(y != P.final)
    parts = {}
    for tag, Q in (("with 1801", P), ("without 1801", P0)):
        e = Q.energy(P.final)
        row = {"whole lower answer": round(Q.energy(y) - e, 3)}
        for al in np.unique(y[D]):
            S = D[y[D] == al]
            qa = P.final.copy()
            qa[S] = al
            row[f"{len(S)} lots -> {names[al]}"] = round(Q.energy(qa) - e, 3)
        parts[tag] = row
    out["parts_from_stage_answer"] = parts
    # without 1801: the solver's moves from the stage's answer, and its whole sequence
    p_pol = L.polish(P0.U, P0.a, P0.b, P0.odds, P.final)
    p_seq = L.solver_sequence(P0.U, P0.a, P0.b, P0.odds)
    out["without_1801"] = {
        "solver_moves_from_stage_answer": {"dE": round(P0.energy(p_pol) - P0.energy(P.final), 3),
                                           "right": right(P, p_pol, ids, recs)},
        "solver_sequence_from_scratch": {"energy": round(P0.energy(p_seq), 3),
                                         "energy_stage_answer": round(P0.energy(P.final), 3),
                                         "right": right(P, p_seq, ids, recs),
                                         "lots_changed_vs_stage": int((p_seq != P.final).sum())}}
    # with 1801: the experiments
    exp = {}
    for v in VARIANTS:
        f = run_file(L.SEARCH_PREFIX + v, f"-{q}", L.run_folder(v))
        if f.is_file():
            R = np.load(f, allow_pickle=True)
            x = np.array([names.index(n) for n in np.asarray(R["names"]).astype(str)])[R["pred"]]
            exp[v] = right(P, x, ids, recs)
    out["experiments_with_1801"] = exp
    print(json.dumps(out, indent=1))
    (L.runs.EXPERIMENTS_DIR / "incidents-search-diagnosis" / f"label1801-{q}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
