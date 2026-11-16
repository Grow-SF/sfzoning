"""The ceiling check (`solve --true-borders`) on the Mission strip: with the table's own borders as near-certain walls
(solve.TRUE_BORDER_ODDS) and the stage's labels and placement, does the solver's search still miss the RH-1 district
behind Mission St, and is the right answer there lower in that energy?  J7's write-up blamed the labels for its
version of this incident ("with the true borders as walls, 1,844 of them still go wrong (they take NC-1)").

    uv run python scripts/incidents/search_true_borders.py QUARTER [ID ...]

For each incident given (default the quarter's search incidents): the share of its lots the true-borders solve gets
right and their codes, and the energy (under the true-borders energy) of that solve's answer with the incident's
changed set D put at the table's codes, against the solve's.  Also the same from the stage's answer.  Prints and
writes data/solver/experiments/incidents-search-diagnosis/true-borders-<q>.json."""
import json
import sys
from collections import Counter

import numpy as np

from types import SimpleNamespace

import search_lab as L
from common import CENSUS_DIR, changed_set, ring, search_on
from sfzmaps.solver import energy as E

OUT = L.runs.EXPERIMENTS_DIR / "incidents-search-diagnosis"


def terms(P, x, y):
    """the energy change from x to y by term: label costs (with the tie-break), borders whose lots differ where the
    odds say no line, borders whose lots share a code where the odds say a line (known pairs at the true borders'
    +-6, the rest at the stage's odds)"""
    a, b, t, ra, rb, rr = E.wall_terms(P.a, P.b, P.odds, L.SAME)
    n = np.arange(len(x))
    return {"labels": round(float((P.U[n, y] - P.U[n, x]).sum()), 3),
            "no_line": round(float((t * ((y[a] != y[b]).astype(float) - (x[a] != x[b]))).sum()), 3),
            "same_code_across_line": round(float((rr * ((y[ra] == y[rb]).astype(float) - (x[ra] == x[rb]))).sum()),
                                           3)}
DEFAULT = {"sw": [1, 14], "se": [4, 7, 9], "nw": [18], "ne": [12, 13, 15, 21]}


def main(q, ids):
    P = L.problem(q, true_borders=True)
    p = L.solver_sequence(P.U, P.a, P.b, P.odds)
    names = np.array(P.names)
    doc = json.loads((CENSUS_DIR / "incidents.json").read_text())
    out = {"quarter": q, "energy_true_borders_solve": round(P.energy(p), 3), "incidents": []}
    for i in ids or DEFAULT[q]:
        rec = next(t for t in doc["list"] if t["id"] == i)
        I = np.array(rec["lots"])
        row = {"id": i, "lots": len(I), "table": rec["table"], "solved_final": rec["solved"],
               "right_share": round(float(np.mean(p[I] == P.table_codes[I])), 3),
               "codes": Counter(names[p[I]].tolist()).most_common(3)}
        D = changed_set(SimpleNamespace(table_codes=P.table_codes, codes=p), I)
        if len(D):
            y = p.copy()
            y[D] = P.table_codes[D]
            row.update(changed_set=int(len(D)), table_answer_on_D_dE=round(P.energy(y) - P.energy(p), 3),
                       table_answer_on_D_terms=terms(P, p, y))
            # the cheapest answer right on the incident under this energy (census method: the incident at the
            # table's codes, D and two rings around it searched from the table's answer)
            ok = P.table_codes[I] >= 0
            best = None
            for hops in range(3):
                z = search_on(SimpleNamespace(U=P.U, odds=P.odds), y, np.setdiff1d(ring(D, hops), I[ok]))
                if best is None or P.energy(z) < P.energy(best):
                    best = z
            row.update(cheapest_right_dE=round(P.energy(best) - P.energy(p), 3),
                       cheapest_right_terms=terms(P, p, best))
        out["incidents"].append(row)
        print(row, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"true-borders-{q}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], [int(v) for v in sys.argv[2:]])
