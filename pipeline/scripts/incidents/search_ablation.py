"""Which fusion proposal fixes which search incident: each table-free proposal of search_experiments' `fusion`
variant fused ALONE into the stage's answer (one QPBO fusion, then the solver's region, compound and expansion
moves), per quarter: the energy change and the share of each of the quarter's search incidents right.

    uv run python scripts/incidents/search_ablation.py QUARTER

Prints and writes incidents-search-fusion/ablation-<q>.json."""
import json
import sys

import numpy as np

import search_lab as L
from common import CENSUS_DIR
from search_experiments import proposals, stage_answers
from search_summary import SEARCH_INCIDENTS


def main(q):
    P = L.problem(q)
    doc = json.loads((CENSUS_DIR / "incidents.json").read_text())
    recs = [r for r in doc["list"] if r["id"] in SEARCH_INCIDENTS and r["quarter"] == q]
    x = P.final
    e0 = P.energy(x)
    out = []
    for name, y in stage_answers(P) + proposals(P, x):
        f, labelled, nvar, d = L.fuse(P, x, y, improve=True)
        g = L.polish(P.U, P.a, P.b, P.odds, f)
        row = {"proposal": name, "fusion_dE": round(d, 3), "then_solver_moves_dE": round(P.energy(g) - e0, 3),
               "right": {r["id"]: round(float(np.mean(g[r["lots"]] == P.table_codes[r["lots"]])), 2) for r in recs}}
        out.append(row)
        print(json.dumps(row), flush=True)
    (L.run_folder("fusion") / f"ablation-{q}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
