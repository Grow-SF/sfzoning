"""Where two runs differ, and why each run's own energy keeps its answer: per changed set (connected lots whose
codes differ between the runs' joins), under each run's saved costs, its own answer against the other run's and the
table's on that set, split into terms (common.split).

    uv run python scripts/incidents/compare_runs.py RUN_A RUN_B [--walls-a DIR] [--walls-b DIR]
                                                    [--incident ID | --min-lots N]

RUN_A / RUN_B: 'final', an experiment in data/solver/experiments/, solved with its costs (`sfzmaps solver solve`
keeps them).  --walls-a / --walls-b: the walls each was solved with (default the stage's), which its placement was
fitted with.  --incident: only the changed set holding the census incident ID's lots; else every changed set of
--min-lots (default 20) lots or more.  Per set: its lots, quarter, codes (table, A, B), scored lots right under each
run, and per run's costs the energy change from its own answer to the other run's and to the table's (positive:
the run's own answer is cheaper), by term.  Reads only."""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from common import CENSUS_DIR, label_charges, load_costs, run_codes, split
from sfzmaps.solver import paths
from sfzmaps.solver.city import city

TERMS = ("labels", "placement", "far_side", "tie_break", "walls_no_line", "walls_same_code_across_line", "total")


def main(run_a, run_b, walls, incident, min_lots) -> None:
    C = city()
    a, b = C.a, C.b
    join = {r: run_codes(r) for r in (run_a, run_b)}
    differ = join[run_a] != join[run_b]
    m = differ[a] & differ[b]
    _, comp = connected_components(coo_matrix((np.ones(int(m.sum())), (a[m], b[m])), shape=(C.n, C.n)),
                                   directed=False)
    if incident is not None:
        rec = next(t for t in json.loads((CENSUS_DIR / "incidents.json").read_text())["list"] if t["id"] == incident)
        lots = np.array(rec["lots"])
        ids = np.unique(comp[lots[differ[lots]]])
        sets = [np.flatnonzero(differ & np.isin(comp, ids))] if len(ids) else []
    else:
        sizes = Counter(comp[differ].tolist())
        sets = [np.flatnonzero(differ & (comp == c)) for c, n in sizes.most_common() if n >= min_lots]
    print(f"{run_a} vs {run_b}: {int(differ.sum()):,} lots differ; {len(sets)} changed sets shown")
    cache = {}
    for S in sets:
        q = paths.QUARTERS[Counter(C.quarter[S].tolist()).most_common(1)[0][0]]
        right = {r: int((join[r][S] == C.table[S])[C.scored[S]].sum()) for r in join}
        outside = int((C.quarter[S] != paths.QUARTERS.index(q)).sum())
        print(f"== {len(S)} lots, quarter {q} (lots in other quarters {outside}); scored right: {run_a} "
              f"{right[run_a]}, {run_b} {right[run_b]} of {int(C.scored[S].sum())}")
        print("   (table, A, B): " + ", ".join(f"{t}|{x}|{y} {n}" for (t, x, y), n in
                                             Counter(zip(C.table[S], join[run_a][S], join[run_b][S])).most_common(6)))
        for r, other in ((run_a, run_b), (run_b, run_a)):
            if (r, q) not in cache:
                costs = load_costs(r, q)
                cache[(r, q)] = (costs, label_charges(r, q, costs, walls[r]))
            costs, ch = cache[(r, q)]
            x0 = costs.codes
            for name, target in ((other, costs.to_index(join[other][S], x0[S])),
                                 ("the table", np.where(costs.table_codes[S] >= 0, costs.table_codes[S], x0[S]))):
                x1 = x0.copy()
                x1[S] = target
                s = split(costs, ch, x0, x1, S)
                print(f"   [{r}'s costs] its answer -> {name}'s: " + ", ".join(f"{t} {s[t]:+.1f}" for t in TERMS))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run_a")
    ap.add_argument("run_b")
    ap.add_argument("--walls-a", type=Path)
    ap.add_argument("--walls-b", type=Path)
    ap.add_argument("--incident", type=int)
    ap.add_argument("--min-lots", type=int, default=20)
    args = ap.parse_args()
    main(args.run_a, args.run_b, {args.run_a: args.walls_a, args.run_b: args.walls_b}, args.incident, args.min_lots)
