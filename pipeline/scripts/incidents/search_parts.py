"""Why each single-code part of a lower-energy answer is uphill on its own: the energy change of "the lots the
lower answer gives code c switch to c" (one expansion move), split into the label term and the wall terms on the
borders it touches, the wall terms grouped by the codes on the two sides after the move.

    uv run python scripts/incidents/search_parts.py ID [ID ...]

Reads the stage's solve of the incident's quarter (search_lab.problem) and the census's cheapest answer right on the
incident.  Prints only."""
import json
import sys
from collections import defaultdict

import numpy as np

import search_lab as L
from common import CENSUS_DIR
from sfzmaps.solver import energy as E


def parts(P, x, y):
    names = np.array(P.names)
    N = len(x)
    D = np.flatnonzero(x != y)
    a, b, t, ra, rb, rr = E.wall_terms(P.a, P.b, P.odds, L.SAME)
    for al in np.unique(y[D]):
        S = D[y[D] == al]
        q = x.copy()
        q[S] = al
        inS = np.zeros(N, bool)
        inS[S] = True
        dU = float((P.U[S, al] - P.U[S, x[S]]).sum())
        groups = defaultdict(float)
        for pa, pb, w, kind in ((a, b, t, "no line, codes differ"), (ra, rb, rr, "line, one code")):
            touch = inS[pa] | inS[pb]
            for xs, sgn in ((q, 1.0), (x, -1.0)):
                cost = w * ((xs[pa] != xs[pb]) if kind.startswith("no") else (xs[pa] == xs[pb]))
                m = touch & (cost > 0)
                for i in np.flatnonzero(m):
                    other = pb[i] if inS[pa[i]] else pa[i]
                    groups[(kind, names[x[other]] + ("" if not inS[other] else " (moving)"))] += sgn * cost[i]
        total = P.energy(q) - P.energy(x)
        print(f"   part -> {names[al]} ({len(S)} lots, from {sorted(set(names[x[S]].tolist()))}): dE {total:+.2f} = "
              f"labels {dU:+.2f} + walls {total - dU:+.2f}")
        for k, v in sorted(groups.items(), key=lambda kv: -abs(kv[1]))[:8]:
            if abs(v) > 0.05:
                print(f"      {k[0]}, other side {k[1]}: {v:+.2f}")


def main(ids):
    doc = json.loads((CENSUS_DIR / "incidents.json").read_text())
    cache = {}
    for i in ids:
        rec = next(t for t in doc["list"] if t["id"] == i)
        q = rec["quarter"]
        P = cache.setdefault(q, L.problem(q))
        z = np.load(CENSUS_DIR / "answers" / f"cheapest-right-{i}.npz")
        y = P.final.copy()
        y[z["lots"]] = P.to_index(z["codes"].astype(str))
        print(f"#{i} ({q}): lower answer dE {P.energy(y) - P.energy(P.final):+.2f}")
        parts(P, P.final, y)


if __name__ == "__main__":
    main([int(v) for v in sys.argv[1:]])
