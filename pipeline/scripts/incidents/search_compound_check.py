"""What the solver's compound move does with an incident's region: the region (of the stage's answer) holding most of
the lower answer's main change, the code its labels favour (the compound move's `want`), and for each touching
region holding that code the code the compound move would send it to (its best code none of its neighbours holds)
against the code the lower answer gives it; and the energy of the compound move as the solver builds it.

    uv run python scripts/incidents/search_compound_check.py ID [ID ...]

Prints only."""
import json
import sys
from collections import Counter

import numpy as np
from scipy.sparse import coo_matrix

import search_lab as L
from common import CENSUS_DIR
from sfzmaps.solver import search


def main(ids):
    doc = json.loads((CENSUS_DIR / "incidents.json").read_text())
    cache = {}
    for i in ids:
        rec = next(t for t in doc["list"] if t["id"] == i)
        q = rec["quarter"]
        P = cache.setdefault(q, L.problem(q))
        names = np.array(P.names)
        x = P.final
        z = np.load(CENSUS_DIR / "answers" / f"cheapest-right-{i}.npz")
        y = x.copy()
        y[z["lots"]] = P.to_index(z["codes"].astype(str))
        reg, R = search._regions(P.a, P.b, P.odds, x)
        rc = np.zeros(R, int)
        rc[reg] = x
        SU = np.zeros((R, P.U.shape[1]))
        np.add.at(SU, reg, P.U)
        e = reg[P.a] != reg[P.b]
        nbr = coo_matrix((np.ones(int(e.sum())), (reg[P.a[e]], reg[P.b[e]])), shape=(R, R)).tocsr()
        nbr = ((nbr + nbr.T) > 0).tocsr()
        D = np.flatnonzero(x != y)
        r0 = Counter(reg[D].tolist()).most_common(1)[0][0]
        want = SU[r0].argmin()
        lots0 = np.flatnonzero(reg == r0)
        print(f"#{i} ({q}): region of {len(lots0)} lots, code {names[rc[r0]]}; its labels favour {names[want]} "
              f"(by {SU[r0, rc[r0]] - SU[r0, want]:.2f}); the lower answer gives it "
              f"{Counter(names[y[lots0]].tolist()).most_common(2)}")
        qq = x.copy()
        qq[lots0] = want
        for n_ in nbr.indices[nbr.indptr[r0]:nbr.indptr[r0 + 1]]:
            if rc[n_] != want:
                continue
            held = set(rc[nbr.indices[nbr.indptr[n_]:nbr.indptr[n_ + 1]]].tolist()) | {want}
            alt = next((c for c in np.argsort(SU[n_]) if c not in held), None)
            ln = np.flatnonzero(reg == n_)
            ranked = [names[c] for c in np.argsort(SU[n_])[:4]]
            print(f"   touching region of {len(ln)} lots holding {names[want]}: compound sends it to "
                  f"{names[alt] if alt is not None else None}; the lower answer gives it "
                  f"{Counter(names[y[ln]].tolist()).most_common(2)}; its labels rank {ranked}; codes its neighbours "
                  f"hold (barred): {sorted(names[list(held)].tolist())[:12]}")
            if alt is not None:
                qq[ln] = alt
        print(f"   the compound move's energy change: {P.energy(qq) - P.energy(x):+.2f}")


if __name__ == "__main__":
    main([int(v) for v in sys.argv[1:]])
