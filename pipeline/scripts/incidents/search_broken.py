"""Where a lower-energy search answer gets MORE lots wrong: for an experiment, its largest patches of lots right in
final and wrong in the experiment, each with the connected set of lots the experiment changed around it (in the
quarter's solve), and the energy of putting final's codes back on that set: positive means the run's own energy
prefers the experiment's wrong answer there (the model, not the search), split into labels, no-line borders and
same-code-across-a-line borders, with the walls on the borders whose drawn state changes.

    uv run python scripts/incidents/search_broken.py VARIANT [--top 5]

Prints and writes incidents-search-<variant>/broken.json."""
import argparse
import json
from collections import Counter

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.special import expit

import search_lab as L
from sfzmaps.solver import energy as E
from sfzmaps.solver import paths
from sfzmaps.solver.city import city, load_run, run_file


def terms(P, x, y):
    a, b, t, ra, rb, rr = E.wall_terms(P.a, P.b, P.odds, L.SAME)
    n = np.arange(len(x))
    return {"labels": round(float((P.U[n, y] - P.U[n, x]).sum()), 3),
            "no_line": round(float((t * ((y[a] != y[b]).astype(float) - (x[a] != x[b]))).sum()), 3),
            "same_code_across_line": round(float((rr * ((y[ra] == y[rb]).astype(float) - (x[ra] == x[rb]))).sum()),
                                           3)}


def main(variant, top):
    C = city()
    final = load_run("final")
    codes = load_run(L.SEARCH_PREFIX + variant, L.run_folder(variant))
    broken = C.scored & (final == C.table) & (codes != C.table)
    key = np.char.add(np.char.add(C.table, "|"), codes)
    j = broken[C.a] & broken[C.b] & (key[C.a] == key[C.b])
    _, comp = connected_components(coo_matrix((np.ones(int(j.sum())), (C.a[j], C.b[j])), shape=(C.n, C.n)),
                                   directed=False)
    sizes = np.bincount(comp[broken], minlength=comp.max() + 1)
    blk = C.lots["blklot"].astype(str).str[:4].values
    probs = {}
    out = []
    for c in np.argsort(-sizes)[:top]:
        if sizes[c] == 0:
            break
        patch = np.flatnonzero(broken & (comp == c))
        qi = Counter(C.quarter[patch].tolist()).most_common(1)[0][0]
        q = paths.QUARTERS[qi]
        P = probs.setdefault(q, L.problem(q))
        R = np.load(run_file(L.SEARCH_PREFIX + variant, f"-{q}", L.run_folder(variant)), allow_pickle=True)
        names = list(P.names)
        y = np.array([names.index(n) for n in np.asarray(R["names"]).astype(str)])[R["pred"]]
        x = P.final
        diff = x != y
        m = diff[P.a] & diff[P.b]
        _, cc = connected_components(coo_matrix((np.ones(int(m.sum())), (P.a[m], P.b[m])), shape=(C.n, C.n)),
                                     directed=False)
        S = np.flatnonzero(np.isin(cc, np.unique(cc[patch[diff[patch]]])) & diff)
        back = y.copy()
        back[S] = x[S]
        dE = P.energy(back) - P.energy(y)
        flips = np.flatnonzero(((y[P.a] != y[P.b]) != (back[P.a] != back[P.b])) & (np.isin(P.a, S) | np.isin(P.b, S)))
        p_ = expit(P.odds[flips])
        row = {"lots": int(len(patch)), "table": str(C.table[patch[0]]), "final": str(final[patch[0]]),
               "variant": str(codes[patch[0]]), "quarter": q,
               "blocks": [b_ for b_, _ in Counter(blk[patch]).most_common(3)],
               "changed_set": int(len(S)),
               "changed_codes": [[f"{names[a_]} -> {names[b_]}", int(n_)] for (a_, b_), n_ in
                                 Counter(zip(x[S].tolist(), y[S].tolist())).most_common(4)],
               "dE_put_final_back": round(dE, 3), "terms_put_final_back": terms(P, y, back),
               "borders_changing": int(len(flips)),
               "their_wall_p": {"median": round(float(np.median(p_)), 3) if len(p_) else None,
                                "share_above_half": round(float((p_ > 0.5).mean()), 3) if len(p_) else None}}
        out.append(row)
        print(json.dumps(row), flush=True)
    (L.run_folder(variant) / "broken.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("variant")
    ap.add_argument("--top", type=int, default=5)
    args = ap.parse_args()
    main(args.variant, args.top)
