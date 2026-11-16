"""Why the search misses the lower-energy answers the census found: which single move of the solver's repertoire
would reach each, and why none lowers the energy.

    uv run python scripts/incidents/search_diagnose.py QUARTER

For the quarter's solve (search_lab.problem, rebuilt from its held-out inputs), first the solver's own search is
replayed move for move (search_lab.solver_sequence's calls) with a snapshot after each call; the replay must end on
the stage's answer exactly.  Then for every census incident of the quarter whose cheapest answer right on it
(answers/cheapest-right-<id>.npz) lowers the whole city's energy, with x the stage's answer and y that answer:

  * the change: lots, codes from -> to;
  * expansion: for each code al that y gives, the move "the lots y gives al switch to al" (part of y): its true
    energy change and the change the expansion move's cut sees (search_lab.expansion_parts: the same-code cost
    across a line bounded from above), split into the bound's overcharge on line borders whose lots share a code
    and on those whose lots hold two other codes; the cut's own best move to al from x (true and bounded); and the
    exact expansion move to al by roof duality (QPBO, unlabelled lots keep x);
  * region and compound moves: the regions of x (lots of one code joined across borders with odds <= 0) that the
    change touches, how much of each it takes, the energy of recoding each touched region whole, and the no-line
    borders a region move would have to cut to take only the changed part;
  * fusion: QPBO fusion of x with y (all of y offered at once): does it take y?
  * the history: on the incident's lots, the codes after each call of the replayed search.

Writes data/solver/experiments/incidents-search-diagnosis/diagnose-<q>.json.  Reads only the solver's outputs."""
import argparse
import json
from collections import Counter
from types import SimpleNamespace

import numpy as np

import search_lab as L
from common import CENSUS_DIR, changed_set, search_on
from sfzmaps.solver import energy as E
from sfzmaps.solver import search
from sfzmaps.solver.city import city

OUT = L.runs.EXPERIMENTS_DIR / "incidents-search-diagnosis"


def r(x, d=3):
    return round(float(x), d)


def counts(names, x, lots, top=6):
    return [[n, int(c)] for n, c in Counter(names[x[lots]].tolist()).most_common(top)]


def replay(P):
    """solve_joint's calls one by one, a snapshot (label, energy, answer) after each"""
    U, a, b, odds = P.U, P.a, P.b, P.odds
    snaps = []
    p = U.argmin(1)
    snaps.append(("start: each lot's cheapest code", P.energy(p), p.copy()))
    p = search.expansion_moves(U, a, b, odds, 0.0, p)
    snaps.append(("expansion without the same-code cost", P.energy(p), p.copy()))
    p = search.expansion_moves(U, a, b, odds, L.SAME, p)
    snaps.append(("expansion with it", P.energy(p), p.copy()))
    for rd in range(4):
        e0 = P.energy(p)
        p = search.region_moves(U, a, b, odds, L.SAME, p)
        snaps.append((f"round {rd + 1}: region moves", P.energy(p), p.copy()))
        p = search.compound_moves(U, a, b, odds, L.SAME, p)
        snaps.append((f"round {rd + 1}: compound moves", P.energy(p), p.copy()))
        p = search.expansion_moves(U, a, b, odds, L.SAME, p)
        snaps.append((f"round {rd + 1}: expansion", P.energy(p), p.copy()))
        if P.energy(p) >= e0 - 1e-6:
            break
    return p, snaps


def overcharge(P, x, al, s):
    """the bound's overcharge on the switch set s (expansion to al from x), on line borders (odds > 0) by kind:
    'same' = both lots hold one code (not al) and only the side less keen to switch switches; 'two' = they hold two
    other codes and the less keen side switches alone"""
    _, _, _, ra, rb, rr = E.wall_terms(P.a, P.b, P.odds, L.SAME)
    xa, xb = x[ra], x[rb]
    a_keener = (P.U[ra, al] - P.U[ra, xa]) <= (P.U[rb, al] - P.U[rb, xb])
    sa, sb = s[ra], s[rb]
    keen_sw = np.where(a_keener, sa, sb)
    less_sw = np.where(a_keener, sb, sa)
    same = (xa != al) & (xa == xb) & less_sw & ~keen_sw       # true 0, bound 2 rr
    two = (xa != al) & (xb != al) & (xa != xb) & less_sw & ~keen_sw   # true 0, bound rr
    return {"same_code_pairs": int(same.sum()), "same_code_over": r(2 * rr[same].sum()),
            "two_code_pairs": int(two.sum()), "two_code_over": r(rr[two].sum())}


def cut_best(P, x, al):
    """the solver's own expansion cut to al from x: (lots switched, true dE, bounded dE)"""
    import maxflow
    parts = L.expansion_parts(P.U, P.a, P.b, P.odds, L.SAME, x, al)
    c0, c1, ea, eb, w, const = parts
    m = np.minimum(c0, c1)
    G = maxflow.Graph[float](len(x), len(ea))
    nodes = G.add_nodes(len(x))
    G.add_grid_tedges(nodes, c1 - m, c0 - m)
    G.add_edges(ea.astype(np.int64), eb.astype(np.int64), np.maximum(w, 0.0), np.zeros(len(w)))
    G.maxflow()
    s = G.get_grid_segments(nodes) & (x != al)
    q = x.copy()
    q[s] = al
    e0 = L.bounded_energy(parts, np.zeros(len(x), bool))
    return int(s.sum()), r(P.energy(q) - P.energy(x)), r(L.bounded_energy(parts, s) - e0)


def regions_of(P, x):
    return search._regions(P.a, P.b, P.odds, x)[0]


def main(q: str) -> None:
    C = city()
    P = L.problem(q)
    names = np.array(P.names)
    doc = json.loads((CENSUS_DIR / "incidents.json").read_text())
    p, snaps = replay(P)
    out = {"quarter": q, "replay_equals_stage": bool(np.array_equal(p, P.final)),
           "energy_final": r(P.energy(P.final), 4),
           "replay": [[lab, r(e, 2)] for lab, e, _ in snaps], "incidents": []}
    print(q, "replay equals the stage's answer:", out["replay_equals_stage"], flush=True)
    x = P.final
    e_x = P.energy(x)
    reg = regions_of(P, x)
    reg_size = np.bincount(reg)
    for rec in doc["list"]:
        if rec["quarter"] != q:
            continue
        z = np.load(CENSUS_DIR / "answers" / f"cheapest-right-{rec['id']}.npz")
        y = x.copy()
        y[z["lots"]] = P.to_index(z["codes"].astype(str))
        dE = P.energy(y) - e_x
        source = "cheapest answer right on the incident"
        if dE >= -1e-6 and not rec["solved_is_minimum"]:
            # the census's lower answer was the search confined to D (from the solved or the table's answer)
            D0 = changed_set(SimpleNamespace(table_codes=P.table_codes, codes=x), rec["lots"])
            xD = x.copy()
            xD[D0] = P.table_codes[D0]
            for start, x_ in (("solved", x), ("table", xD)):
                y_ = search_on(SimpleNamespace(U=P.U, odds=P.odds), x_, D0)
                if P.energy(y_) - e_x < dE:
                    y, dE, source = y_, P.energy(y_) - e_x, f"the search confined to D from the {start} answer"
        if dE >= -1e-6:
            continue
        I = np.array(rec["lots"])
        D = np.flatnonzero(y != x)
        trans = Counter(zip(names[x[D]].tolist(), names[y[D]].tolist()))
        row = {"id": rec["id"], "lots": rec["lots_count"], "table": rec["table"], "solved": rec["solved"],
               "dE": r(dE), "answer": source, "changed": int(len(D)),
               "incident_right_after": r(np.mean(y[I] == P.table_codes[I])),
               "transitions": [[f, t, int(n)] for (f, t), n in trans.most_common(8)]}
        # expansion: each code y gives, as its own move
        exp = []
        for al in np.unique(y[D]):
            S = D[y[D] == al]
            s = np.zeros(C.n, bool)
            s[S] = True
            qa = x.copy()
            qa[S] = al
            parts = L.expansion_parts(P.U, P.a, P.b, P.odds, L.SAME, x, al)
            e0 = L.bounded_energy(parts, np.zeros(C.n, bool))
            n_cut, d_cut, b_cut = cut_best(P, x, al)
            yq, labelled, nvar, d_q = L.fuse(P, x, np.full(C.n, al))
            exp.append({"code": names[al], "lots": int(len(S)), "true_dE": r(P.energy(qa) - e_x),
                        "bounded_dE": r(L.bounded_energy(parts, s) - e0), **overcharge(P, x, al, s),
                        "cut_best": {"lots": n_cut, "true_dE": d_cut, "bounded_dE": b_cut},
                        "qpbo_expansion": {"lots_switched": int((yq != x).sum()), "labelled": labelled,
                                           "variables": nvar, "dE": r(d_q),
                                           "switches_in_change": int(((yq != x) & (y == yq) & (y != x)).sum())}})
        row["expansion"] = exp
        # regions
        touched = np.unique(reg[D])
        regs = []
        for g in touched[np.argsort(-np.bincount(reg[D], minlength=reg.max() + 1)[touched])][:6]:
            members = np.flatnonzero(reg == g)
            inD = np.isin(members, D)
            # recode the whole region to y's main code on its changed part
            tgt = Counter(y[members[inD]].tolist()).most_common(1)[0][0]
            qr = x.copy()
            qr[members] = tgt
            # no-line borders inside the region between its changed and unchanged parts
            m_in = np.zeros(C.n, bool)
            m_in[members] = True
            m_d = np.zeros(C.n, bool)
            m_d[D] = True
            inner = m_in[P.a] & m_in[P.b] & (P.odds <= 0) & (m_d[P.a] != m_d[P.b])
            regs.append({"code": names[x[members[0]]], "lots": int(reg_size[g]), "in_change": int(inD.sum()),
                         "recode_whole_to": names[tgt], "recode_whole_dE": r(P.energy(qr) - e_x),
                         "no_line_borders_to_cut": int(inner.sum()),
                         "their_cost": r(np.maximum(-P.odds[inner], 0).sum()),
                         "their_median_p": r(np.median(1 / (1 + np.exp(-P.odds[inner])))) if inner.any() else None})
        row["regions"] = regs
        row["region_change_is_whole_regions"] = bool(all(g["in_change"] == g["lots"] for g in regs))
        # fusion with y itself
        yf, labelled, nvar, d_f = L.fuse(P, x, y)
        row["fusion_with_y"] = {"labelled": labelled, "variables": nvar, "dE": r(d_f),
                                "takes_all_of_y": bool(np.array_equal(yf, y))}
        # history on the incident's lots
        row["history_on_incident"] = [[lab, counts(names, sn, I, 3)] for lab, _, sn in snaps]
        row["history_on_change"] = [[lab, counts(names, sn, D, 4)] for lab, _, sn in snaps]
        out["incidents"].append(row)
        print(json.dumps(row)[:3000], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"diagnose-{q}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("quarter")
    main(ap.parse_args().quarter)
