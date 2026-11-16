"""Candidate improvements to the solver's search, measured as experiments.  The solver's code is not changed: each
variant calls the solver's own moves (search.expansion_moves / region_moves / compound_moves) and adds moves of its
own here, on the same energy as the stage's solve (search_lab.problem: rebuilt from each quarter's held-out inputs).

    uv run python scripts/incidents/search_experiments.py VARIANT QUARTER     one quarter's solve
    uv run python scripts/incidents/search_experiments.py VARIANT --score     join the four and score them

Variants (every one deterministic; a seeded one says so):
  qpbo        the stage's search, then the exact expansion move (no bound on the same-code-across-a-line cost)
              by roof duality (QPBO; unlabelled lots keep their code) for every code, alternated with the solver's
              region, compound and expansion moves until the energy stops falling
  fusion      the stage's search, then QPBO fusion with table-free proposals (the solver's own answers on the way:
              its start = each lot's cheapest code, the end of its no-same-code stage, the end of its first
              same-code stage; each region its labels' favourite and second favourite code; each region the code of
              a neighbouring region that its labels favour most), alternated with the solver's moves until the energy
              stops falling
  starts      the solver's own sequence from other starts (without its first, no-same-code stage; codes taken in
              reverse order; three seeded shuffles of the code order), each end fused into the stage's answer by
              QPBO, then the solver's moves
  regions     the stage's search, then region moves on finer regions (lots of one code joined only across borders
              whose odds are at most -1, -2, -3, -4 or -6: a weak no-line border no longer ties two parts together),
              alternated with the solver's moves
  anneal      the stage's search, then simulated annealing over region recodes on the finer regions (seeded),
              then the solver's moves; the lowest energy seen is kept
  all         qpbo, fusion and regions together, alternated until nothing improves

Each quarter's answer goes to data/solver/experiments/incidents-search-<variant>/runs/incidents-search-<variant>-<q>.npz
with its energy and steps in steps-<q>.json; --score joins them (solve.join) and writes measures.json and
measures.md (the six measures on every lot, against final's)."""
import argparse
import json
import time
from collections import Counter

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

import search_lab as L
from sfzmaps.solver import energy as E
from sfzmaps.solver import paths, search
from sfzmaps.solver.city import city

JOIN_LADDER = (-1.0, -2.0, -3.0, -4.0, -6.0)    # the finer regions' joins: borders with odds at most this
OUTER = 8                                       # rounds of new moves + the solver's, at most


def solver_order(p, K):
    from collections import Counter
    seen = set(p.tolist())
    return [k for k, _ in Counter(p.tolist()).most_common()] + [k for k in range(K) if k not in seen]


def qpbo_expansions(P, p, log, sweeps=3):
    """exact expansion moves by QPBO, every code in the solver's order, until a sweep changes nothing"""
    K = P.U.shape[1]
    for sw in range(sweeps):
        moved, gain = 0, 0.0
        for al in solver_order(p, K):
            q, _, _, d = L.fuse(P, p, np.full(len(p), al), improve=True)
            if d < 0:
                moved += int((q != p).sum())
                gain += d
                p = q
        log.append(["qpbo expansion", sw, moved, round(P.energy(p), 3)])
        if not moved:
            break
    return p


def region_stats(P, p, join=0.0):
    """regions of p (one code, joined across borders with odds <= join), their label sums SU[region, code]"""
    a, b = P.a, P.b
    same = (p[a] == p[b]) & (P.odds <= join)
    _, reg = connected_components(coo_matrix((np.ones(int(same.sum())), (a[same], b[same])), shape=(len(p),) * 2),
                                  directed=False)
    R = reg.max() + 1
    SU = np.zeros((R, P.U.shape[1]))
    np.add.at(SU, reg, P.U)
    return reg, R, SU


def proposals(P, p):
    """table-free proposals around p: each lot's cheapest code; each region (of p) its labels' favourite and second
    favourite code; each region the neighbouring region's code its labels favour most"""
    out = [("each lot's cheapest code", P.U.argmin(1))]
    reg, R, SU = region_stats(P, p)
    order = np.argsort(SU, 1)
    out.append(("each region its labels' favourite code", order[reg, 0]))
    out.append(("each region its labels' second code", order[reg, 1]))
    x = reg[P.a] != reg[P.b]
    ra, rb = reg[P.a[x]], reg[P.b[x]]
    rc = np.zeros(R, int)
    rc[reg] = p
    # per region, the neighbouring regions' codes; take the one its own label sum favours most
    best = rc.copy()
    best_cost = np.full(R, np.inf)
    for r1, r2 in ((ra, rb), (rb, ra)):
        c = rc[r2]
        cost = SU[r1, c]
        o = np.lexsort((cost, r1))
        first = np.r_[True, r1[o][1:] != r1[o][:-1]]
        rr_, cc_, co_ = r1[o][first], c[o][first], cost[o][first]
        better = co_ < best_cost[rr_]
        best[rr_[better]], best_cost[rr_[better]] = cc_[better], co_[better]
    out.append(("each region its labels' favourite neighbouring code", best[reg]))
    # the compound move as one proposal, without its two restrictions: every region whose labels favour another
    # code takes it, and the lots of each touching region holding that code offer their own next code (by U, any
    # code but that one, also codes their neighbours hold); the fusion decides lot by lot which of it to take
    want = order[:, 0]
    gain = SU[np.arange(R), rc] - SU[np.arange(R), want]
    y = p.copy()
    taken = np.zeros(R, bool)
    nbr = coo_matrix((np.ones(len(ra)), (ra, rb)), shape=(R, R)).tocsr()
    nbr = ((nbr + nbr.T) > 0).tocsr()
    member_start = np.argsort(reg, kind="stable")
    st = np.searchsorted(reg[member_start], np.arange(R + 1))
    for r_ in np.flatnonzero((want != rc) & (gain > 1e-3))[np.argsort(-gain[(want != rc) & (gain > 1e-3)])]:
        if taken[r_]:
            continue
        lots_r = member_start[st[r_]:st[r_ + 1]]
        y[lots_r] = want[r_]
        taken[r_] = True
        for n_ in nbr.indices[nbr.indptr[r_]:nbr.indptr[r_ + 1]]:
            if rc[n_] != want[r_] or taken[n_]:
                continue
            lots_n = member_start[st[n_]:st[n_ + 1]]
            Un = P.U[lots_n].copy()
            Un[:, want[r_]] = np.inf
            y[lots_n] = Un.argmin(1)
            taken[n_] = True
    out.append(("compound proposal: favoured codes, the neighbours holding them each lot its next code", y))
    return out


def stage_answers(P):
    """the solver's own answers on the way (its sequence's start and the ends of its first two stages): table-free
    proposals the later stages moved away from"""
    p0 = P.U.argmin(1)
    s0 = search.expansion_moves(P.U, P.a, P.b, P.odds, 0.0, p0)
    s1 = search.expansion_moves(P.U, P.a, P.b, P.odds, L.SAME, s0)
    return [("the sequence's start", p0), ("the end of its no-same-code stage", s0),
            ("the end of its first same-code stage", s1)]


def fusions(P, p, log, extra=()):
    for name, y in list(extra) + proposals(P, p):
        q, labelled, nvar, d = L.fuse(P, p, y, improve=True)
        log.append(["fusion: " + name, nvar, labelled, round(d, 3)])
        if d < 0:
            p = q
    return p


def region_moves_join(P, p, join, rounds=50):
    """search.region_moves with regions joined only across borders with odds <= join (the solver joins at 0):
    recode a region to its cheapest code, or swap two touching regions' codes; the best non-overlapping ones
    together, kept when the exact energy falls.  Copied from the solver with the join made a parameter."""
    U, a, b, odds = P.U, P.a, P.b, P.odds
    N, K = U.shape
    t = np.maximum(-odds, 0.0)
    rr = np.maximum(odds, 0.0) * L.SAME
    p = p.copy()
    for rd in range(rounds):
        reg, R, SU = region_stats(P, p, join)
        rc = np.zeros(R, int)
        rc[reg] = p
        x = reg[a] != reg[b]
        ra_, rb_ = reg[a[x]], reg[b[x]]
        lo_r, hi_r = np.minimum(ra_, rb_), np.maximum(ra_, rb_)
        uk, inv = np.unique(lo_r.astype(np.int64) * R + hi_r, return_inverse=True)
        Tp = np.bincount(inv, t[x])
        Rp = np.bincount(inv, rr[x])
        P1, P2 = uk // R, uk % R
        codes = np.arange(K)[None, :]
        Em = SU.copy()
        np.add.at(Em, P1, Tp[:, None] * (codes != rc[P2][:, None]) + Rp[:, None] * (codes == rc[P2][:, None]))
        np.add.at(Em, P2, Tp[:, None] * (codes != rc[P1][:, None]) + Rp[:, None] * (codes == rc[P1][:, None]))
        best = Em.argmin(1)
        gain = Em[np.arange(R), rc] - Em[np.arange(R), best]
        c1, c2 = rc[P1], rc[P2]
        e1 = Em[P1, c2] - Em[P1, c1] - (Rp - Tp)
        e2 = Em[P2, c1] - Em[P2, c2] - (Rp - Tp)
        dsw = np.where(c1 != c2, e1 + e2, 0.0)
        cand = [(g_, 0, r_) for r_, g_ in zip(np.flatnonzero(gain > 1e-6), gain[gain > 1e-6])]
        cand += [(-d_, 1, i_) for i_, d_ in zip(np.flatnonzero(dsw < -1e-6), dsw[dsw < -1e-6])]
        cand.sort(key=lambda z: (-z[0], z[1], z[2]))
        nbr = coo_matrix((np.ones(len(P1)), (P1, P2)), shape=(R, R)).tocsr()
        nbr = (nbr + nbr.T).tocsr()
        newc = rc.copy()
        used = np.zeros(R, bool)
        moved = 0
        for _, kind, i_ in cand:
            if kind == 0:
                nb = nbr.indices[nbr.indptr[i_]:nbr.indptr[i_ + 1]]
                if used[i_] or used[nb].any():
                    continue
                newc[i_] = best[i_]
                used[i_] = True
            else:
                r1, r2 = P1[i_], P2[i_]
                nb = np.r_[nbr.indices[nbr.indptr[r1]:nbr.indptr[r1 + 1]], nbr.indices[nbr.indptr[r2]:nbr.indptr[r2 + 1]]]
                if used[r1] or used[r2] or used[nb].any():
                    continue
                newc[r1], newc[r2] = rc[r2], rc[r1]
                used[r1] = used[r2] = True
            moved += 1
        q = newc[reg]
        if moved == 0 or P.energy(q) >= P.energy(p) - 1e-6:
            break
        p = q
    return p


def fine_regions(P, p, log):
    for join in JOIN_LADDER:
        e0 = P.energy(p)
        p = region_moves_join(P, p, join)
        log.append([f"region moves joined at odds <= {join:g}", round(P.energy(p) - e0, 3), round(P.energy(p), 3)])
    return p


def anneal(P, p, log, seed=0, sweeps=40, t0=2.0, t1=0.02):
    """simulated annealing over region recodes (regions joined at odds <= -3, seeded): each step a region takes a
    code held by a neighbouring region or its labels' favourite, accepted by the Metropolis rule; the partition is
    rebuilt each sweep; the lowest-energy answer seen is kept"""
    rng = np.random.default_rng(seed)
    U, a, b, odds = P.U, P.a, P.b, P.odds
    t = np.maximum(-odds, 0.0)
    rr = np.maximum(odds, 0.0) * L.SAME
    best_p, best_e = p.copy(), P.energy(p)
    e = best_e
    for sw in range(sweeps):
        T = t0 * (t1 / t0) ** (sw / max(sweeps - 1, 1))
        reg, R, SU = region_stats(P, p, -3.0)
        rc = np.zeros(R, int)
        rc[reg] = p
        x = reg[a] != reg[b]
        ea, eb, et, er = reg[a[x]], reg[b[x]], t[x], rr[x]
        # per region its boundary pairs, both directions
        src = np.r_[ea, eb]
        dst = np.r_[eb, ea]
        wt, wr = np.r_[et, et], np.r_[er, er]
        o = np.argsort(src, kind="stable")
        src, dst, wt, wr = src[o], dst[o], wt[o], wr[o]
        st = np.searchsorted(src, np.arange(R + 1))
        want = SU.argmin(1)
        accepted = 0
        for r_ in rng.permutation(R):
            nb = dst[st[r_]:st[r_ + 1]]
            if not len(nb):
                continue
            opts = np.unique(np.r_[rc[nb], want[r_]])
            opts = opts[opts != rc[r_]]
            if not len(opts):
                continue
            c = opts[rng.integers(len(opts))]
            w_t, w_r, nc = wt[st[r_]:st[r_ + 1]], wr[st[r_]:st[r_ + 1]], rc[nb]
            d = (SU[r_, c] - SU[r_, rc[r_]] + (w_t * ((nc != c).astype(float) - (nc != rc[r_]))).sum()
                 + (w_r * ((nc == c).astype(float) - (nc == rc[r_]))).sum())
            if d < 0 or rng.random() < np.exp(-d / T):
                rc[r_] = c
                e += d
                accepted += 1
        p = rc[reg]
        e = P.energy(p)
        if e < best_e - 1e-6:
            best_p, best_e = p.copy(), e
        log.append(["anneal sweep", sw, round(T, 4), accepted, round(e, 3)])
    return best_p


def solver_moves(P, p, log):
    steps = []
    p = L.polish(P.U, P.a, P.b, P.odds, p, steps)
    log.append(["solver's region, compound and expansion moves", len(steps), round(P.energy(p), 3)])
    return p


class Log(list):
    """the steps, each also printed as it comes (the run's progress)"""

    def append(self, item):
        super().append(item)
        print(json.dumps(item), flush=True)


def run_variant(variant: str, q: str) -> None:
    P = L.problem(q)
    log = Log()
    t0 = time.perf_counter()
    p = P.final.copy()
    e_final = P.energy(p)
    if variant == "starts":
        K = P.U.shape[1]
        rng = np.random.default_rng(0)
        ends = []
        p0 = P.U.argmin(1)
        x = search.expansion_moves(P.U, P.a, P.b, P.odds, L.SAME, p0)        # no first, no-same-code stage
        ends.append(("without the first stage", L.polish(P.U, P.a, P.b, P.odds, x)))
        for nm, perm in [("codes in reverse order", None)] + [(f"shuffle {s}", rng.permutation(K)) for s in range(3)]:
            ends.append((nm, ordered_sequence(P, p0, perm)))
        for nm, y in ends:
            log.append(["start: " + nm, round(P.energy(y), 3)])
        for nm, y in ends:
            q_, labelled, nvar, d = L.fuse(P, p, y, improve=True)
            log.append(["fusion with " + nm, nvar, labelled, round(d, 3)])
            if d < 0:
                p = q_
        p = solver_moves(P, p, log)
    else:
        extra = stage_answers(P) if variant in ("fusion", "all") else ()
        for rd in range(OUTER):
            e0 = P.energy(p)
            if variant in ("qpbo", "all"):
                p = qpbo_expansions(P, p, log)
            if variant in ("fusion", "all"):
                p = fusions(P, p, log, extra)
            if variant in ("regions", "all"):
                p = fine_regions(P, p, log)
            if variant == "anneal" and rd == 0:
                p = anneal(P, p, log)
            p = solver_moves(P, p, log)
            log.append(["round", rd, round(P.energy(p), 3)])
            if P.energy(p) >= e0 - 1e-6:
                break
    out = {"variant": variant, "quarter": q, "energy_final": round(e_final, 4), "energy": round(P.energy(p), 4),
           "dE": round(P.energy(p) - e_final, 4), "lots_changed": int((p != P.final).sum()), "steps": log}
    L.save_quarter(variant, P, p)
    (L.run_folder(variant) / f"steps-{q}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "steps"}),
          f"{time.perf_counter() - t0:.0f} s", flush=True)


def ordered_sequence(P, p0, perm):
    """solve_joint's sequence with the expansion moves taking the codes in another order (reverse when perm is
    None, else the permutation), from p0"""
    K = P.U.shape[1]

    def exp(p, same):
        for sweep in range(5):
            before = p.copy()
            order = solver_order(p, K)
            order = order[::-1] if perm is None else [k for k in perm]
            for al in order:
                p = one_expansion(P, p, al, same)
            if np.array_equal(p, before):
                break
        return p

    p = exp(p0.copy(), 0.0)
    p = exp(p, L.SAME)
    for _ in range(4):
        e0 = P.energy(p)
        p = search.region_moves(P.U, P.a, P.b, P.odds, L.SAME, p)
        p = search.compound_moves(P.U, P.a, P.b, P.odds, L.SAME, p)
        p = exp(p, L.SAME)
        if P.energy(p) >= e0 - 1e-6:
            break
    return p


def one_expansion(P, p, al, same):
    """one of search.expansion_moves' cuts (code al, its bound on the same-code cost), kept if the energy falls"""
    import maxflow
    c0, c1, ea, eb, w, _ = L.expansion_parts(P.U, P.a, P.b, P.odds, same, p, al)
    m = np.minimum(c0, c1)
    G = maxflow.Graph[float](len(p), len(ea))
    nodes = G.add_nodes(len(p))
    G.add_grid_tedges(nodes, c1 - m, c0 - m)
    G.add_edges(ea.astype(np.int64), eb.astype(np.int64), np.maximum(w, 0.0), np.zeros(len(w)))
    G.maxflow()
    s = G.get_grid_segments(nodes) & (p != al)
    if s.any():
        q = p.copy()
        q[s] = al
        if E.energy(P.U, P.a, P.b, P.odds, q, same) < E.energy(P.U, P.a, P.b, P.odds, p, same) - 1e-6:
            return q
    return p


def score(variant: str) -> None:
    m = L.join_and_score(variant)
    fin = L.final_measures()
    energies = {}
    for q in paths.QUARTERS:
        f = L.run_folder(variant) / f"steps-{q}.json"
        if f.is_file():
            s = json.loads(f.read_text())
            energies[q] = {"final": s["energy_final"], "variant": s["energy"], "dE": s["dE"],
                           "lots_changed": s["lots_changed"]}
    C = city()
    wrong_q = {}
    from sfzmaps.solver.city import load_run
    codes = load_run(L.SEARCH_PREFIX + variant, L.run_folder(variant))
    wrong = C.scored & (codes != C.table)
    for k, q in enumerate(paths.QUARTERS):
        wrong_q[q] = int((wrong & (C.quarter == k)).sum())
    final_codes = load_run("final")
    right_f = C.scored & (final_codes == C.table)
    right_v = C.scored & (codes == C.table)
    patches = {}
    for name, m_ in (("broken", right_f & ~right_v), ("fixed", ~right_f & right_v)):
        # connected patches of lots the variant breaks (right in final, wrong here) or fixes, by (table, final, variant)
        key = np.char.add(np.char.add(np.char.add(C.table, "|"), final_codes), np.char.add("|", codes))
        j = m_[C.a] & m_[C.b] & (key[C.a] == key[C.b])
        _, comp = connected_components(coo_matrix((np.ones(int(j.sum())), (C.a[j], C.b[j])), shape=(C.n, C.n)),
                                       directed=False)
        sizes = np.bincount(comp[m_], minlength=comp.max() + 1)
        blk = C.lots["blklot"].astype(str).str[:4].values
        rows = []
        for c in np.argsort(-sizes)[:8]:
            if sizes[c] == 0:
                break
            ix = np.flatnonzero(m_ & (comp == c))
            i0 = ix[0]
            rows.append({"lots": int(len(ix)), "table": str(C.table[i0]), "final": str(final_codes[i0]),
                         "variant": str(codes[i0]), "quarter": paths.QUARTERS[C.quarter[i0]],
                         "blocks": [b_ for b_, _ in Counter(blk[ix]).most_common(3)]})
        patches[name] = {"lots": int(m_.sum()), "largest": rows}
    out = {"variant": variant, "measures_every_lot": m, "final_every_lot": fin, "energies": energies,
           "wrong_lots_by_quarter": wrong_q, "patches": patches}
    (L.run_folder(variant) / "measures.json").write_text(json.dumps(out, indent=1))
    md = (f"# Search experiment: {variant}\n\n" + L.measures_table({"final": fin, variant: m}) + "\n"
          + "| Quarter | Energy, final | Energy, " + variant + " | Change | Lots changed | Wrong lots |\n"
          + "|---|---|---|---|---|---|\n"
          + "".join(f"| {q} | {e['final']:,.1f} | {e['variant']:,.1f} | {e['dE']:+.1f} | {e['lots_changed']:,} | "
                    f"{wrong_q[q]:,} |\n" for q, e in energies.items())
          + "".join(f"\n{name.capitalize()} lots ("
                    + ("right in final, wrong here" if name == "broken" else "wrong in final, right here")
                    + f"): {p['lots']:,}; largest patches: " + "; ".join(f"{r['lots']} {r['table']} (final {r['final']}, here {r['variant']}, "
                                           f"{r['quarter']}, blocks {', '.join(r['blocks'])})"
                                           for r in p["largest"]) + "\n"
                    for name, p in patches.items()))
    (L.run_folder(variant) / "measures.md").write_text(md)
    print(md)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("variant", choices=["qpbo", "fusion", "starts", "regions", "anneal", "all"])
    ap.add_argument("quarter", nargs="?")
    ap.add_argument("--score", action="store_true")
    args = ap.parse_args()
    if args.score:
        score(args.variant)
    else:
        run_variant(args.variant, args.quarter)
