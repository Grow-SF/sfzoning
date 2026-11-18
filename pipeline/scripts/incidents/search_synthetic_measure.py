"""The numbers the compound fusion move's two synthetic cases draw from (tests/test_solver_search.py), measured on the
real map: each quarter's energy rebuilt from its held-out inputs (search_lab.problem), the stage's answer x and the
census's lower answer right on the incident y (incidents-census/answers/cheapest-right-<id>.npz; for #14 the
census's search confined to the changed set, as search_diagnose.py).

The cases (Ruling R10) start from x and test the basin: x is a local minimum of the stage's moves on the incident and
the compound fusion move reaches y.  Each case is the incident's neighbourhood, measured block by block:
  * the changed lots, one block per transition (x code -> y code), each with its lots, the borders inside it and
    the labels charged on it (readings, trusts, the placement's chance there, far-side copies);
  * every x-region touching the changed lots, region by region, each with its code, its lots touching them, its
    unchanged lots, the borders inside it (quantiles) and the labels reading its own code charged on it;
  * the regions whose labels trigger the compound proposal that offers the incident's main block its next code
    (search.compound_proposal on x), when that is not the main block's own region, likewise;
  * the borders between blocks: between the lots touching the changed set (and the changed lots), and every border
    between a trigger region and another block's region.
Per incident also, for the report: the energy of each part alone and together, and for the bound shape the
overcharged two-code line borders of the expansion move (the bound's charge against the true change).

(a) The chain: #1/#34 sw, #7 se, #9/#44 se (#34 is #1's tip; #44's answer holds #9's).
(b) The expansion-bound shape: #4/#23 se, #14 sw.

    uv run python scripts/incidents/search_synthetic_measure.py              measured.json and costs.npz
    uv run python scripts/incidents/search_synthetic_measure.py --fixture    the tests' fixtures from them

Writes data/solver/experiments/incidents-search-synthetic/measured.json, and with --fixture the part of it the tests'
samplers read.  Read-only on the solver."""
import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.sparse import coo_matrix

import search_lab as L
from common import CENSUS_DIR, changed_set, search_on
from sfzmaps.solver import energy as E
from sfzmaps.solver import runs, search
from sfzmaps.solver import paths
from sfzmaps.solver.city import city, wall_odds
from sfzmaps.solver.codes import table_vocabulary
from sfzmaps.solver.labels import readings
from sfzmaps.solver.placement import load_options

OUT = runs.EXPERIMENTS_DIR / "incidents-search-synthetic"
FIXTURE = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "search_synthetic_measured.json"
COSTS_FIXTURE = FIXTURE.with_name("search_synthetic_costs.npz")
CASES = {"chain": {1: ("sw", 1), 7: ("se", 7), 9: ("se", 44)},     # id: quarter, the census answer used
         "bound": {4: ("se", 4), 14: ("sw", 14)}}
QUANTILES = np.linspace(0, 1, 101)
KEEP_ODDS = 400            # a border class with more odds than this is kept as its quantiles


def answer(P, i):
    """the census's cheapest answer right on incident i; when it is not lower than x (#14), the lower answer the
    census found by the search confined to the changed set from the table's answer (as search_diagnose.py)"""
    z = np.load(CENSUS_DIR / "answers" / f"cheapest-right-{i}.npz")
    x = P.final
    y = x.copy()
    y[z["lots"]] = P.to_index(z["codes"].astype(str))
    if P.energy(y) < P.energy(x) - 1e-6:
        return y
    rec = next(t for t in json.loads((CENSUS_DIR / "incidents.json").read_text())["list"] if t["id"] == i)
    D0 = changed_set(SimpleNamespace(table_codes=P.table_codes, codes=x), rec["lots"])
    xD = x.copy()
    xD[D0] = P.table_codes[D0]
    return min((search_on(SimpleNamespace(U=P.U, odds=P.odds), s, D0) for s in (x, xD)), key=P.energy)


def label_groups(q, P):
    """(label, far) -> {'top', 'trust', 'readings', 'where': [(lot, chance)]}, as energy.label_costs charges them"""
    C = city()
    rd = readings(q, table_vocabulary(C.lots))
    options = load_options(q, runs.solve_dir(runs.FINAL_RUN))
    K = len(rd.names)
    groups = {}
    for r in range(len(rd.lot)):
        groups.setdefault((int(rd.label[r]), bool(rd.far[r])), []).append(r)
    out = {}
    for (label, far), rows in groups.items():
        T = np.zeros(K)
        for r in rows:
            T[rd.code[r]] = max(T[rd.code[r]], rd.trust[r])
        top = int(np.argmax(T))
        own = int(rd.lot[rows[0]])
        where = [(own, 1.0)]
        if not far and label in options:
            stay = [o for o in options[label] if o[2]]
            if stay and stay[0][0] == own:
                where = [(o[0], o[1]) for o in options[label]]
        out[(label, far)] = {"top": P.names[top], "trust": round(float(T[top]), 4),
                             "readings": sorted(((P.names[k], round(float(T[k]), 4)) for k in np.flatnonzero(T)),
                                                key=lambda z: -z[1]),
                             "where": where}
    return out, K


def labels_on(groups, mask, top=None):
    """the label groups charged on the lots of mask (reading `top` first, when given): readings and the charge there"""
    rows = []
    for (label, far), g in groups.items():
        on = [(lot, pi) for lot, pi in g["where"] if mask[lot]]
        if on and (top is None or g["top"] == top):
            rows.append({"label": label, "far": far, "readings": g["readings"],
                         "chance": round(sum(pi for _, pi in on), 4)})
    return sorted(rows, key=lambda z: (z["label"], z["far"]))


def label_lots(groups, mask):
    """the lot of mask where each label charged on it is charged most"""
    lots = set()
    for g in groups.values():
        on = [(pi, lot) for lot, pi in g["where"] if mask[lot]]
        if on:
            lots.add(int(max(on)[1]))
    return lots


def dist(odds):
    """a border class: its count and its odds (or their quantiles when there are many)"""
    odds = np.asarray(odds, float)
    out = {"count": int(len(odds))}
    if not len(odds):
        return out
    out["median_p"] = round(float(np.median(1 / (1 + np.exp(-odds)))), 4)
    out["line_cost"] = round(float(np.maximum(odds, 0).sum()), 3)
    out["no_line_cost"] = round(float(np.maximum(-odds, 0).sum()), 3)
    if len(odds) <= KEEP_ODDS:
        out["odds"] = [round(float(v), 4) for v in odds]
    else:
        out["quantiles"] = [round(float(v), 4) for v in np.quantile(odds, QUANTILES)]
    return out


def trigger_regions(P, x, X, reg):
    """the regions whose labels trigger the compound proposal (search.compound_proposal's order) and whose offer
    covers X's lots: X's own region, or one touching it whose favoured code X's region holds"""
    N, K = P.U.shape
    R = reg.max() + 1
    rc = np.zeros(R, int)
    rc[reg] = x
    SU = np.zeros((R, K))
    np.add.at(SU, reg, P.U)
    want = SU.argmin(1)
    gain = SU[np.arange(R), rc] - SU[np.arange(R), want]
    e = reg[P.a] != reg[P.b]
    nbr = coo_matrix((np.ones(int(e.sum())), (reg[P.a[e]], reg[P.b[e]])), shape=(R, R)).tocsr()
    nbr = ((nbr + nbr.T) > 0).tocsr()
    cand = np.flatnonzero((want != rc) & (gain > 1e-3))
    cand = cand[np.argsort(-gain[cand], kind="stable")]
    rX = set(np.unique(reg[X]).tolist())
    taken = np.zeros(R, bool)
    out = []
    for r_ in cand:
        if taken[r_]:
            continue
        taken[r_] = True
        offered = []
        for n_ in nbr.indices[nbr.indptr[r_]:nbr.indptr[r_ + 1]]:
            if rc[n_] == want[r_] and not taken[n_]:
                taken[n_] = True
                offered.append(int(n_))
        if r_ in rX or rX & set(offered):
            out.append({"region": int(r_), "own": bool(r_ in rX), "favours": P.names[want[r_]],
                        "gain": round(float(gain[r_]), 3)})
    return out


def neighbourhood(P, x, y, groups, main):
    """the incident's blocks (see the module's docstring), the map lots each stands for, and the map's borders
    among those lots, by the pair of blocks they join: main is the changed block whose triggers are measured (the
    chain's D, the bound shape's X)"""
    names = np.array(P.names)
    a, b, odds = P.a, P.b, P.odds
    N = len(x)
    ch = x != y
    reg, _ = search._regions(a, b, odds, x)
    block = np.full(N, -1)
    blocks = []
    for (f, t), n in Counter(zip(x[ch].tolist(), y[ch].tolist())).most_common():
        m = ch & (x == f) & (y == t)
        block[m] = len(blocks)
        blocks.append({"name": f"C{len(blocks)}", "changed": True, "x": names[f], "y": names[t], "region": None,
                       "unchanged_region_lots": int(n), "internal": dist(odds[m[a] & m[b]]),
                       "labels": labels_on(groups, m), "near": m})
    touch = np.zeros(N, bool)
    touch[b[ch[a]]] = True
    touch[a[ch[b]]] = True
    touch &= ~ch
    region_block = {}

    def add_region(r_, kind, near):
        m = (reg == r_) & ~ch
        code = names[x[np.flatnonzero(m)[0]]]
        region_block[r_] = len(blocks)
        blocks.append({"name": f"{kind}{len(blocks)}", "changed": False, "x": code, "y": code, "region": int(r_),
                       "unchanged_region_lots": int(m.sum()), "internal": dist(odds[m[a] & m[b]]),
                       "labels": labels_on(groups, m, top=code), "near": m & near})
        return m

    for r_, _ in Counter(reg[touch].tolist()).most_common():
        add_region(r_, "N", touch)
    triggers = trigger_regions(P, x, ch & (x == P.names.index(blocks[main]["x"]))
                               & (y == P.names.index(blocks[main]["y"])), reg)
    for t in triggers:                                   # a trigger is its whole region's labels: measured whole
        if t["own"]:
            t["block"] = main
            continue
        mW = (reg == t["region"]) & ~ch
        if t["region"] in region_block:
            blocks[region_block[t["region"]]]["near"] = mW.copy()
        else:
            for r2, k2 in list(region_block.items()):   # and the other regions' lots it borders
                m2 = (reg == r2) & ~ch
                between = (mW[a] & m2[b]) | (mW[b] & m2[a])
                ends = np.r_[a[between], b[between]]
                blocks[k2]["near"][ends[m2[ends]]] = True
            add_region(t["region"], "W", mW)
        t["block"] = region_block[t["region"]]
    # each block's lots: its lots near the incident and the lots carrying its own labels; a lot in one block only
    rows, row_block = [], []
    for k, z in enumerate(blocks):
        m = z.pop("near")
        lots = set(np.flatnonzero(m).tolist())
        if not z["changed"]:
            lots |= label_lots(groups, (reg == z["region"]) & ~ch)
        lots = sorted(lots)
        z["lots"] = len(lots)
        z["map_lots"] = lots
        rows += lots
        row_block += [k] * len(lots)
    rows = np.asarray(rows, int)
    pos = np.full(N, -1)
    pos[rows] = np.arange(len(rows))
    row_block = np.asarray(row_block, int)
    both = (pos[a] >= 0) & (pos[b] >= 0)
    one = (pos[a] >= 0) != (pos[b] >= 0)
    pa, pb, po = pos[a[both]], pos[b[both]], odds[both]
    ka, kb = row_block[pa], row_block[pb]
    swap = ka > kb
    pa, pb, ka, kb = np.where(swap, pb, pa), np.where(swap, pa, pb), np.minimum(ka, kb), np.maximum(ka, kb)
    classes, cls = np.unique(np.c_[ka, kb], axis=0, return_inverse=True)
    borders = []
    for c, (i, j) in enumerate(classes):
        # a block's own borders draw from its whole inside (a region's, over all its lots); between blocks, from
        # the borders measured between them
        d = blocks[i]["internal"] if i == j else dist(po[cls == c])
        borders.append({"a": int(i), "b": int(j), "pairs": int((cls == c).sum()), **d})
    return {"blocks": blocks, "borders": borders, "triggers": triggers,
            "rows": rows, "row_block": row_block, "pairs": np.c_[pa, pb, cls.ravel()],
            "pair_index": np.flatnonzero(both),
            # the borders from the blocks' lots to the rest of the city, held at x: each an exact term of its lot
            "edge_index": np.flatnonzero(one), "edge_row": pos[np.where(pos[a] >= 0, a, b)[one]],
            "edge_code": x[np.where(pos[a] >= 0, b, a)[one]],
            "row_region": np.unique(reg[rows], return_inverse=True)[1]}


def overcharge(P, x, X, al):
    """the expansion move X -> al: its overcharged two-code line borders (the bound charges an X lot whose
    neighbour holds a third code and stays), the true and the bounded energy change"""
    a, b, t, ra, rb, rr = E.wall_terms(P.a, P.b, P.odds, L.SAME)
    xa, xb = x[ra], x[rb]
    both = (xa != al) & (xb != al) & (xa != xb)
    a_keener = (P.U[ra, al] - P.U[ra, xa]) <= (P.U[rb, al] - P.U[rb, xb])
    charged = np.where(a_keener, rb, ra)
    other = np.where(a_keener, ra, rb)
    over = both & X[charged] & ~X[other]
    q = x.copy()
    q[X] = al
    parts = L.expansion_parts(P.U, P.a, P.b, P.odds, L.SAME, x, al)
    return {"overcharged": {k: v for k, v in dist(rr[over]).items() if k not in ("odds", "quantiles")},
            "codes_across": Counter(np.array(P.names)[x[other[over]]].tolist()).most_common(),
            "true_dE": round(P.energy(q) - P.energy(x), 3),
            "bounded_dE": round(L.bounded_energy(parts, X) - L.bounded_energy(parts, np.zeros(len(x), bool)), 3)}


def measure(kind, P, groups, K, i, q, which):
    x, y = P.final, answer(P, which)
    hood = neighbourhood(P, x, y, groups, main=0)
    b0 = hood["blocks"][0]
    idx, edge = hood.pop("pair_index"), hood.pop("edge_index")
    walls = [wall_odds(q2) for q2 in paths.QUARTERS]
    fits = np.stack([w[idx] for w in walls], 1)          # each border's odds in the four quarter fits
    edge_fits = np.stack([w[edge] for w in walls], 1)
    costs = {"U": P.U[hood.pop("rows")].astype(np.float32), "block": hood.pop("row_block").astype(np.int16),
             "pairs": hood.pop("pairs").astype(np.int32), "odds": P.odds[idx].astype(np.float32),
             "odds_sd": fits.std(1).astype(np.float32),
             "edge_row": hood.pop("edge_row").astype(np.int32), "edge_code": hood.pop("edge_code").astype(np.int16),
             "edge_odds": P.odds[edge].astype(np.float32), "edge_sd": edge_fits.std(1).astype(np.float32),
             "region": hood.pop("row_region").astype(np.int32)}
    hood["edge_borders"] = int(len(edge))
    hood["odds_spread"] = {"median_sd": round(float(np.median(fits.std(1))), 4),
                           "q90_sd": round(float(np.quantile(fits.std(1), 0.9)), 4)}
    out = {"quarter": q, "K": K, "names": list(P.names), "kind": kind, **hood,
           "map": {"energy_change": round(P.energy(y) - P.energy(x), 3),
                   "x_right_on_main": 0.0, "y_lots": int((x != y).sum())}}
    m0 = (x == P.names.index(b0["x"])) & (y == P.names.index(b0["y"]))
    q0 = x.copy()
    q0[m0] = y[m0]
    out["map"]["main_alone"] = round(P.energy(q0) - P.energy(x), 3)
    if kind == "chain":
        r = b0["y"]
        tip = next(k for k, z in enumerate(hood["blocks"]) if z["changed"] and z["x"] == r)
        out["incident"] = {"D": 0, "T": tip}
        out["parts"] = {"D": 0, "T": tip}
        mt = (x == P.names.index(r)) & (y == P.names.index(hood["blocks"][tip]["y"]))
        q1 = x.copy()
        q1[mt] = y[mt]
        out["map"]["tip_alone"] = round(P.energy(q1) - P.energy(x), 3)
    else:
        out["incident"] = {"X": 0}
        out["parts"] = {"X": 0}
        out["map"].update(overcharge(P, x, m0, P.names.index(b0["y"])))
    # the incident's blocks: the parts with lots in the solve's own quarter (the join scores each lot from the solve
    # that held its quarter out: #7's tip lies in sw, where #1's template (#34) covers it)
    C = city()
    k_q = paths.QUARTERS.index(q)
    out["incident"] = {name: k for name, k in out["parts"].items()
                       if (C.quarter[np.asarray(hood["blocks"][k]["map_lots"])] == k_q).any()}
    return out, costs


def fixture(measured: dict) -> dict:
    """what tests/test_solver_search.py's samplers read, beside search_synthetic_costs.npz (per incident
    `<kind>-<id>-U`: the cost rows U of the map lots the blocks stand for, labels, placement and tie-break included;
    `-block`: each row's block; `-pairs`: the map's borders among those lots, as two rows and a border class (the
    pair of blocks); `-odds`: each border's odds in the quarter's fit; `-odds_sd`: their spread over the four quarter
    fits; `-edge_*`: the borders from those lots to the rest of the city, which is held at x: the lot's row, the code
    across, the odds and their spread; `-region`: each row's x-region on the map): the code names (U's columns), each
    block's x and y codes and lots, the incident's blocks, and the map's numbers for the record"""
    out = {"source": "pipeline/scripts/incidents/search_synthetic_measure.py --fixture, from "
                     "data/solver/experiments/incidents-search-synthetic/measured.json and costs.npz"}
    for kind in CASES:
        out[kind] = {}
        for i, m in measured[kind].items():
            out[kind][i] = {
                "quarter": m["quarter"], "names": m["names"], "incident": m["incident"],
                "map": {**m["map"], "odds_spread": m["odds_spread"], "edge_borders": m["edge_borders"]},
                "blocks": [{"name": z["name"], "changed": z["changed"], "x": z["x"], "y": z["y"], "lots": z["lots"]}
                           for z in m["blocks"]]}
    return out


def main():
    if "--fixture" in sys.argv:
        FIXTURE.write_text(json.dumps(fixture(json.loads((OUT / "measured.json").read_text())), indent=1) + "\n")
        z = np.load(OUT / "costs.npz")
        np.savez_compressed(COSTS_FIXTURE, **{k: z[k] for k in z.files})
        print(f"wrote {FIXTURE} and {COSTS_FIXTURE}")
        return
    out = {}
    costs = {}
    cache = {}
    for kind, table in CASES.items():
        out[kind] = {}
        for i, (q, which) in table.items():
            if q not in cache:
                P = L.problem(q)
                cache[q] = (P, *label_groups(q, P))
            P, groups, K = cache[q]
            out[kind][str(i)], c = measure(kind, P, groups, K, i, q, which)
            costs.update({f"{kind}-{i}-{k}": v for k, v in c.items()})
            m = out[kind][str(i)]
            print(kind, i, q, len(m["blocks"]), "blocks", len(m["borders"]), "border classes", json.dumps(m["map"]),
                  flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "measured.json").write_text(json.dumps(out, indent=1))
    np.savez_compressed(OUT / "costs.npz", **costs)
    print(f"wrote {OUT / 'measured.json'} and {OUT / 'costs.npz'}")


if __name__ == "__main__":
    main()
