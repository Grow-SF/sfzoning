"""Printed lines between lots of one code: how many the walls read, whether the printed labels could tell them, and
what the solve would gain if the energy did not charge them (a confined oracle, no full solve).

    uv run python scripts/incidents/same_code_lines.py census [--walls DIR]
    uv run python scripts/incidents/same_code_lines.py labels [--walls DIR] [--trust T ...]
    uv run python scripts/incidents/same_code_lines.py oracle [--incident ID ...] [--rings N]
    uv run python scripts/incidents/same_code_lines.py summary

census   city-wide, the held-out pairs (both lots known, each pair read with its first lot's quarter's odds) the walls
         read as a line (p > 0.5) whose two lots share the table's code (corrected, the scores' target): their number
         and ft by kind of print (walls_compare.py describe), against the read lines the table has; and per census
         incident of 20+ lots, the ft of such lines touching it and the log odds they charge a right answer.
labels   could the labels tell these lines without the table?  The printed regions are the lots joined across pairs
         read as no line (p <= 0.5); a region's confident codes are the top readings (trust >= T) of the labels whose
         text lies on its lots, read with the labels' own held-out quarter.  A read line between two regions that
         share a confident code is flagged.  Per T: the flagged pairs and ft, the share whose lots share the table's
         code (precision), and the share of all same-code read lines flagged (recall).  Measured, never an input.
oracle   for census incidents (default the walls-driven ones), the solver's own search (common.search_on) confined to
         the incident's changed set D plus N rings, from the solved answer and from the table's, under the run's saved
         costs with the odds changed on the pairs touching that set: "table" sets to 0 the odds of every read line
         whose lots share the table's code (an oracle: the most a fix of the same-code charge could gain there);
         "labels" does the same only on the lines `labels` flags at trust 0.9 (no table); "missed" (the other
         direction, an oracle) sets to 0 the odds of every table line the walls read as no line (odds < 0), so a code
         change there costs nothing; "both" does "table" and "missed" together.  "none" changes nothing but searches
         from the table's answer too, so the oracles' gains are measured against it.  Per variant: the lowest energy
         found against the solved answer's, and the scored lots right on the set before and after.
Reads only; writes OUT/same_code.json (walls_compare.OUT)."""
import argparse
import json
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.special import expit

from common import CENSUS_DIR, changed_set, energy_on, load_costs, mask, ring, search_on
from sfzmaps.solver import paths, runs
from sfzmaps.solver.city import city
from sfzmaps.solver.codes import table_vocabulary
from sfzmaps.solver.labels import readings
from walls_compare import KINDS, OUT, PLACES, kinds, odds_per_pair, place

WALL_DRIVEN = (2, 3, 6, 8)        # the census's walls-driven incidents of 100+ lots


def basics(walls_dir: Path):
    C = city()
    a, b = C.a, C.b
    p = expit(odds_per_pair(walls_dir))
    known = C.known[a] & C.known[b] & (C.table[a] != "") & (C.table[b] != "")
    same = known & (C.table[a] == C.table[b])
    return C, p, known, same


def census(walls_dir: Path) -> dict:
    C, p, known, same = basics(walls_dir)
    d = dict(np.load(OUT / "pair_print.npz"))
    K, Pl = kinds(d), place(d)
    read = p > 0.5
    ft = C.border_ft
    out = {"same_code_read": {"pairs": int((same & read).sum()), "ft": float(ft[same & read].sum())},
           "table_lines_read": {"pairs": int((known & ~same & read).sum()),
                                "ft": float(ft[known & ~same & read].sum())},
           "by_kind": {}}
    print(f"read lines (p > .5) on known pairs: {out['table_lines_read']['ft']:,.0f} ft on table lines, "
          f"{out['same_code_read']['ft']:,.0f} ft between lots of one table code "
          f"({out['same_code_read']['pairs']:,} pairs)")
    for ki, kn in enumerate(KINDS):
        for pi_, pn in enumerate(PLACES):
            m = (K == ki) & (Pl == pi_) & read & known
            if m.any():
                s, t = float(ft[m & same].sum()), float(ft[m & ~same].sum())
                out["by_kind"][f"{kn} / {pn}"] = {"same_code_ft": s, "table_line_ft": t}
                print(f"  {kn:16} {pn:18} same code {s:9,.0f} ft  table lines {t:10,.0f} ft  "
                      f"(same-code share {s / max(s + t, 1):.1%})")
    doc = json.loads((CENSUS_DIR / "incidents.json").read_text())
    rows = []
    for rec in doc["list"]:
        inI = mask(C.n, rec["lots"])
        P = np.flatnonzero((inI[C.a] | inI[C.b]) & same & read)
        lo = np.log(p[P] / (1 - p[P]))
        rows.append({"id": rec["id"], "lots": rec["lots_count"], "pairs": int(len(P)), "ft": float(ft[P].sum()),
                     "log_odds": float(lo.sum()), "kinds": dict(Counter(KINDS[k] for k in K[P]))})
    print("census incidents with 200+ ft of same-code read lines touching them (id, lots, pairs, ft, log odds):")
    for r in sorted(rows, key=lambda r: -r["ft"]):
        if r["ft"] >= 200:
            print(f"  #{r['id']:<3} {r['lots']:5} lots: {r['pairs']:3} pairs {r['ft']:6,.0f} ft, log odds "
                  f"{r['log_odds']:+7.1f}; {r['kinds']}")
    out["incidents"] = rows
    return out


def region_codes(p: np.ndarray, trust_min: float):
    """printed regions (lots joined across pairs read as no line) and each region's confident codes"""
    C = city()
    m = p <= 0.5
    _, comp = connected_components(coo_matrix((np.ones(int(m.sum())), (C.a[m], C.b[m])), shape=(C.n, C.n)),
                                   directed=False)
    codes = defaultdict(set)
    vocab = table_vocabulary(C.lots)
    for qi, q in enumerate(paths.QUARTERS):
        rd = readings(q, vocab)
        best = {}
        for lab, lot, code, t, far in zip(rd.label, rd.lot, rd.code, rd.trust, rd.far):
            if far or C.quarter[lot] != qi:
                continue
            if t > best.get(int(lab), (None, -1.0, -1))[1]:
                best[int(lab)] = (rd.names[code], float(t), int(lot))
        for name, t, lot in best.values():
            if t >= trust_min:
                codes[int(comp[lot])].add(name)
    return comp, codes


def flagged(p: np.ndarray, trust_min: float, sole: bool = True) -> np.ndarray:
    """read lines between two printed regions that share a confident code (sole: each region's confident codes are
    that one code alone, so a region leaking through a gap into its neighbours, which gathers many codes, flags
    nothing)"""
    C = city()
    comp, codes = region_codes(p, trust_min)
    ra, rb = comp[C.a], comp[C.b]
    read = (p > 0.5) & (ra != rb)
    out = np.zeros(len(p), bool)
    for i in np.flatnonzero(read):
        ca, cb = codes.get(int(ra[i])), codes.get(int(rb[i]))
        if ca and cb and (ca == cb and len(ca) == 1 if sole else ca & cb):
            out[i] = True
    return out


def labels(walls_dir: Path, trusts) -> dict:
    C, p, known, same = basics(walls_dir)
    read = p > 0.5
    ft = C.border_ft
    tot = float(ft[same & read].sum())
    out = {}
    for sole in (False, True):
        for t in trusts:
            f = flagged(p, t, sole)
            fk = f & known
            prec = float(ft[fk & same].sum() / max(ft[fk].sum(), 1))
            rec = float(ft[f & same & read].sum() / max(tot, 1))
            out[f"{'sole' if sole else 'shared'} {t}"] = {
                "flagged_pairs": int(f.sum()), "flagged_ft": float(ft[f].sum()),
                "flagged_known_ft": float(ft[fk].sum()), "precision_ft": prec, "recall_ft": rec,
                "flagged_table_line_ft": float(ft[fk & ~same].sum())}
            print(f"{'one code each' if sole else 'a code shared'}, trust >= {t}: flagged {int(f.sum()):,} pairs "
                  f"{ft[f].sum():,.0f} ft; on known pairs {ft[fk].sum():,.0f} ft, of which same table code "
                  f"{prec:.1%} (table lines {ft[fk & ~same].sum():,.0f} ft); recall of same-code read lines "
                  f"{rec:.1%} of {tot:,.0f} ft")
    return out


def oracle(ids, rings: int) -> dict:
    C = city()
    doc = json.loads((CENSUS_DIR / "incidents.json").read_text())
    run = doc["run"]
    flags = {}
    out = []
    for ident in ids:
        rec = next(t for t in doc["list"] if t["id"] == ident)
        q = rec["quarter"]
        costs = load_costs(run, q)
        D = changed_set(costs, np.array(rec["lots"]))
        S = ring(D, rings)
        inS = mask(C.n, S)
        touch = inS[C.a] | inS[C.b]
        known = C.known[C.a] & C.known[C.b] & (C.table[C.a] != "") & (C.table[C.b] != "")
        same = known & (C.table[C.a] == C.table[C.b])
        if q not in flags:
            pq = expit(costs.odds)
            flags[q] = flagged(pq, 0.9)
        x0 = costs.codes
        xt = x0.copy()
        t = costs.table_codes[S]
        xt[S] = np.where(t >= 0, t, x0[S])
        sc = C.scored & mask(C.n, S)
        right0 = int((sc & (C.table == np.array(costs.names, dtype=object)[x0].astype(str))).sum())
        r = {"id": ident, "quarter": q, "set_lots": int(len(S)), "scored": int(sc.sum()), "right_solved": right0}
        line = known & (C.table[C.a] != C.table[C.b])
        for name, sel in (("none", np.zeros(len(C.a), bool)), ("table", touch & same & (costs.odds > 0)),
                          ("labels", touch & flags[q] & (costs.odds > 0)),
                          ("missed", touch & line & (costs.odds < 0)),
                          ("both", touch & ((same & (costs.odds > 0)) | (line & (costs.odds < 0))))):
            odds = costs.odds.copy()
            odds[sel] = 0.0
            cm = replace(costs, odds=odds)
            e0 = energy_on(cm, x0, S)
            best, be = x0, e0
            for start in (x0, xt):
                x = search_on(cm, start, S)
                e = energy_on(cm, x, S)
                if e < be - 1e-9:
                    best, be = x, e
            right = int((sc & (C.table == np.array(costs.names, dtype=object)[best].astype(str))).sum())
            r[name] = {"pairs_changed": int(sel.sum()), "ft_changed": float(C.border_ft[sel].sum()),
                       "energy_change": round(be - e0, 3), "right": right}
            print(f"#{ident} ({q}, {len(S)} lots, {int(sc.sum())} scored; right as solved {right0}): {name}: "
                  f"{int(sel.sum())} pairs zeroed ({C.border_ft[sel].sum():,.0f} ft); lowest energy found "
                  f"{be - e0:+.1f} against the solved answer; right {right}", flush=True)
        out.append(r)
    return {"run": run, "rings": rings, "incidents": out}


VARIANTS = ("table", "labels", "missed", "both")


def summary(doc: dict) -> None:
    """the oracles' gains in scored lots right over "none", each set counted once (incidents sharing a changed set
    give the same set: same quarter, size, scored lots and right lots)"""
    seen, tot = {}, Counter()
    for r in doc["oracle"]["incidents"]:
        key = (r["quarter"], r["set_lots"], r["scored"], r["right_solved"])
        if key in seen:
            seen[key]["ids"].append(r["id"])
            continue
        seen[key] = {"ids": [r["id"]], **{v: r[v]["right"] - r["none"]["right"] for v in VARIANTS},
                     "searched": r["none"]["right"] - r["right_solved"]}
    print("set (incidents): search from the table's answer alone | gains over it: " + ", ".join(VARIANTS))
    for s in seen.values():
        tot.update({v: s[v] for v in VARIANTS + ("searched",)})
        if any(s[v] for v in VARIANTS + ("searched",)):
            print(f"  #{'/#'.join(map(str, s['ids']))}: {s['searched']:+} | " +
                  ", ".join(f"{v} {s[v]:+}" for v in VARIANTS))
    print(f"total over {len(seen)} sets: search alone {tot['searched']:+} | " +
          ", ".join(f"{v} {tot[v]:+}" for v in VARIANTS))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("census")
    lb = sub.add_parser("labels")
    o = sub.add_parser("oracle")
    for x in (c, lb):
        x.add_argument("--walls", type=Path, default=runs.WALLS_DIR)
    lb.add_argument("--trust", type=float, nargs="+", default=[0.8, 0.9, 0.95])
    o.add_argument("--incident", type=int, nargs="+", default=list(WALL_DRIVEN))
    o.add_argument("--rings", type=int, default=2)
    sub.add_parser("summary")
    args = ap.parse_args()
    f = OUT / "same_code.json"
    doc = json.loads(f.read_text()) if f.is_file() else {}
    if args.cmd == "census":
        doc["census"] = census(args.walls)
    elif args.cmd == "labels":
        doc["labels"] = labels(args.walls, args.trust)
    elif args.cmd == "summary":
        summary(doc)
        raise SystemExit
    else:
        doc["oracle"] = oracle(args.incident, args.rings)
        summary(doc)
    OUT.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(doc, indent=1) + "\n")
