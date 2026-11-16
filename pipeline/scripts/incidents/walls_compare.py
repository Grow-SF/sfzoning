"""How two wall folders differ, on what kind of printed thing, and what that does to the solves made with them.

    uv run python scripts/incidents/walls_compare.py describe
    uv run python scripts/incidents/walls_compare.py pairs [--walls-a DIR] [--walls-b DIR]
    uv run python scripts/incidents/walls_compare.py sets [--run-a RUN] [--run-b RUN] [--walls-a DIR] [--walls-b DIR]
                                                         [--min-lots N]
    uv run python scripts/incidents/walls_compare.py incident ID [--run-b RUN] [--walls-a DIR] [--walls-b DIR]

describe   per neighbour pair, what the raw merged page prints along its border (the border px as `wallevidence.Border`
           reads them), into OUT/pair_print.npz:
             near_share   the share of border px with a stroke (`strokes.npz` centre line) within NEAR_FT
             w_near       the mean median width (ft) of the nearest stroke over those px (0 where none)
             t_near       the mean text chance of that stroke (`wallevidence.stroke_text_chance` on the walls stage's
                          text_chance.png: the detector's chance the fit read)
             len_near     the mean log10 length (ft) of that stroke
             heavy_share  the share of border px within HEAVY_REACH of a stroke at least HEAVY_FT wide whose text
                          chance is below one half
             ink_at       the share of border px inked on the page; text_at  their mean text chance
             touching     the pair's lots touch (else they face across a street); street_ft  the gap between the
                          two lots' outlines (0 when touching)
pairs      city-wide, per kind of printed thing (KINDS: what the descriptors above say lies along the border), the
           held-out pairs (both lots known and in one quarter, that quarter's odds; the corrected table, the scores'
           target): their number and ft, the table-line share, each folder's mean probability and log loss (summed:
           the kind's share of the gap), and the ft each folder reads on the wrong side of one half where the other
           does not.  Only pairs both folders read (J7's import gives a pair it lacks exactly 0).
sets       the changed sets of two runs (connected lots whose codes differ; default final and the solve with J7's
           walls): per set its lots, scored lots right under each run, and its deciding pairs (touching the set,
           drawn by one run and not the other) by kind, with each folder's probability and whether the table has a
           line there; then the sum, per kind, of the deciding ft where only one folder sides with the table, over
           the sets each run wins.  Also writes OUT/sets.json.
incident   a census incident's table lines left out and lines drawn that the table lacks (pairs touching it), grouped
           by the two sides' blocks and codes (table, the census run's, run B's), with their kind and each folder's
           median probability.
Defaults: walls A the stage's (runs.WALLS_DIR), B J7's (`import_j7.py`).  The kinds are descriptive bins for this
report, never an input to anything.  Reads only; writes into OUT = data/solver/experiments/incidents-walls/."""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.special import expit

from common import J7_DIR, J7_WALLS_RUN, run_codes
from sfzmaps.solver import grid, lots, page, paths, runs, wallevidence
from sfzmaps.solver.city import city

OUT = runs.EXPERIMENTS_DIR / "incidents-walls"
NEAR_FT = 8.0           # a border px "has a stroke" when a centre line is this close (2 grid px)
HEAVY_FT = 20.0         # a heavy stroke: median width at least this (the 80th percentile of stroke length by width)
HEAVY_REACH = 14.0      # ... within this of the border px
TEXT = 0.5              # a stroke is text when its mean text chance is at least this
WIDE_STREET = 82.5      # ft: wider than the widest common street width (walls.STREET_WIDTH_KNOTS)
FLOOR = 1e-6


def describe() -> None:
    import geopandas as gpd
    import shapely
    L = lots.load_lots()
    Bd = wallevidence.Border(L)
    strokes = dict(np.load(paths.LABELS_DIR / "strokes.npz"))
    text = cv2.imread(str(runs.WALLS_DIR / "text_chance.png"), cv2.IMREAD_GRAYSCALE)
    st_text = wallevidence.stroke_text_chance(strokes, text)
    wmed = strokes["wmed"].astype(float)
    loglen = np.log10(np.maximum(strokes["length"].astype(float), 1.0))
    sid = wallevidence.stroke_raster(strokes, L.lot.shape)
    near, dist = Bd.nearest(sid)
    del sid
    ok = (near >= 0) & (dist <= NEAR_FT)
    k = np.maximum(near, 0)
    share = Bd.mean(ok.astype(float))
    den = np.maximum(share, 1e-9)
    out = {"near_share": share,
           "w_near": np.where(share > 0, Bd.mean(np.where(ok, wmed[k], 0.0)) / den, 0.0),
           "t_near": np.where(share > 0, Bd.mean(np.where(ok, st_text[k], 0.0)) / den, 0.0),
           "len_near": np.where(share > 0, Bd.mean(np.where(ok, loglen[k], 0.0)) / den, 0.0)}
    sid = wallevidence.stroke_raster(strokes, L.lot.shape, only=(wmed >= HEAVY_FT) & (st_text < TEXT))
    near2, dist2 = Bd.nearest(sid)
    del sid
    out["heavy_share"] = Bd.mean(((near2 >= 0) & (dist2 <= HEAVY_REACH)).astype(float))
    ink = page.load_ink()
    out["ink_at"] = Bd.mean(ink[Bd.rows, Bd.cols].astype(float))
    out["text_at"] = Bd.mean(text[Bd.rows, Bd.cols] / 255.0)
    out["touching"] = L.touching > 0
    geoms = gpd.read_parquet(paths.LOTS_DIR / "table.parquet").geometry.values
    street = np.zeros(len(L.a))
    s = ~out["touching"]
    street[s] = shapely.distance(geoms[L.a[s]], geoms[L.b[s]])
    out["street_ft"] = street
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "pair_print.npz", **{k_: np.asarray(v) for k_, v in out.items()})
    print(f"wrote {OUT / 'pair_print.npz'}: {len(L.a):,} pairs; strokes within {NEAR_FT:.0f} ft on "
          f"{(share > 0.5).mean():.1%} of pairs' borders (most px)")


KINDS = ("text", "heavy line", "thin line", "nothing printed")


def kinds(d) -> np.ndarray:
    """each pair's kind (index into KINDS): text when the nearest strokes are mostly text; a heavy line when a heavy
    non-text stroke runs along most of the border; a thin line when strokes run along most of it; else nothing"""
    k = np.full(len(d["near_share"]), 3)
    k[d["near_share"] >= 0.5] = 2
    k[(d["heavy_share"] >= 0.5) & (d["t_near"] < TEXT)] = 1
    k[(d["near_share"] >= 0.5) & (d["t_near"] >= TEXT)] = 0
    return k


def place(d) -> np.ndarray:
    """0 touching lots, 1 across a street, 2 across a wide roadway or open ground (gap above WIDE_STREET)"""
    return np.where(d["touching"], 0, np.where(d["street_ft"] > WIDE_STREET, 2, 1))


PLACES = ("touching", "street", "wide roadway/open")


def odds_per_pair(walls_dir: Path) -> np.ndarray:
    """each pair's held-out log odds: the fit without the quarter of its first lot"""
    C = city()
    q = C.quarter[C.a]
    lo = np.zeros(len(C.a))
    for qi, name in enumerate(paths.QUARTERS):
        v = np.load(Path(walls_dir) / f"walls_{name}.npz")["lo_ab"].astype(float)
        lo[q == qi] = v[q == qi]
    return lo


def loss(lo, y):
    p = np.clip(expit(lo), FLOOR, 1 - FLOOR)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def pairs(walls_a: Path, walls_b: Path, examples: int = 0) -> None:
    C = city()
    d = dict(np.load(OUT / "pair_print.npz"))
    K, Pl = kinds(d), place(d)
    la, lb = odds_per_pair(walls_a), odds_per_pair(walls_b)
    a, b = C.a, C.b
    held = C.known[a] & C.known[b] & (C.quarter[a] == C.quarter[b]) & (C.table[a] != "") & (C.table[b] != "")
    common = held & (lb != 0.0) & (la != 0.0)
    y = (C.table[a] != C.table[b]).astype(float)
    ft = C.border_ft
    LA, LB = loss(la, y), loss(lb, y)
    pa, pb = expit(la), expit(lb)
    print(f"held-out pairs both read: {int(common.sum()):,} of {int(held.sum()):,}; "
          f"mean log loss A {LA[common].mean():.4f}"
          f" B {LB[common].mean():.4f} (A {walls_a}, B {walls_b})")
    print("kind / place: pairs, ft, table-line share; mean p A, B on table lines | on other pairs; log loss sum A, B "
          "(B - A); ft only A wrong side of .5 (lines missed, false lines); ft only B wrong side")
    rows = []
    for ki, kn in enumerate(KINDS):
        for pi_, pn in enumerate(PLACES):
            m = common & (K == ki) & (Pl == pi_)
            if not m.any():
                continue
            ln, nl = m & (y == 1), m & (y == 0)
            a_wrong = np.where(y == 1, pa < 0.5, pa > 0.5)
            b_wrong = np.where(y == 1, pb < 0.5, pb > 0.5)
            only_a, only_b = m & a_wrong & ~b_wrong, m & b_wrong & ~a_wrong
            r = {"kind": kn, "place": pn, "pairs": int(m.sum()), "ft": float(ft[m].sum()),
                 "line_share": float(y[m].mean()),
                 "p_a_lines": float(pa[ln].mean()) if ln.any() else None,
                 "p_b_lines": float(pb[ln].mean()) if ln.any() else None,
                 "p_a_other": float(pa[nl].mean()) if nl.any() else None,
                 "p_b_other": float(pb[nl].mean()) if nl.any() else None,
                 "loss_a": float(LA[m].sum()), "loss_b": float(LB[m].sum()),
                 "only_a_wrong_ft": [float(ft[only_a & (y == 1)].sum()), float(ft[only_a & (y == 0)].sum())],
                 "only_b_wrong_ft": [float(ft[only_b & (y == 1)].sum()), float(ft[only_b & (y == 0)].sum())]}
            rows.append(r)
            f = (lambda v: "  -  " if v is None else f"{v:.3f}")
            print(f"{kn:16} {pn:18} {r['pairs']:7,} {r['ft']:10,.0f} {r['line_share']:.3f}; "
                  f"{f(r['p_a_lines'])} {f(r['p_b_lines'])} | {f(r['p_a_other'])} {f(r['p_b_other'])}; "
                  f"{r['loss_a']:8.0f} {r['loss_b']:8.0f} ({r['loss_b'] - r['loss_a']:+7.0f}); "
                  f"A only {r['only_a_wrong_ft'][0]:8,.0f} {r['only_a_wrong_ft'][1]:8,.0f}; "
                  f"B only {r['only_b_wrong_ft'][0]:8,.0f} {r['only_b_wrong_ft'][1]:8,.0f}")
    tot_a, tot_b = sum(r["loss_a"] for r in rows), sum(r["loss_b"] for r in rows)
    print(f"total log loss A {tot_a:.0f}, B {tot_b:.0f} (B - A {tot_b - tot_a:+.0f})")
    if examples:
        blk = C.lots["blklot"].astype(str).values
        mid = (C.centroid[a] + C.centroid[b]) / 2
        a_wrong = np.where(y == 1, pa < 0.5, pa > 0.5)
        b_wrong = np.where(y == 1, pb < 0.5, pb > 0.5)
        for ki, kn in enumerate(KINDS):
            for pi_, pn in enumerate(PLACES):
                for yy, what in ((1, "table lines"), (0, "pairs of one code")):
                    for who, sel in (("A", a_wrong & ~b_wrong), ("B", b_wrong & ~a_wrong)):
                        m = np.flatnonzero(common & (K == ki) & (Pl == pi_) & (y == yy) & sel)
                        if len(m) < 50:
                            continue
                        top = m[np.argsort(-ft[m])][:examples]
                        print(f"  {kn} / {pn}, {what} only {who} reads wrong ({len(m)} pairs): " + "; ".join(
                            f"{blk[a[i]]}|{blk[b[i]]} {ft[i]:.0f} ft at ({mid[i, 0]:.0f}, {mid[i, 1]:.0f}) "
                            f"p {pa[i]:.2f}/{pb[i]:.2f}" for i in top))
    (OUT / "pairs.json").write_text(json.dumps({"walls_a": str(walls_a), "walls_b": str(walls_b), "rows": rows},
                                               indent=1) + "\n")


def sets(run_a: str, run_b: str, walls_a: Path, walls_b: Path, min_lots: int) -> None:
    C = city()
    d = dict(np.load(OUT / "pair_print.npz"))
    K, Pl = kinds(d), place(d)
    la, lb = odds_per_pair(walls_a), odds_per_pair(walls_b)
    pa, pb = expit(la), expit(lb)
    a, b = C.a, C.b
    A, B = run_codes(run_a), run_codes(run_b)
    differ = A != B
    m = differ[a] & differ[b]
    _, comp = connected_components(coo_matrix((np.ones(int(m.sum())), (a[m], b[m])), shape=(C.n, C.n)),
                                   directed=False)
    da, db = A[a] != A[b], B[a] != B[b]
    both = C.known[a] & C.known[b]
    tline = both & (C.table[a] != C.table[b])
    blk = C.lots["blklot"].astype(str).values
    sizes = Counter(comp[differ].tolist())
    right_a = C.scored & (A == C.table)
    right_b = C.scored & (B == C.table)
    print(f"{run_a} vs {run_b}: {int(differ.sum()):,} lots differ in {len(sizes):,} sets; scored lots right "
          f"{int(right_a.sum()):,} vs {int(right_b.sum()):,} (B - A {int(right_b.sum() - right_a.sum()):+,})")
    inset = np.zeros(C.n, int) - 1
    inset[differ] = comp[differ]
    # deciding pairs: drawn by one run and not the other, touching a changed set
    dec = (da != db) & ((inset[a] >= 0) | (inset[b] >= 0))
    set_of_pair = np.where(inset[a] >= 0, inset[a], inset[b])
    out = []
    win = defaultdict(lambda: np.zeros((len(KINDS), 2)))   # per winner, per kind: ft where only A / only B
    for c, n in sizes.most_common():                       # sides with the table
        S = np.flatnonzero(differ & (comp == c))
        gain = int(right_b[S].sum() - right_a[S].sum())
        P = np.flatnonzero(dec & (set_of_pair == c))
        sa = (pa[P] > 0.5) == tline[P]       # folder A's odds side with the table on this pair
        sb = (pb[P] > 0.5) == tline[P]
        rec = {"set": int(c), "lots": int(n), "quarter": paths.QUARTERS[Counter(C.quarter[S].tolist())
                                                                          .most_common(1)[0][0]],
               "right_a": int(right_a[S].sum()), "right_b": int(right_b[S].sum()), "gain_b": gain,
               "codes": [[f"{t}|{x}|{z}", int(k)] for (t, x, z), k in
                         Counter(zip(C.table[S], A[S], B[S])).most_common(4)],
               "blocks": [bk for bk, _ in Counter(blk[S].astype("U4").tolist()).most_common(4)],
               "window": [float(v) for v in window_of(S)],
               "deciding": {"pairs": int(len(P)), "ft": float(C.border_ft[P].sum()),
                            "by_kind": {KINDS[ki]: {"ft": float(C.border_ft[P][K[P] == ki].sum()),
                                                    "only_a_sides_with_table_ft":
                                                        float(C.border_ft[P][(K[P] == ki) & sa & ~sb].sum()),
                                                    "only_b_sides_with_table_ft":
                                                        float(C.border_ft[P][(K[P] == ki) & sb & ~sa].sum())}
                                        for ki in range(len(KINDS))}}}
        if gain:
            w = "b" if gain > 0 else "a"
            for ki in range(len(KINDS)):
                win[w][ki] += [rec["deciding"]["by_kind"][KINDS[ki]]["only_a_sides_with_table_ft"],
                               rec["deciding"]["by_kind"][KINDS[ki]]["only_b_sides_with_table_ft"]]
        out.append(rec)
        if n >= min_lots:
            top = P[np.argsort(-np.abs(lb[P] - la[P]) * C.border_ft[P])][:5]
            print(f"== set {c}: {n} lots ({rec['quarter']}, blocks {', '.join(rec['blocks'])}); right {run_a} "
                  f"{rec['right_a']}, {run_b} {rec['right_b']} ({gain:+}); codes (table|A|B) {rec['codes']}")
            print(f"   deciding pairs {len(P)}, {rec['deciding']['ft']:.0f} ft; by kind (ft; only A / only B sides "
                  "with table): " + "; ".join(f"{k_} {v['ft']:.0f} ({v['only_a_sides_with_table_ft']:.0f} / "
                                               f"{v['only_b_sides_with_table_ft']:.0f})"
                                               for k_, v in rec["deciding"]["by_kind"].items() if v["ft"]))
            for i in top:
                print(f"     {blk[a[i]]}|{blk[b[i]]} table {C.table[a[i]]}|{C.table[b[i]]} {PLACES[Pl[i]]} "
                      f"{C.border_ft[i]:.0f} ft, {KINDS[K[i]]} (width {d['w_near'][i]:.0f}, text {d['t_near'][i]:.2f}"
                      f", heavy {d['heavy_share'][i]:.2f}): p A {pa[i]:.3f}, B {pb[i]:.3f}; drawn A {bool(da[i])},"
                      f" B {bool(db[i])}")
    tot = sum(r["gain_b"] for r in out)
    wins_b = sum(r["gain_b"] for r in out if r["gain_b"] > 0)
    wins_a = -sum(r["gain_b"] for r in out if r["gain_b"] < 0)
    print(f"sum over sets: {run_b} gains {wins_b:,} lots where it wins, loses {wins_a:,} where {run_a} wins; "
          f"net {tot:+,}")
    for w, name in (("b", run_b), ("a", run_a)):
        print(f"sets {name} wins: deciding ft where only A / only B sides with the table, by kind: " +
              "; ".join(f"{KINDS[ki]} {win[w][ki][0]:,.0f} / {win[w][ki][1]:,.0f}" for ki in range(len(KINDS))))
    (OUT / "sets.json").write_text(json.dumps({"run_a": run_a, "run_b": run_b, "walls_a": str(walls_a),
                                               "walls_b": str(walls_b), "sets": out}, indent=1) + "\n")


def incident(ident: int, run_b: str, walls_a: Path, walls_b: Path) -> None:
    """a census incident's border pairs that matter: the table lines the run leaves out and the lines it draws that
    the table lacks (pairs touching the incident), grouped by the two sides' blocks, with their kind, each folder's
    probability, and whether run B draws them"""
    from common import CENSUS_DIR
    doc = json.loads((CENSUS_DIR / "incidents.json").read_text())
    rec = next(t for t in doc["list"] if t["id"] == ident)
    C = city()
    d = dict(np.load(OUT / "pair_print.npz"))
    K, Pl = kinds(d), place(d)
    q = paths.QUARTERS.index(rec["quarter"])
    lo_a = np.load(Path(walls_a) / f"walls_{rec['quarter']}.npz")["lo_ab"].astype(float)
    lo_b = np.load(Path(walls_b) / f"walls_{rec['quarter']}.npz")["lo_ab"].astype(float)
    A, B = run_codes(doc["run"]), run_codes(run_b)
    a, b = C.a, C.b
    inI = np.zeros(C.n, bool)
    inI[rec["lots"]] = True
    touch = inI[a] | inI[b]
    both = C.known[a] & C.known[b]
    tline = both & (C.table[a] != C.table[b])
    da, db = A[a] != A[b], B[a] != B[b]
    blk = C.lots["blklot"].astype(str).str[:4].values
    print(f"#{ident}: {rec['lots_count']} lots {rec['table']} -> {rec['solved']} ({rec['quarter']}, quarter {q}); "
          f"run A {doc['run']}, run B {run_b}")
    for title, sel in (("table lines left out", touch & tline & ~da),
                       ("lines drawn not in table", touch & both & ~tline & da)):
        P = np.flatnonzero(sel)
        print(f"  {title}: {len(P)} pairs, {C.border_ft[P].sum():.0f} ft; by kind " +
              ", ".join(f"{KINDS[k]} {C.border_ft[P][K[P] == k].sum():.0f} ft" for k in range(len(KINDS))) +
              "; by place " + ", ".join(f"{PLACES[k]} {C.border_ft[P][Pl[P] == k].sum():.0f} ft"
                                        for k in range(len(PLACES))))
        groups = defaultdict(list)
        for i in P:
            groups[(blk[a[i]], blk[b[i]], C.table[a[i]], C.table[b[i]], A[a[i]], A[b[i]], B[a[i]], B[b[i]])].append(i)
        for key, ks in sorted(groups.items(), key=lambda kv: -C.border_ft[kv[1]].sum()):
            ks = np.array(ks)
            print(f"    blocks {key[0]}|{key[1]} table {key[2]}|{key[3]} A {key[4]}|{key[5]} B {key[6]}|{key[7]}: "
                  f"{len(ks)} pairs {C.border_ft[ks].sum():.0f} ft; kinds {Counter(KINDS[k] for k in K[ks])}; "
                  f"p A {np.median(expit(lo_a[ks])):.3f}, B {np.median(expit(lo_b[ks])):.3f}; drawn by B "
                  f"{int(db[ks].sum())}/{len(ks)}")


def window_of(S, margin_ft: float = 400.0):
    """the map-ft box (x0, y0, x1, y1) of the lots S's px, plus a margin"""
    C = city()
    inS = np.zeros(C.n + 1, bool)
    inS[S] = True
    lot = C.lot_raster
    rr, cc = np.nonzero(inS[np.where(lot >= 0, lot, C.n)])
    (x0, y1), (x1, y0) = grid.to_map(np.array([[cc.min(), rr.min()], [cc.max() + 1, rr.max() + 1]], float))
    return x0 - margin_ft, y0 - margin_ft, x1 + margin_ft, y1 + margin_ft


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("describe")
    p = sub.add_parser("pairs")
    s = sub.add_parser("sets")
    i_ = sub.add_parser("incident")
    i_.add_argument("id", type=int)
    for x in (p, s, i_):
        x.add_argument("--walls-a", type=Path, default=runs.WALLS_DIR)
        x.add_argument("--walls-b", type=Path, default=J7_DIR)
    for x in (s, i_):
        x.add_argument("--run-b", default=J7_WALLS_RUN)
    s.add_argument("--run-a", default=runs.FINAL_RUN)
    s.add_argument("--min-lots", type=int, default=20)
    p.add_argument("--examples", type=int, default=0, help="list this many longest pairs per kind, place, truth "
                                                           "and folder that alone reads them wrong")
    args = ap.parse_args()
    if args.cmd == "describe":
        describe()
    elif args.cmd == "pairs":
        pairs(args.walls_a, args.walls_b, args.examples)
    elif args.cmd == "incident":
        incident(args.id, args.run_b, args.walls_a, args.walls_b)
    else:
        sets(args.run_a, args.run_b, args.walls_a, args.walls_b, args.min_lots)
