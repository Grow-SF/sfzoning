"""Measuring a map against the April 2000 table by its mistakes, not by its lot count (one missing wall can flip a
thousand lots: that is one mistake).

  borders    recall: share of the table's district-line borders (both lots known) the map draws (its two lots get
             different codes); precision: share of the map's drawn borders that are table lines.  By count and length.
  labels     share of printed zoning labels whose district comes out right.  A label stands for the nearest table
             district carrying its top reading (the one under it when its lot has that code); right when most of that
             district's scored lots get that code.  Labels per quarter from that quarter's held-out print set; overlay
             names left out; a top reading naming no table code counts wrong.
  holes      short breaks in a drawn district line: a connected run of missed table-line borders (joined at
             junctions, any angle) up to 100 ft long that meets found table-line borders at two or more places
  incidents  wrong lots grouped into connected patches with the same table code and the same wrong code

Data errors (data_errors.npz, from holes.py): parcels the table codes differently from what the drawing shows.  They
are left out of the borders, labels and holes."""
from collections import Counter

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from .borders import junction_neighbours
from .data import OUT, QUARTERS, city
from .labels import top_readings

DATA_ERRORS = OUT / "data_errors.npz"
HOLE_FT = 100.0


def data_error_lots():
    C = city()
    m = np.zeros(C.n, bool)
    if DATA_ERRORS.exists():
        m[np.load(DATA_ERRORS)["lots"]] = True
    return m


class Table:
    """the table's district lines and districts, and the printed labels, prepared once"""

    def __init__(self, exclude):
        C = self.C = city()
        a, b = C.a, C.b
        self.both = C.known[a] & C.known[b] & ~exclude[a] & ~exclude[b]
        self.line = self.both & (C.table[a] != C.table[b])
        same = self.both & ~self.line                   # table districts: same-code lots joined through any border
        _, self.district = connected_components(coo_matrix((np.ones(int(same.sum())), (a[same], b[same])),
                                                           shape=(C.n, C.n)), directed=False)
        trees = {}

        def nearest_district(code, xy):
            if code not in trees:
                ix = np.flatnonzero(C.table == code)
                trees[code] = (cKDTree(C.centroid[ix]), ix) if len(ix) else None
            if trees[code] is None:
                return -1
            return self.district[trees[code][1][trees[code][0].query(xy)[1]]]

        labels = [(lot, code) for k, q in enumerate(QUARTERS) for _, lot, code in top_readings(q)
                  if C.quarter[lot] == k]
        vocab = set(C.table[C.known].tolist()) - {""}
        self.label_code = np.array([c for _, c in labels])
        self.label_in_vocab = np.array([c in vocab for c in self.label_code])
        self.label_district = np.array([self.district[lot] if C.table[lot] == c
                                        else nearest_district(c, C.centroid[lot]) for lot, c in labels])
        self.I, self.J = junction_neighbours()


def holes(T, drawn):
    """(holes: list of border-index arrays, missed runs over HOLE_FT: count and ft)"""
    C = T.C
    M = len(C.a)
    miss, found = T.line & ~drawn, T.line & drawn
    mm = miss[T.I] & miss[T.J]
    _, comp = connected_components(coo_matrix((np.ones(int(mm.sum())), (T.I[mm], T.J[mm])), shape=(M, M)),
                                   directed=False)
    mi = np.flatnonzero(miss)
    length = np.bincount(comp[mi], C.border_ft[mi], minlength=M)
    tf = miss[T.I] & found[T.J]
    meets = np.bincount(comp[T.I[tf]], minlength=M)
    runs = np.unique(comp[mi])
    hs = runs[(length[runs] <= HOLE_FT) & (meets[runs] >= 2)]
    longer = runs[length[runs] > HOLE_FT]
    order = np.argsort(comp[mi], kind="stable")
    by_run = np.split(mi[order], np.flatnonzero(np.diff(comp[mi][order])) + 1)
    members = {int(comp[m[0]]): m for m in by_run}
    return [members[int(h)] for h in hs], len(longer), float(length[longer].sum())


def score(codes, T):
    C = T.C
    a, b, L = C.a, C.b, C.border_ft
    drawn = codes[a] != codes[b]
    out = dict(recall=(T.line & drawn).sum() / T.line.sum(), recall_ft=L[T.line & drawn].sum() / L[T.line].sum(),
               precision=(T.line & drawn).sum() / max((T.both & drawn).sum(), 1),
               precision_ft=L[T.line & drawn].sum() / max(L[T.both & drawn].sum(), 1))
    nd = T.district.max() + 1
    lots_in = np.bincount(T.district[C.scored], minlength=nd)
    right_in = np.bincount(T.district[(codes == C.table) & C.scored], minlength=nd)
    d = np.maximum(T.label_district, 0)
    counted = (T.label_district < 0) | (lots_in[d] > 0)          # misread (names no table code) counts wrong
    right = (T.label_district >= 0) & (lots_in[d] > 0) & (right_in[d] > lots_in[d] / 2)
    zoning = counted & T.label_in_vocab
    out.update(labels=right.sum() / counted.sum(), n_labels=int(counted.sum()),
               zoning_labels=(right & zoning).sum() / zoning.sum(), n_zoning_labels=int(zoning.sum()))
    hs, n_long, long_ft = holes(T, drawn)
    out.update(holes=len(hs), hole_ft=float(sum(L[h].sum() for h in hs)), long_runs=n_long, long_miles=long_ft / 5280)
    return out


def incidents(codes):
    C = city()
    a, b = C.a, C.b
    wrong = C.scored & (codes != C.table)
    key = np.char.add(np.char.add(C.table, "|"), codes)
    m = wrong[a] & wrong[b] & (key[a] == key[b])
    _, comp = connected_components(coo_matrix((np.ones(int(m.sum())), (a[m], b[m])), shape=(C.n, C.n)), directed=False)
    sizes = Counter(comp[wrong].tolist())
    sz = np.array(sorted(sizes.values(), reverse=True))
    block = C.lots["blklot"].astype(str).str[:4].values
    biggest = []
    for c, n in sizes.most_common(6):
        mm = wrong & (comp == c)
        i = np.flatnonzero(mm)[0]
        biggest.append((int(n), C.table[i], codes[i], [b_ for b_, _ in Counter(block[mm]).most_common(3)]))
    return dict(wrong_lots=int(wrong.sum()), incidents=len(sz), median=int(np.median(sz)) if len(sz) else 0,
                big=int((sz >= 100).sum()), lots_in_big=int(sz[sz >= 100].sum()), biggest=biggest)


def report(name, codes, exclude_data_errors=True):
    ex = data_error_lots() if exclude_data_errors else np.zeros(city().n, bool)
    s = score(codes, Table(ex))
    i = incidents(codes)
    print(f"{name}: district lines drawn {s['recall']:.1%} ({s['recall_ft']:.1%} by length); drawn lines that are "
          f"table lines {s['precision']:.1%} ({s['precision_ft']:.1%}); zoning labels right "
          f"{s['zoning_labels']:.1%} of "
          f"{s['n_zoning_labels']:,} (all labels {s['labels']:.1%} of {s['n_labels']:,}); holes {s['holes']} "
          f"({s['hole_ft']:,.0f} ft; missed runs over 100 ft: {s['long_runs']}, {s['long_miles']:.1f} mi)"
          + (f" [{int(ex.sum())} data-error parcels left out]" if ex.any() else ""))
    print(f"   wrong lots {i['wrong_lots']:,} in {i['incidents']:,} incidents (median {i['median']} lots); "
          f"incidents of 100+ lots: {i['big']} holding {i['lots_in_big']:,}")
    for n, t, r, blocks in i["biggest"]:
        print(f"   {n:7,} lots  {t} -> {r or '(none)'}  blocks {', '.join(blocks)}")
    return s, i
