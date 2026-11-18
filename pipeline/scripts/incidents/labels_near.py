"""The labels around a census incident: every label whose readings or placement options touch its changed set D,
with its readings and, per run, its placement options on D.

    uv run python scripts/incidents/labels_near.py ID [--run RUN ...]

Per label: its position (map ft), the lot under its text (block-lot, table code, in D or not), its readings (code,
trust; far-side copies apart, with their lot), and for each run (default the census run; an experiment's placement
is read from its folder) its five likeliest placement options: the option's lot, its table code, the run's code
there, the chance, stay or move, and whether the lot is in D.  Reads only."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from common import CENSUS_DIR, changed_set, load_costs, mask, run_codes
from sfzmaps.solver import paths, runs
from sfzmaps.solver.city import city
from sfzmaps.solver.codes import table_vocabulary
from sfzmaps.solver.labels import readings
from sfzmaps.solver.placement import placement_file
from sfzmaps.solver.trust import load_prints


def main(incident: int, run_names) -> None:
    doc = json.loads((CENSUS_DIR / "incidents.json").read_text())
    rec = next(t for t in doc["list"] if t["id"] == incident)
    q = rec["quarter"]
    run_names = run_names or [doc["run"]]
    C = city()
    P = load_prints()
    block = C.lots["blklot"].astype(str).values
    D = changed_set(load_costs(doc["run"], q), np.array(rec["lots"]))
    inD = mask(C.n, D)
    rd = readings(q, table_vocabulary(C.lots))
    codes = {r: run_codes(r) for r in run_names}
    opts = {}
    for r in run_names:
        Z = np.load(placement_file(Path(runs.solve_dir(r))))
        m = Z["holdout"] == paths.QUARTERS.index(q)
        o = defaultdict(list)
        for lab, lot, chance, stay in zip(Z["label"][m], Z["lot"][m], Z["chance"][m], Z["stay"][m]):
            o[int(lab)].append((int(lot), float(chance), bool(stay)))
        opts[r] = o
    labels = {int(lab) for lab, lot in zip(rd.label, rd.lot) if inD[lot]}
    for o in opts.values():
        labels |= {lab for lab, ol in o.items() if any(inD[lot] for lot, _, _ in ol)}
    print(f"#{incident}: {rec['lots_count']} lots {rec['table']} -> {rec['solved']} ({q}); changed set D {len(D)} "
          f"lots; {len(labels)} labels touch it")
    for lab in sorted(labels):
        rows = np.flatnonzero(rd.label == lab)
        rr = sorted({(rd.names[rd.code[i]], round(float(rd.trust[i]), 3), bool(rd.far[i]), int(rd.lot[i]))
                     for i in rows}, key=lambda t: -t[1])
        under = int(P["lab_lot"].iloc[lab])
        print(f" label {lab} at ({P['x'].iloc[lab]:.0f}, {P['y'].iloc[lab]:.0f}) on lot "
              f"{block[under] if under >= 0 else '-'} (table {C.table[under] if under >= 0 else '-'}, in D "
              f"{bool(inD[under]) if under >= 0 else '-'}); readings {[(c, t) for c, t, f, _ in rr if not f]}; "
              f"far-side copies {[(c, t, block[l_]) for c, t, f, l_ in rr if f]}")
        for r in run_names:
            ol = sorted(opts[r].get(lab, []), key=lambda t: -t[1])[:5]
            print(f"    {r:20} options: " + str([(block[lot], str(C.table[lot]), str(codes[r][lot]), round(ch, 3),
                                                  "stay" if st else "move", "D" if inD[lot] else "-")
                                                 for lot, ch, st in ol]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("id", type=int)
    ap.add_argument("--run", action="append", help="a run whose placement to show (repeatable)")
    args = ap.parse_args()
    main(args.id, args.run)
