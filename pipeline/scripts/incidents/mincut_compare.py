"""The solve before and after the min-cut swap (solver/mincut.py): the six measures, and per quarter the lots whose
code changed with the energy of both answers (a tie: equal within the cut's scaling bound, which is ~6e-4).

Before: data/solver/experiments/mincut-swap/before/ (a copy of the stage's report and runs made before the rebuild);
after: the stage's folder, data/solver/solve/.

    uv run python scripts/incidents/mincut_compare.py"""
import json

import numpy as np

import search_lab as L
from sfzmaps import config

BEFORE = config.DATA_DIR / "solver" / "experiments" / "mincut-swap" / "before"
AFTER = config.DATA_DIR / "solver" / "solve"


def main():
    for name, d in (("before", BEFORE), ("after", AFTER)):
        r = json.loads((d / "report.json").read_text())
        print(name, json.dumps(r.get("six_measures", r.get("measures", None)), sort_keys=True)[:1500])
    for q in ("nw", "ne", "sw", "se"):
        P = L.problem(q)                                   # its `final` is the stage's current (after) answer
        B = np.load(BEFORE / "runs" / f"final-{q}.npz", allow_pickle=True)
        names = np.asarray(B["names"]).astype(str)[B["pred"]]
        lookup = {n: i for i, n in enumerate(P.names)}
        before = np.array([lookup[L.canon(n)] if L.canon(n) in lookup else lookup[n] for n in names])
        after = P.final
        diff = np.nonzero(before != after)[0]
        eb, ea = P.energy(before), P.energy(after)
        print(f"{q}: {len(before)} lots, {len(diff)} changed; energy before {eb:.6f} after {ea:.6f} "
              f"(after - before {ea - eb:+.6f}); "
              + ("same energy within the bound: a tie" if abs(ea - eb) < 6e-4 else "ENERGIES DIFFER"))
        for i in diff[:15]:
            print(f"   lot {i}: {P.names[before[i]]} -> {P.names[after[i]]}")


if __name__ == "__main__":
    main()
