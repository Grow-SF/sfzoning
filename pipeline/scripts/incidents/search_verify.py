"""Are the census's lower-energy answers real?  An adversarial check from scratch.

    uv run python scripts/incidents/search_verify.py

Per quarter: the energy is rebuilt from its held-out inputs (search_lab.problem: the stage's walls, the readings,
the run's placement, as solve_quarter builds it, in float64), not from the saved costs.  Checks that it gives the
saved U (float32) and the solve's logged energy, and that the stage's answer is the saved one.  Then, for every census
incident (incidents.json), the whole city's energy (energy.energy over every lot and pair) of the stage's answer
with the incident's cheapest answer right on it (answers/cheapest-right-<id>.npz) put in, against the stage's
answer: the change, and the wrong scored lots of the quarter before and after.

The search oracle: per quarter, every answer that lowers the energy put in, lowest first, each kept only if it still
lowers the energy on top of those already in; joined (each lot from its quarter's solve) and scored with the six
measures on every lot.  It writes runs into data/solver/experiments/incidents-search-oracle/ and
incidents-search-verify.json there.  The oracle uses the table (the answers were searched from the table's
answer): a ceiling, never an input."""
import json
import re

import numpy as np

import search_lab as L
from common import CENSUS_DIR
from sfzmaps.solver import paths, runs
from sfzmaps.solver.city import city, run_file


def logged_energy(q: str) -> float:
    text = (runs.SOLVE_DIR / "runs" / f"final-{q}.log").read_text()
    return float(re.search(r"solved ([\d,\.]+)", text).group(1).replace(",", ""))


def main() -> None:
    C = city()
    doc = json.loads((CENSUS_DIR / "incidents.json").read_text())
    out = {"quarters": {}, "incidents": []}
    name = "oracle"
    for k, q in enumerate(paths.QUARTERS):
        P = L.problem(q)
        saved = np.load(run_file("final", f"-{q}-costs", runs.SOLVE_DIR))
        e_final = P.energy(P.final)
        names = np.array(P.names)
        inq = C.quarter == k
        wrong = C.scored & inq
        out["quarters"][q] = {
            "max_abs_U_vs_saved": float(np.abs(P.U - saved["U"].astype(float)).max()),
            "odds_equal_saved": bool(np.array_equal(P.odds.astype(np.float32), saved["odds"])),
            "answer_equals_saved_codes": bool(np.array_equal(P.final, saved["codes"])),
            "energy_final": round(e_final, 4), "logged": logged_energy(q),
            "wrong_scored_lots_in_quarter": int((wrong & (names[P.final] != C.table)).sum())}
        print(q, out["quarters"][q], flush=True)
        cands = []
        for rec in doc["list"]:
            if rec["quarter"] != q:
                continue
            z = np.load(CENSUS_DIR / "answers" / f"cheapest-right-{rec['id']}.npz")
            lots, codes = z["lots"], P.to_index(z["codes"].astype(str))
            y = P.final.copy()
            y[lots] = codes
            dE = P.energy(y) - e_final
            w0 = int((wrong & (names[P.final] != C.table)).sum())
            w1 = int((wrong & (names[y] != C.table)).sum())
            census = rec["groups"]["cheapest_right_on_incident"]
            row = {"id": rec["id"], "quarter": q, "lots": rec["lots_count"], "table": rec["table"],
                   "solved": rec["solved"], "lots_changed": int(len(lots)),
                   "dE_full_city": round(dE, 4), "dE_census": round(sum(census.values()), 4),
                   "census_min": rec["solved_is_minimum"],
                   "wrong_in_quarter_before": w0, "wrong_in_quarter_after": w1}
            out["incidents"].append(row)
            print(row, flush=True)
            if dE < -1e-6:
                cands.append((dE, rec["id"], lots, codes))
        # the oracle: lower answers put in, lowest first, each only while it still lowers the energy
        x = P.final.copy()
        kept = []
        for dE, i, lots, codes in sorted(cands, key=lambda c: c[0]):
            y = x.copy()
            y[lots] = codes
            d = P.energy(y) - P.energy(x)
            if d < -1e-6:
                x, _ = y, kept.append((i, round(d, 4)))
        out["quarters"][q].update(oracle_kept=kept, oracle_energy=round(P.energy(x), 4),
                                  oracle_dE=round(P.energy(x) - e_final, 4),
                                  oracle_wrong_in_quarter=int((wrong & (names[x] != C.table)).sum()))
        L.save_quarter(name, P, x)
        print(q, "oracle", kept, out["quarters"][q]["oracle_dE"], flush=True)
    out["oracle_measures"] = L.join_and_score(name)
    out["final_measures"] = L.final_measures()
    folder = L.run_folder(name)
    (folder / "incidents-search-verify.json").write_text(json.dumps(out, indent=1))
    print(L.measures_table({"final": out["final_measures"], "search oracle": out["oracle_measures"]}))


if __name__ == "__main__":
    main()
