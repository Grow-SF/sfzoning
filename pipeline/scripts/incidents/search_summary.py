"""The search experiments side by side: the six measures on every lot, the energy per quarter, and per census search
incident the share of its lots each run gets right.

    uv run python scripts/incidents/search_summary.py [VARIANT ...]

Reads incidents-search-<variant>/measures.json (search_experiments.py --score) and the runs; writes
data/solver/experiments/incidents-search-summary.md and prints it."""
import json
import sys

import numpy as np

import search_lab as L
from common import CENSUS_DIR
from sfzmaps.solver.city import city, load_run

SEARCH_INCIDENTS = (1, 7, 34, 4, 23, 9, 44, 18, 14, 21, 12, 13, 15, 42, 22)   # 22: walls, in #1's changed set
VARIANTS = ("oracle", "qpbo", "regions", "fusion", "compound", "compound-qpbo", "starts", "anneal", "all")


def main(variants):
    C = city()
    doc = json.loads((CENSUS_DIR / "incidents.json").read_text())
    recs = {r["id"]: r for r in doc["list"]}
    cols, energies, right = {"final": L.final_measures()}, {}, {}
    final = load_run("final")
    right["final"] = {i: float(np.mean(final[recs[i]["lots"]] == C.table[recs[i]["lots"]])) for i in SEARCH_INCIDENTS}
    for v in variants:
        folder = L.run_folder(v)
        if v == "oracle":
            f = folder / "incidents-search-verify.json"
            if not f.is_file():
                continue
            d = json.loads(f.read_text())
            cols[v] = d["oracle_measures"]
            energies[v] = {q: d["quarters"][q]["oracle_dE"] for q in d["quarters"]}
        else:
            f = folder / "measures.json"
            if not f.is_file():
                continue
            d = json.loads(f.read_text())
            cols[v] = d["measures_every_lot"]
            energies[v] = {q: e["dE"] for q, e in d["energies"].items()}
        codes = load_run(L.SEARCH_PREFIX + v, folder)
        right[v] = {i: float(np.mean(codes[recs[i]["lots"]] == C.table[recs[i]["lots"]])) for i in SEARCH_INCIDENTS}
    md = "# Search experiments: the six measures on every lot\n\n" + L.measures_table(cols)
    md += "\n## Energy change against final, per quarter\n\n| Run | nw | ne | sw | se |\n|---|---|---|---|---|\n"
    for v, e in energies.items():
        md += f"| {v} | " + " | ".join(f"{e.get(q, float('nan')):+.1f}" for q in ("nw", "ne", "sw", "se")) + " |\n"
    md += ("\n## Share of each search incident's lots right\n\n| # | Lots | Table → final | Q | "
           + " | ".join(right) + " |\n|---|---|---|---|" + "---|" * len(right) + "\n")
    for i in SEARCH_INCIDENTS:
        r = recs[i]
        md += (f"| {i} | {r['lots_count']} | {r['table']} → {r['solved']} | {r['quarter']} | "
               + " | ".join(f"{right[v][i]:.0%}" for v in right) + " |\n")
    (L.runs.EXPERIMENTS_DIR / "incidents-search-summary.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main(sys.argv[1:] or VARIANTS)
