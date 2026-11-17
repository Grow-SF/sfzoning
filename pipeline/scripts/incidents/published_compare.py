"""Today's published 2000 map scored with the stage report's measures, beside the stage's run `final` (the solve with
the compound fusion move on) and the old `final` (before it went on: the report kept in the S-A copy
data/solver/experiments/mincut-swap/before/report.json).

The published map is the code per parcel of data/work/2000-digital-zoning/districts_raw.parquet
(`solver.published.save_published`, the run `published`).  Every measure is report.six_measures on every lot, the
column the runs of this build compare on.  Wrong lots, incidents and labels right: the published map against the new
final with criteria_check's paired block bootstrap (reused: its per_block, bootstrap and known_error).

Also counts the lots where the published code is not final's, by what the scoring table says, and saves them.

    uv run python scripts/incidents/published_compare.py

Writes data/solver/experiments/published-compare/compare.json, compare.md and differences.csv (blklot, published code,
final code, table code, quarter).  Reads only otherwise (it saves the published map as the run `published`)."""
import csv
import json

import numpy as np

import criteria_check as cc
from sfzmaps.solver import paths, runs
from sfzmaps.solver.city import city, load_run
from sfzmaps.solver.published import save_published
from sfzmaps.solver.report import six_measures

OLD_FINAL = runs.EXPERIMENTS_DIR / "mincut-swap" / "before" / "report.json"
SEARCH_COMPOUND = runs.EXPERIMENTS_DIR / "search-compound" / "report.json"
OUT = runs.EXPERIMENTS_DIR / "published-compare"
KEYS = (("lines_drawn", "Table district lines the map draws", "{:.2%}"),
        ("drawn_are_table_lines", "Lines the map draws that are table lines", "{:.2%}"),
        ("labels_right", "Zoning labels whose district comes out right", "{:.2%}"),
        ("holes", "Holes", "{:,}"), ("incidents", "Wrong-lot incidents", "{:,}"),
        ("wrong_lots", "Wrong lots", "{:,}"))
COLUMNS = ("published", "final", "final_before_compound_fusion", "search-compound")


def main():
    C = city()
    save_published()
    pub, fin = load_run("published"), load_run("final")
    nothing = np.zeros(C.n, bool)
    m_pub, m_fin = six_measures(pub, nothing), six_measures(fin, nothing)
    m_old = json.loads(OLD_FINAL.read_text())["measures"]["every_lot"]
    m_sc = json.loads(SEARCH_COMPOUND.read_text())["measures"]["every_lot"]
    _, blocks = np.unique(C.lots["blklot"].astype(str).str[:4].values, return_inverse=True)
    nb = int(blocks.max()) + 1
    p_pub, p_fin = cc.per_block(pub, blocks, nb), cc.per_block(fin, blocks, nb)
    for name, pb, m in (("published", p_pub, m_pub), ("final", p_fin, m_fin)):   # the split adds up to the measures
        assert round(pb["incidents"].sum()) == m["incidents"], name
        assert abs(pb["right"].sum() / pb["zoning"].sum() - m["labels_right"]) < 1e-5, name
        assert pb["wrong_lots"].sum() == m["wrong_lots"], name
    ci = cc.bootstrap(p_fin, p_pub, nb)                  # published - final
    err = cc.known_error(fin)
    diff = {"incidents": m_pub["incidents"] - m_fin["incidents"],
            "labels_right": m_pub["labels_right"] - m_fin["labels_right"],
            "wrong_lots": m_pub["wrong_lots"] - m_fin["wrong_lots"]}
    # the lots where published != final, by what the table says (the scored lots; the rest the table does not score)
    d = pub != fin
    sc = d & C.scored
    final_right, pub_right = sc & (fin == C.table), sc & (pub == C.table)
    counts = {"lots": int(C.n), "differ": int(d.sum()), "differ_scored": int(sc.sum()),
              "final_right": int(final_right.sum()), "published_right": int(pub_right.sum()),
              "neither_right": int((sc & ~final_right & ~pub_right).sum()), "not_scored": int((d & ~C.scored).sum()),
              "published_blank": int((d & (pub == "")).sum()), "final_blank": int((d & (fin == "")).sum())}
    assert counts["final_right"] + counts["published_right"] + counts["neither_right"] == counts["differ_scored"]
    assert counts["differ_scored"] + counts["not_scored"] == counts["differ"]
    OUT.mkdir(parents=True, exist_ok=True)
    blk = C.lots["blklot"].astype(str).values
    with open(OUT / "differences.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["blklot", "published", "final", "table", "table_scored", "quarter"])
        for i in np.flatnonzero(d):
            w.writerow([blk[i], pub[i], fin[i], C.table[i] if C.known[i] else "", int(C.scored[i]),
                        paths.QUARTERS[C.quarter[i]]])
    out = {"measure": "report.six_measures on every lot (no data errors left out)",
           "measures_every_lot": dict(zip(COLUMNS, (m_pub, m_fin, m_old, m_sc))),
           "published_minus_final": diff, "bootstrap_95": ci, "draws": cc.DRAWS, "seed": cc.SEED, "blocks": nb,
           "table_known_error": err, "differences": counts, "differences_file": "differences.csv"}
    (OUT / "compare.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    (OUT / "compare.md").write_text(markdown(out))
    print(markdown(out))


def markdown(o):
    m, ci, d, err, n = (o["measures_every_lot"], o["bootstrap_95"], o["published_minus_final"],
                        o["table_known_error"], o["differences"])
    L = ["# The published 2000 map against the solver's final", "",
         "Measures: `report.six_measures` on every lot (no data errors left out). Published: the pipeline's "
         "district reader's code per parcel (`districts_raw.parquet`). Final: the stage's run, the compound fusion "
         "move on. Old final: the stage before it was on.", "",
         "| Measure | Published | Final (new) | Final (old) | search-compound |", "|---|---|---|---|---|"]
    for k, lab, fmt in KEYS:
        L.append(f"| {lab} | " + " | ".join(fmt.format(m[r][k]) for r in COLUMNS) + " |")
    L += ["", "## Published against the new final (paired block bootstrap)", "",
          f"{o['draws']:,} draws over {o['blocks']:,} blocks (`criteria_check.py`'s: assessor blocks, the same "
          f"draw for both maps, seed {o['seed']}). Difference = published - final; a positive wrong-lot or incident "
          "difference means the published map is worse.", "",
          "| Measure | Published | Final | Difference | 95% interval | Table's known error |",
          "|---|---|---|---|---|---|",
          f"| Wrong lots | {m['published']['wrong_lots']:,} | {m['final']['wrong_lots']:,} | "
          f"{d['wrong_lots']:+,} | [{ci['wrong_lots'][0]:+,.0f}, {ci['wrong_lots'][1]:+,.0f}] | "
          f"{err['lots_corrected']:,} lots corrected |",
          f"| Incidents | {m['published']['incidents']:,} | {m['final']['incidents']:,} | {d['incidents']:+,} | "
          f"[{ci['incidents'][0]:+.1f}, {ci['incidents'][1]:+.1f}] | {err['incidents']:.0f} corrections |",
          f"| Labels right | {m['published']['labels_right']:.2%} | {m['final']['labels_right']:.2%} | "
          f"{d['labels_right'] * 100:+.2f} pts | [{ci['labels_right'][0] * 100:+.2f}, "
          f"{ci['labels_right'][1] * 100:+.2f}] pts | {err['labels_right'] * 100:.2f} pts |", "",
          f"The scoring table's known error: {err['corrections_applied']} by-eye corrections applied "
          f"({err['lots_corrected']:,} lots). They change the new final's incidents "
          f"{err['final_incidents_corrected_vs_raw'][1]:,} (raw table) to "
          f"{err['final_incidents_corrected_vs_raw'][0]:,} (corrected) and its labels right "
          f"{err['final_labels_right_corrected_vs_raw'][1]:.2%} to {err['final_labels_right_corrected_vs_raw'][0]:.2%}.",
          "", "## Lots where the published code is not final's", "",
          f"Of {n['lots']:,} parcels, {n['differ']:,} differ.", "",
          "| What the scoring table says | Lots |", "|---|---|",
          f"| Final right (published wrong) | {n['final_right']:,} |",
          f"| Published right (final wrong) | {n['published_right']:,} |",
          f"| Neither right | {n['neither_right']:,} |",
          f"| Not scored (the table gives no code, or a data error) | {n['not_scored']:,} |",
          f"| Total | {n['differ']:,} |", "",
          f"Among all that differ, {n['published_blank']:,} have no published code (blank) and {n['final_blank']:,} "
          "no final code. The per-lot table is `differences.csv` (blklot, published, final, table, table_scored, "
          "quarter)."]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
