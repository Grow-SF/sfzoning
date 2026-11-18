# Brief: task T0 of docs/superpowers/plans/2026-11-16-text-reader-labels-and-publish.md

### T0: Fix the scoring labels before the label source changes

- **Why:** "labels right" is computed from today's LLM readings (`measures.Table`, `report.top_reading_calibration`).
  If the solve's labels change, the score's label set must not change with them.
- **What:**
  - The 1,881 scoring labels stay those of today's chain.
  - The solve's own labels get their own location: `trust.load_prints`, `labels.py`, `streetside.py` and
    `placement.py` read from it.
  - `prints.parquet` stays as is: `solver-text-reader` uses it to define its held-out blocks.
- **Also (the previous plan's final review):**
  - `criteria_check.label_lots` repeats the label source in `measures.Table`. Make both read the one fixed scoring
    set.
  - The per-block totals check is weak (it can't catch a regrouping). Strengthen it.
  - The energy criterion ("energy lower in every quarter") means nothing once walls or labels change. Say so in
    `criteria_check`, and leave it out of comparisons across label sources.
- **Measure:** re-solving `final` gives a byte-identical `report.json`.


## Global constraints

- **No LLM output at inference.** Terra's and Sol's saved answers may train and score. The owner's answers and
  `label_fixes.csv` may train and score.
- **Apache-2.0 code.** No new copyleft dependency. python-doctr cannot be a dependency (its opencv conflicts), so
  the text reader is re-expressed in plain torch. If torchvision is needed, it is BSD.
- **Held out by quarter,** as everything in the solver is. The answered blocks are never trained on.
- **One measure when comparing:** the solve's six measures, every-lot column, with the paired block bootstrap
  (`scripts/incidents/criteria_check.py`), against today's `final`.
- **Better is enough:** each step is kept when it is measurably better, or no worse within noise, on the agreed
  measures. No step chases every miss.
- **Load near 20; one GPU job at a time; deterministic.** Commit locally; the owner merges and ships.

