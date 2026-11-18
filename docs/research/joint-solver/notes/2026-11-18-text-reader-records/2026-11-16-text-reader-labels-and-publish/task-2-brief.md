# Brief: task T2 of docs/superpowers/plans/2026-11-16-text-reader-labels-and-publish.md

### T2: Word boxes from our detector (`solver-boxes`)

- **What:** spike S3's plain grouping, run on the per-quarter text chance of every 2000 sheet. Each box records:
  - corners, axis (mod 180°), length, height;
  - mean chance and kind probabilities;
  - quarter;
  - map `x`/`y`, `lab_lot`, `street` (through `prints._place` / `under`).
- **The level:** chosen per quarter on the other quarters, by the solve's measures if cheap, else by code-word
  recall.
- **Measure:** per quarter, the recall of Terra's zoning-code words (IoU 0.3) and the share with axis within 10°;
  how many of the owner's answered labels get a box.


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

