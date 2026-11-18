# Brief: task T1 of docs/superpowers/plans/2026-11-16-text-reader-labels-and-publish.md

### T1: The walls without Terra's boxes, from the honest per-quarter text chance

- **What:**
  - The walls' letter share uses the existing `text_weighted` option (the detector's chance in place of Terra's
    share). The README's held-out log loss shows a tie: .0982/.1026/.0452/.0612 against
    .0978/.1027/.0452/.0611.
  - `textgrid` reads the per-quarter chance from `solver-text-reader`. Each px then comes from the detector that never
    saw its quarter; today it comes from `solver-text`, which trained on these very sheets.
  - Drop `terra_boxes.json` from `solver-walls`' reads.
- **Measure:** the walls' held-out log loss, then the six measures.


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

