# Brief: task T4 of docs/superpowers/plans/2026-11-16-text-reader-labels-and-publish.md

### T4: The solve's labels from boxes and the text reader

- **What:**
  - Boxes where a code has real mass become the solve's prints.
  - A calibration replaces the trust fit. Same target, rows and quarter rule as `trust.py`. Features: closed-list
    mass, margin, p(other), p(none), box height, the detector's p(zoning code), on a street.
  - It writes `readings_{q}.npz` in today's format.
  - Street sides and placement refit as they do today.
  - Fix the `leader_moves` quirk: it matches on the read code, not `fix_code`, in 21 of 101 rows.
  - **`TOP_READING_FIT`** (`energy.py:12,49`) maps a label's top trust through constants fitted to the LLM chain's
    trust. The new calibration fills the same slot. Refit it, held out by quarter, on the new readings, or set it to
    identity on purpose because the calibration is already a probability. Don't leave the old mapping in silently.
  - **Readings per label.** Today a print has about 1.24 readings. The tie-break's path search (`energy.py:66-75`)
    starts from every lot with any reading of a code, and the leftover mass per label (`41-52`) depends on how many
    codes are written. Write only the codes with real mass, and report the number per label. Keep the `names` list,
    which decides the codes the solve can output.
- **Measure:**
  - the six measures against `final`, with intervals (expected somewhat worse; "better than published" is the bar);
  - the calibration's held-out log loss;
  - `top_reading_calibration`.


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

