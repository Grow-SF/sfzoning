# Brief: task T3 of docs/superpowers/plans/2026-11-16-text-reader-labels-and-publish.md

### T3: The text reader as a stage (`solver-text-read`), in plain torch

- **What:**
  - **The model:** MobileNetV3-small features, a 2-layer BiLSTM 128 and CTC, written in plain torch (~150 lines).
    The parameter names stay the spike's, so the spike's weights load as a regression check.
  - **Four readers,** one per held-out quarter, with the answered blocks held out of all four. The same for the 0°/180°
    classifier. About 90 min of GPU in all.
  - **Inputs** the synthetic crops need: the template reader's font, code forms and prior. They are made by a
    stage, not copied from scratch.
  - **Reading:** each box is read by its quarter's reader, both ways along its axis. The output is a posterior over
    the code list's printed forms (exact CTC forward), plus "other" and "none".
- **Measure:**
  - top-1, top-3 and −log p on the owner's answers matched to boxes, box-found and box-missed apart, against the
    spike (0.956–0.973 closed on Terra's boxes) and the LLM chain (96.9%);
  - the same reader on Terra's boxes, so that box error and reader error show separately.


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

