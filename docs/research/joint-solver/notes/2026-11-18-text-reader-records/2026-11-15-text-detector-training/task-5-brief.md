### Task 5: The measured loop and the chosen recipe

**Files:**
- Modify: `pipeline/src/sfzmaps/solver/textnet/stage.py` (`DEFAULT_RECIPE`), `pipeline/README.md` (the results
  table)
- Scratch: the per-run results and the minutes stay in `data/solver/experiments/` and the report.

The GPU runs one job at a time. Run each `text-trial` in the foreground or wait on it, never two at once.

- [ ] **Step 1: Baseline.** `text-trial` with the default recipe (today's). Record the minutes and the validation
  scores.
- [ ] **Step 2: The big levers first.** These make every later run cheaper:
  - `half_size=True`;
  - `ink_only=ink` and `ink_only=ink+1px`;
  - then the best of those together.

  Keep each by the keep rule against the current best.
- [ ] **Step 3: The tolerance.** Two seeds of the current fast recipe give the spread; tolerance = max(2 × spread,
  0.5 points on words found). Re-judge Step 2's decisions with it if any were within it.
- [ ] **Step 4: The schedule.**
  - `iterations` ∈ {2000, 4000, 8000} × `max_lr` ∈ {2e-3, 6e-3}, with `pct_start=0.1`;
  - then `weight_average=0.999` on the best.

  Keep by the rule, preferring the shortest that passes.
- [ ] **Step 5: The rest.** In order: `heads="text"`, then `prefetch=True` (it must give identical scores; it's
  speed only), then `decoder="light"` only if the clock isn't met.
- [ ] **Step 6: The mixed-style check.** Train the chosen recipe on the modern-lettering maps only and compare it
  with the both-styles model on the 1980 test drawing. If both-styles is worse beyond the tolerance, report it and
  stop before Step 7.
- [ ] **Step 7: Lock it in.**
  - Set `DEFAULT_RECIPE` to the chosen recipe.
  - Run `uv run sfzmaps solver text` twice: the chances agree within 0.01 and the minutes are within the clock.
  - Run `uv run sfzmaps solver text-scores`.
  - Write the README table: each step's minutes and validation scores, the test scores, and today's quarter models
    for context.
  - Full suite. Commit.
