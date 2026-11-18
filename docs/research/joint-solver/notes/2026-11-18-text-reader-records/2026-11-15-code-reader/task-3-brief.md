### Task 3: The posterior

**Files:**
- Create: `pipeline/src/sfzmaps/solver/codereader/posterior.py`
- Test: `pipeline/tests/test_codereader.py`

**Interfaces:**
- **Consumes:** `Font`, `NoiseModel`, `Label`, `line_mask`, `code_list`.
- **Produces:**
  - **`posterior.score_label(label, font, noise, prior, constants) -> dict[str, float]`:** the posterior over the
    codes plus `"none"` and `"other"`.
    - The log-likelihood of each hypothesis sums over a pose grid: position with a Gaussian prior around the print's
      point (σ of a few letter heights), size from the measured spread, angle from the box (or upright/sideways).
      Use logsumexp, not max.
    - `"none"`: the ink explained by background only, with line pixels unobserved.
    - `"other"`: an open-vocabulary pass, a character bigram over the font's characters.
  - **`posterior.code_prior(training_labels) -> dict`:** counts of printed codes from training reads, smoothed.
  - **`posterior.Constants`:** temperature, none mass, other mass.

- [ ] **Step 1: Failing tests:**
  - `test_line_through_h_does_not_make_it_m` (Review Focus 2);
  - `test_sideways_code_reads` (Review Focus 3);
  - `test_lines_only_window_prefers_none` (Review Focus 4);
  - `test_posterior_sums_to_one`.
- [ ] **Step 2:** Implement; the tests pass. Time it: every label on the CPU, and in one GPU turn if faster.
- [ ] **Step 3:** Full suite. Commit.

