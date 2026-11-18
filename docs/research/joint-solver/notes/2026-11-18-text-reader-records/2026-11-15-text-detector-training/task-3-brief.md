### Task 3: Scores on the validation area and the test drawings

**Files:**
- Create: `pipeline/src/sfzmaps/solver/textnet/scores.py`, replacing `evaluate.py`'s scoring. `evaluate.py` goes,
  along with its callers.
- Test: `pipeline/tests/test_solver_textnet.py`

**Interfaces:**
- **Consumes:** Task 1's data, Task 2's `predict` (chance per sheet), and `sfzmaps.solver.strokes` (the skeleton
  strokes, reused for the long-line measure).
- **Produces:**
  - `score_sheets(chances: dict[(series, sheet), np.ndarray], where: "validation" | "whole") -> dict`, per series
    and overall: AUC, log-loss, text ink found, words found by kind, ink outside boxes called text, long-line ink
    called text.
  - `LONG_LINE_FT = 200.0`, `THIN_LINE_FT = 12.0`, converted per sheet with `ft_per_px`.

- [ ] **Step 1: Write failing tests:**
  - `test_long_line_measure_uses_feet_per_sheet` (Review Focus 2): the same drawing at 2 scales gives the same
    lines.
  - `test_validation_scores_use_only_validation_pixels`.
  - `test_words_found_counts_a_box_with_half_its_ink_called_text`: the rule today's evaluate uses, kept.
- [ ] **Step 2:** Implement; the tests pass. Full suite. Commit.

