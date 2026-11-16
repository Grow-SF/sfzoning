# Code Reader, Read Along the Word: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** the code reader finds each label's word from a reader-only detector, then reads along it with one
left-to-right model shared by codes, "other text" and "no code". This replaces the brute-force pose grid of
`codereader/posterior.py`.

**Architecture:**
- **R1:** a new stage `solver-text-reader` trains four detectors, one per held-out quarter (answered blocks held out of
  all four), and saves all three outputs (text chance, kind, reading direction) for the 2000 sheets.
- **R2:** `codereader/frames.py` turns those outputs, plus Terra's box or Sol's point, into a few weighted word frames.
- **R3:** `codereader/reading.py` sums exactly over segmentations along each frame, on one common window of page px.
- **R4:** `posterior.evidence` uses R2 and R3. The old pose grid is deleted, the training-label gate is run, and the
  full run is timed.

This revises Task 3 of `docs/superpowers/plans/2026-11-15-code-reader.md`. Its Tasks 1–2 stay as built; its Tasks 4–5
follow this plan unchanged, on R4's records.

**Tech Stack:** numpy, OpenCV, PyTorch (the detector, and the reading pass's sums on the CPU), Typer, DVC, pytest.

**Spec:** `docs/superpowers/specs/2026-11-16-code-reader-read-along-the-word-design.md`. It revises
`docs/superpowers/specs/2026-11-15-code-reader-design.md`, which stays the authority on all it doesn't change. Read
both.

## Global Constraints

- **The owner's answers are the test.**
  - Never learn from an answered label or any label in its block (`Label.held_out`, `Label.held_block`,
    `labels.refuse_held`). That holds for the detector's training (R1), every measured spread (R2), and the priors
    (R4).
  - The only exception is Task 4's three calibration constants, cross-fitted.
- **The reader never sees the table's code at a label's spot.** Code names from the table's list are allowed.
- **No brute-force sweep over angles, positions or sizes.** No hard cutoffs as decisions. A truncation is a numerical
  tolerance whose dropped mass is bounded and reported.
- **Deterministic.**
  - Seeded; `torch.use_deterministic_algorithms(True)`.
  - Scoring workers run one thread each (OMP/OPENBLAS/VECLIB/MKL/NUMEXPR_NUM_THREADS=1, `torch.set_num_threads(1)`,
    `cv2.setNumThreads(1)`).
  - The detector trains in one exclusive GPU turn (`sfzmaps.gpu.gpu_turn(True, "mps")`). Never two GPU jobs at once.
- **Machine load near 20.** Size the workers to it and check `uptime` a minute into any long run.
- **Synthetic cases** draw every number (angles, heights, spacings, offsets, line widths, noise) from measurements on
  the training labels for that situation (`tests-from-measured-distributions`), and come before any real run.
- **One measure for every comparison.** On the training labels: top-1 against the training reads' consensus, and mean
  −log p of it, box and point labels separately. The comparison is the saved records of commit c626e46.
- **No paid or LLM reads.** Bootstrap from primary sources: every input is made by code in the branch.
- **Process.**
  - Tests run on the CPU (`cd pipeline && uv run pytest -q`).
  - Commit locally only, and don't push. Stage only your own files.
  - Commit messages end with `Co-Authored-By: <the model you are> <noreply@anthropic.com>` and
    `Claude-Session: https://claude.ai/code/session_01UjTgrnx2j4PhzZErrad25Q`.

## Review Focus

1. **A label printed upside down or sideways reads right,** with no turn search: the detector's direction gives the
   frame. Test in R2 and R3: `test_upside_down_and_sideways_codes_read`.
2. **A heavy line through a letter doesn't change the code.** Line px are unobserved in every hypothesis. Test in R3:
   `test_line_through_h_does_not_make_it_m`, the existing test, rerun against the new engine.
3. **A Sol point off the word's middle by the measured spread reads the full code,** not a prefix or "other". Test in
   R3: `test_off_centre_point_reads_full_code`. Every offset passes, with no xfail.
4. **A code printed beside a street name reads the code.** The fused grouping is one weighted frame among several. Test
   in R2 and R3: `test_code_beside_street_name_reads_code`.
5. **Each label is read with a detector that never saw its quarter, and no answered block is ever trained on.** Test
   in R1: `test_reader_detector_never_trains_on_its_quarter_or_held_blocks`.

---

### Task R1: The reader's detector (stage `solver-text-reader`)

**Files:**
- Modify: `pipeline/src/sfzmaps/solver/textnet/data.py`
  - a generalised hold-out: `SheetData.held`, a tuple of map-area tests, of which today's validation area is one, so
    trials and the `solver-text` stage behave exactly as before;
  - a wider unknown region around Sol points.
- Modify: `pipeline/src/sfzmaps/solver/textnet/predict.py`: `outputs(net, ink, device) -> (chance, kind, direction)`.
- Create: `pipeline/src/sfzmaps/solver/textnet/reader_stage.py`: `build_reader_text(device=None) -> dict`.
- Modify: `pipeline/src/sfzmaps/build.py` (the stage), `pipeline/src/sfzmaps/cli.py` (`sfzmaps solver text-reader`),
  `pipeline/src/sfzmaps/solver/paths.py` (`READER_TEXT_DIR = data/solver/text-reader`), `pipeline/README.md`.
- Test: `pipeline/tests/test_solver_textnet.py`, `pipeline/tests/test_build.py`

**Interfaces:**
- **Produces** `data/solver/text-reader/`:
  - for each 2000 sheet:
    - `<sheet>/chance.png` (uint8, 0–255);
    - `<sheet>/kind.npz` (uint8 per px, the argmax kind, and float16 probabilities of the 4 kinds);
    - `<sheet>/direction.npz` (float16 cos and sin);
    - each sheet's px take the model whose held-out quarter holds them;
  - `models/<q>.pt`;
  - `held.json`: the held-out blocks, and the unknown radius with its measurement.
- **The unknown radius** around Sol points (today `READ_POINT_RADIUS = 30`): the longest printed code word among
  training Terra boxes of kind "zoning code", in page px, measured and recorded. Use it for every map.
- **Hold-outs are by map area,** through each px's map point (the way `in_validation` works):
  - quarter `q` by `grid.quarter_of`;
  - held-out blocks by the lot under the px (`data/solver/lots`), on every training map at once.
- **The recipe is `stage.DEFAULT_RECIPE`, seed 0, on `paths.TEXT_TRAIN_SERIES`.** `solver-text` and its outputs are
  untouched.

- [ ] **Step 1: Failing tests.**
  - `test_reader_detector_never_trains_on_its_quarter_or_held_blocks`: a fit's drawn crops never centre in, and its
    loss weight is 0 on, quarter q's area and on held-out blocks' lots, on two synthetic maps.
  - `test_unknown_around_sol_points_covers_the_word`: px within the radius of a point are unknown, and px beyond it
    are not.
  - `test_outputs_returns_chance_kind_and_direction`: shapes; probabilities sum to 1; |(cos, sin)| > 0.
  - `test_reader_stage_deps` (test_build): the stage reads the training maps' warps, prints, lots and the code-check
    answers, and its code deps are its import closure.
- [ ] **Step 2: Implement.** The tests pass, and the full suite passes.
- [ ] **Step 3: Real run.** `uv run sfzmaps solver text-reader`, in one exclusive GPU turn: about 4 × 7 min of
  training plus prediction. Report:
  - the minutes;
  - the unknown radius;
  - for each quarter model, its words found on its own held-out quarter (the `scores` measure);
  - a picture of one sheet's direction field over a dense area (say what it shows).

  Then run it twice and check the outputs are identical.
- [ ] **Step 4: Commit.**

### Task R2: Word frames

**Files:**
- Create: `pipeline/src/sfzmaps/solver/codereader/frames.py`
- Test: `pipeline/tests/test_codereader.py`

**Interfaces:**
- **Consumes:** R1's outputs, through `frames.sheet_outputs(sheet) -> Outputs` (chance, kind, direction), and
  `labels.Label` (box or point, `crop`, `to_page`).
- **Produces:**
  - `Frame`: θ (reading direction, radians, full 360°), h (letter height, page px), c (offset across the line), s0 and
    s1 (start and end along the line, page px), and `log_w` (the frame's prior weight from the evidence).
  - `Spreads`: measured spreads.
  - `measure_spreads(training_labels, fits) -> Spreads`. It is measured against Task 2's alignment poses for the
    training labels (`glyphs.learn_font` fits): θ, h, c, s0, s1, the box-angle error and the half-turn rate, with
    point and box labels separately. It refuses held-out labels and blocks.
  - `frames(label, outputs, spreads, n_max) -> list[Frame]`:
    - group text px near the label along the detector's direction;
    - alternative groupings (a fused neighbour word, a word split by a line) become extra frames with their own
      weights;
    - Terra's box enters as evidence on θ and the extent, weighted by its measured spread;
    - a weak direction falls back toward the box angle by those weights.

- [ ] **Step 1: Failing tests.** All use synthetic labels and detector outputs, drawn from the spreads measured in
  Step 3 (until then, from a fixture of the same shape):
  - `test_frames_recover_the_pose`: upright, sideways, upside down, slanted. The best frame is within the measured
    spread of the true pose.
  - `test_upside_down_and_sideways_codes_read` (frames half): the true direction carries most of the weight.
  - `test_lone_p_falls_back_to_the_box`.
  - `test_code_beside_street_name_reads_code` (frames half): the code-only grouping is among the frames, with weight
    > 0.
  - `test_measure_spreads_refuses_held_labels`.
- [ ] **Step 2: Implement.** The tests pass.
- [ ] **Step 3: Measure the spreads on the real training labels.** Report each spread, box and point labels
  separately, and a picture of 12 training labels with their frames drawn (say what it shows). Replace the fixture
  with the measured spreads.
- [ ] **Step 4: Full suite. Commit.**

### Task R3: The reading pass

**Files:**
- Create: `pipeline/src/sfzmaps/solver/codereader/reading.py`
- Test: `pipeline/tests/test_codereader.py`

**Interfaces:**
- **Consumes:** `Frame` (R2); `glyphs.Font` and `glyphs.NoiseModel` (Task 2); the line mask (`labels.line_mask`,
  rules `posterior.RULES`).
- **Produces:**
  - `Window`: the label's common window of page px. It is the union of its frames' extents plus a margin of one
    letter height; it holds its observed mask and its ink.
  - `window(label, frames) -> Window`.
  - `read(window, frames, font, noise, strings, other_bigram, rho, temperatures) -> Reading`. Its fields are:
    - `log_e` (dict string → array over temperatures, for every printed form);
    - `other` (array over temperatures);
    - `none` (array over temperatures; 0 by definition);
    - `seconds`.
  - The sum, for each frame, string or the "other" bigram:
    - a semi-Markov forward pass along the frame's columns;
    - each character's advance and the gap after it have measured distributions (R2's fits);
    - emissions come from the letter shapes under the noise model, on the window's page px;
    - px outside the reading's extent are explained by background or by "other text nearby" at density `rho`
      (`posterior.foreign_density`, kept);
    - line px are unobserved in every hypothesis.
  - Frames are summed with weight `log_w`. Codes, "other" and "none" share the same widths, gaps, noise and px.
  - **Temperatures:** each frame's and segmentation's log-likelihood is computed once and summed per temperature.

- [ ] **Step 1: Failing tests.** All synthetic, with numbers from the measured spreads:
  - `test_line_through_h_does_not_make_it_m`;
  - `test_upside_down_and_sideways_codes_read`;
  - `test_off_centre_point_reads_full_code`: offsets −1.0 to +1.0 word heights, both sides; every offset reads the
    full code with p > 0.5;
  - `test_prefix_pays_for_the_ink_it_leaves` (RH-1 inside RH-1(D) loses);
  - `test_widely_spaced_code_reads_that_code` (the c626e46 strict xfail now passes; remove the xfail);
  - `test_crisp_code_on_clean_paper_reads_that_code_not_other` (kept from c626e46);
  - `test_code_beside_street_name_reads_code`;
  - `test_lines_only_window_prefers_none`;
  - `test_temperature_grid_gives_each_temperature_as_alone`: within 1e-9;
  - `test_posterior_sums_to_one`.
- [ ] **Step 2: Implement.** The tests pass.
- [ ] **Step 3:** Time one box label and one point label. Report the seconds.
- [ ] **Step 4: Full suite. Commit.**

### Task R4: The reader on the new engine, the gate and the full run

**Files:**
- Modify: `pipeline/src/sfzmaps/solver/codereader/posterior.py`
  - `evidence` builds frames (R2), a window, and reads (R3);
  - `Prior` keeps the code prior, the form weights, the bigram, ρ and the masses, and gains the measured kind
    likelihood;
  - delete the pose grid, the 2-D placement prior, the end-break term and `PosePrior`;
  - `Evidence` keeps its fields.
- Create: `pipeline/src/sfzmaps/solver/codereader/run.py`:
  - `score_all(labels, workers, temperatures_held, temperature_rest) -> Path`: one-thread workers;
  - deterministic records in `data/solver/experiments/codes-records/<commit>/`.
- Modify: `pipeline/src/sfzmaps/cli.py`: `sfzmaps solver code-reader-score [--workers N]`.
- Test: `pipeline/tests/test_codereader.py`

**Interfaces:**
- **Consumes:** R2 and R3.
- **Produces:**
  - `posterior.evidence(label, font, noise, prior, temperatures, *, outputs) -> Evidence`, same fields as today;
  - the records Task 4 reads: the held-out labels at T = 16, 23, 32.7, 46; the rest at the prior's own T;
  - **the kind likelihood:** p(detector kind = zoning code | a code is printed), and the same under "other text",
    measured on training labels from R1's out-of-quarter outputs.

- [ ] **Step 1: Failing tests.**
  - `test_evidence_uses_frames_not_a_pose_grid`: monkeypatch `frames.frames` to return one frame; evidence reads only
    that frame's window.
  - `test_kind_evidence_moves_code_against_other`.
  - `test_score_all_is_deterministic`: two runs of three synthetic labels give identical records.
- [ ] **Step 2: Implement.** The tests pass. The old c626e46 tests that exercised the pose grid are removed or ported
  to the new engine; list each one in the report.
- [ ] **Step 3: The gate, on training labels only,** against the saved c626e46 records:
  - box labels' top-1 and mean −log p no worse;
  - point labels' top-1 and mean −log p no worse.

  If it fails, report it and stop.
- [ ] **Step 4: The full run:** `uv run sfzmaps solver code-reader-score`, load near 20. Report:
  - minutes;
  - worker-seconds by label kind;
  - the top reading counts (codes / other / none);
  - point- and box-label agreement on training labels.
- [ ] **Step 5: Full suite. Commit.**

Then the original plan's Task 4 (evaluation on the owner's answers) and Task 5 (the stage and the solver) follow on
R4's records. Task 5's stage also depends on `solver-text-reader`.
