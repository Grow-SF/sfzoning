# Zoning-Code Reader Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A probabilistic reader of the 2000 map's printed zoning codes. For each label it gives
p(code | ink) ∝ p(ink | code) p(code), from glyph templates learned on the page. It is judged on the owner's answers,
then in the solver.

**Architecture:**
- `sfzmaps.solver.codereader` learns glyph templates and a noise model from training labels' crops (hard EM).
- It scores every code (and "no code" / "other text") over a pose grid, with long-line pixels unobserved, and gives
  an exact posterior.
- A stage writes the posteriors. The trust fit uses them as features.

**Tech Stack:** numpy, OpenCV, PyTorch (convolutions for the pose search; CPU or one exclusive GPU turn), Typer, DVC,
pytest.

**Spec:** `docs/superpowers/specs/2026-11-15-code-reader-design.md` (read it, including "Settled after the builder
check"; it is the authority).

## Global Constraints

- **The owner's answers are the test.** `pipeline/inputs/code-check-2000/`: 717 answered codes, 20 "no label",
  593 unflagged as the fair sample.
  - Never train on them, nor on any label in the same block as an answered label.
  - The only exception: the three calibration constants (temperature, "no code" mass, "other" mass), fitted by
    two-fold cross-fitting on the answers.
- **The reader never sees the table's code at a label's spot.** Code names from the table's list are allowed.
- **No hard cutoffs as decisions:** the pose prior is a distribution, and line pixels are unobserved, not deleted.
- **Deterministic:** seeded, `torch.use_deterministic_algorithms(True)`, forward-only pose search on mps, and one
  exclusive GPU turn (`sfzmaps.gpu.gpu_turn(True, device)`) if the GPU is used. Never run two GPU jobs. CPU is fine.
- **Fresh clone only:** no paid reads, no `pipeline/data/rnd`, no scratch inputs, no per-item stamps.
- **The same measures for every comparison:**
  - top-1 and top-3;
  - mean −log p(true);
  - a reliability table (10 bins);
  - the hard subset: box short side < 23 px, or sideways/slanted, or line-touching (ink on the box edge > 0.2).

  Reported for the reader, for each existing reader (Terra, Sol, the red-circle reads, the third read and re-check,
  the turned pass) and for J7's trust, on the same labels.
- **Process:** tests on CPU (`cd pipeline && uv run pytest -q`). Commits end with a `Co-Authored-By:` line naming
  the model you are, then `Claude-Session: https://claude.ai/code/session_01UjTgrnx2j4PhzZErrad25Q`. Stage only your
  own files.
- **Recon to reuse (scratch, read-only):**
  `/private/tmp/claude-501/-Users-sbuss-workspace-sfzoning/eb18d2fc-e22a-48a6-afbc-57c31cd3e134/scratchpad/codereader/`.
  It has the join of the answers to prints (`join.py`, `cases.pkl`), every reader's accuracy (`score.py`), the
  confusions, and `misread_labels.png`. The code goes into the package; the scripts stay scratch.

## Review Focus

1. **Test-set leakage through blocks or calibration.** Test: the training label set excludes every answered label
   and every label in an answered label's block, and the calibration fit never sees the half it scores.
2. **Line pixels.** A label crossed by a heavy line must score its true code no worse than the same label without
   the line. Test on a synthetic H crossed by a bar.
3. **Rotated and sideways labels.** The pose grid covers the box angle and its quarter turns. Test: a code drawn
   sideways reads right.
4. **"No code here".** A window of only lines and background prefers "no code". Test, plus the 20 "no label"
   answers reported.
5. **Aliases.** A printed variant ("NOE VALLEY") maps to its code, not to "other". Test.

---

### Task 1: The label set, the code list and the line masks

**Files:**
- Create: `pipeline/src/sfzmaps/solver/codereader/__init__.py`, `labels.py`
- Test: `pipeline/tests/test_codereader.py`

**Interfaces:**
- **Produces:**
  - **`labels.Label`:** sheet, page point, Terra box (if any), angle, crop (page px, straightened by the box angle
    with a margin), the LLM read strings, `answer` (the owner's code or None), `fold` (0/1 for cross-fitting),
    `block`.
  - **`labels.load_labels() -> list[Label]`,** from `prints.parquet` and the code-check answers. The answers join by
    the recon's rule (a case point inside a print's Terra box, or equal to a kind-2 point).
  - **`labels.training(labels) -> list[Label]`** excludes the answered labels and their blocks.
  - **`labels.code_list() -> dict[str, str]`:** printed form → code. It holds the table's code names plus the
    printed variants seen in training reads.
  - **`labels.line_mask(label) -> bool array`** over the crop: the detector's long-line masks
    (`textnet.scores.line_ink`), dilated by 1 px. Also report the coverage of line ink in label windows.

- [ ] **Step 1: Failing tests:**
  - `test_training_excludes_answered_labels_and_their_blocks` (Review Focus 1);
  - `test_alias_maps_to_its_code` (Review Focus 5);
  - `test_crop_is_straightened_by_the_box_angle`.
- [ ] **Step 2:** Implement; the tests pass.
- [ ] **Step 3:** Real run, CPU. Report:
  - the label counts (all, answered, training);
  - the code list size;
  - how much ink near labels the masks cover. If they miss lines broken by junctions, add a width-based line rule
    (spec); choose by Task 3's training likelihood.
- [ ] **Step 4:** Full suite. Commit.

### Task 2: Glyph templates and the noise model

**Files:**
- Create: `pipeline/src/sfzmaps/solver/codereader/glyphs.py`
- Test: `pipeline/tests/test_codereader.py`

**Interfaces:**
- **Consumes:** Task 1's training labels and crops.
- **Produces:**
  - **`glyphs.Font`:** one template per character (a probability of ink per template pixel), a character width, an
    advance, and a size.
  - **`glyphs.learn_font(labels, iterations) -> Font`:** hard EM. Lay each training string out with the current
    templates, align by the best pose, then re-estimate the templates from the aligned crops with line pixels
    unobserved. Initialise from a plain sans rendering, and say which.
  - **`glyphs.NoiseModel`** with **`fit(...)`** (flip rates by distance from the stroke edge) and
    **`log_lik(rendered, crop, observed) -> float`**.
  - **`glyphs.render(font, text, pose) -> array`.**

- [ ] **Step 1: Failing tests:**
  - `test_em_recovers_templates_from_synthetic_text` (synthetic crops drawn from known templates plus noise);
  - `test_unobserved_pixels_change_nothing`.
- [ ] **Step 2:** Implement; the tests pass. Real run on the training labels:
  - report the training log-likelihood per EM round;
  - write a picture of the learned 2000 glyphs (scratch, say what it shows);
  - try adding block numbers and street names as training text (spec); keep them if the held-back training
    likelihood improves.
- [ ] **Step 3:** Full suite. Commit.

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

### Task 4: Evaluation on the owner's answers

**Files:**
- Create: `pipeline/src/sfzmaps/solver/codereader/evaluate.py`
- Modify: `pipeline/src/sfzmaps/cli.py` (`sfzmaps solver code-reader-eval`, writing to
  `data/solver/experiments/codes-eval/`)
- Test: `pipeline/tests/test_codereader.py`

**Interfaces:**
- **Produces:** **`evaluate.cross_fitted(labels, scores_by_constants) -> report`:**
  - fit the three constants on fold 0 and score fold 1, then swap;
  - the measures of Global Constraints, for the reader and every existing reader and J7's trust on the same labels;
  - the fair (unflagged) subset, the hard subset, the 20 "no label" answers and the 6 out-of-list answers,
    separately.

- [ ] **Step 1: Failing tests:**
  - `test_calibration_never_sees_the_half_it_scores` (Review Focus 1);
  - `test_report_scores_every_reader_on_the_same_labels`.
- [ ] **Step 2:** Implement; the tests pass. Run it. Report the table and a picture of the reader's misses (scratch).
- [ ] **Step 3:** Full suite. Commit.

### Task 5: The stage and the solver

**Files:**
- Modify: `pipeline/src/sfzmaps/build.py`: a stage `solver-codes` writing `data/solver/codes/` (font, noise, a
  posterior per print). It reads the pages, prints, the saved answers and the line-mask inputs; upstream
  solver-prints.
- Modify: `pipeline/src/sfzmaps/solver/trust.py`: the reader's posterior log-odds per candidate code as extra
  features.
- Modify: `labels_stage` reads (and `pipeline/README.md`, `dvc.yaml`).
- Test: `pipeline/tests/test_build.py`, `test_solver_trust.py`

- [ ] **Step 1: Failing tests:** the stage's deps; the trust features present and held out by quarter as today.
- [ ] **Step 2:** Implement. Run the stage, the labels stage and the solve (`uv run sfzmaps solver solve CODES`, an
  experiment run). Report "labels right" and the six measures:
  - against the last solve;
  - with the reader's posterior alone as the reading set.

  It ships if the solver improves without worsening the others (spec). Otherwise report and leave the features off
  by default.
- [ ] **Step 3:** Full suite. Commit.
