# The Map's Labels From the Scan Alone: Implementation Plan

> **SUPERSEDED (2026-11-16 ~12:30).** Written before the red team
> (`.superpowers/sdd/2026-11-16-labels-from-the-scan/redteam.md`) and the library survey (`survey.md`). Next:
> `docs/superpowers/plans/2026-11-16-search-move-and-reader-spikes.md`. The reader rebuild gets a revised spec and
> plan after its spikes.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** the 2000 solver's printed labels come from a local detector of word boxes and a reader of each box, with no
LLM output anywhere in the pipeline. The solve is judged against the current one on the saved LLM reads.

**Architecture:**
- **P1–P2:** the detector gains a line-axis output and per-px box geometry, and treats Terra's misses as unlabelled.
  Its schedule is chosen by every output's measure, and its quarter-held-out models are written by
  `solver-text-reader`.
- **P3:** groups each px's box vote into weighted word boxes.
- **P4:** the reader reads every box both ways (codes, street names, block numbers, other text, none) with priors set
  by where the box is.
- **P5:** rebuilds the solver's labels from boxes and posteriors.
- **P6:** the one eye-check page and the report.

**Tech Stack:** PyTorch (detector, mps in one exclusive turn), numpy, OpenCV, shapely/geopandas (centrelines,
parcels), PyMaxflow (solver, unchanged), Typer, DVC, pytest.

**Spec:** `docs/superpowers/specs/2026-11-16-text-pipeline-from-the-scan-design.md`. It builds on
`2026-11-15-code-reader-design.md`, and on sections 3–4 of `2026-11-16-code-reader-read-along-the-word-design.md` (the
reading pass). It supersedes the R1–R4 plan `2026-11-16-code-reader-read-along-the-word.md`; that plan's R1 (commit
ce828df: the generalised hold-outs `SheetData.held` and the `solver-text-reader` stage) is kept and extended here.

## Global Constraints

- **No LLM output when the pipeline runs on a map:** not Terra's or Sol's boxes, points or strings, not the saved reads,
  not `label_fixes.csv`.
  - These are training labels and scoring sets only.
  - A test asserts that the inference code paths don't read them: the boxes, the reader and the labels stage.
- **Never the zoning table's code at a label's spot** outside scoring. The code list (names) is allowed.
- **Never train on what is scored:**
  - the held-out quarter;
  - the validation area and the test drawings (1980, 1972, 1975);
  - the owner's answered labels and their blocks (`labels.refuse_held`).
  - Calibration constants are cross-fitted on the owner's answers.
- **No brute-force sweeps** (angles, positions, sizes), and no hard cutoffs as decisions. A truncation is a tolerance
  with its dropped mass bounded and reported.
- **Deterministic.**
  - Seeded; `torch.use_deterministic_algorithms(True)`.
  - Workers one thread each (OMP/OPENBLAS/VECLIB/MKL/NUMEXPR_NUM_THREADS=1, `torch.set_num_threads(1)`,
    `cv2.setNumThreads(1)`).
  - One exclusive GPU turn per training or predicting process. Never two GPU jobs.
- **Machine load near 20.** Check `uptime` a minute into any long run.
- **Value before verification.** A long run's quality is measured and shown before any repeat run for determinism.
- **Scoring sets are noisy** (owner).
  - Every score is reported with a bootstrap interval, and with the scoring set's own measured error rate where a
    second source allows.
  - A difference inside both is a tie.
  - Ties are broken by a page of the disagreeing cases with the scan, else by the more principled method. Never a
    one-label gate.
- **One measure per comparison,** on what the fit didn't see. Synthetic cases draw every number from measured
  distributions.
- **Process.**
  - Tests run on the CPU (`cd pipeline && uv run pytest -q`).
  - Commit locally only, never push. Stage only your own files. Never commit `pipeline/data/`.
  - The controller runs `sfzmaps build`.
  - Commit messages end with `Co-Authored-By: <your model> <noreply@anthropic.com>` and
    `Claude-Session: https://claude.ai/code/session_01UjTgrnx2j4PhzZErrad25Q`.

## Review Focus

1. **A word printed upside down, or reading bottom-to-top, is found and read right.** Test in P1:
   `test_axis_target_is_the_same_for_a_word_and_its_half_turn`. Test in P4:
   `test_upside_down_and_sideways_codes_read`.
2. **Text Terra missed isn't taught as "not text".** Test in P1:
   `test_unboxed_ink_is_weighted_by_the_measured_miss_rate`.
3. **A street name printed beside or over a district line isn't read as a code.** Test in P4:
   `test_street_name_beside_a_line_is_not_a_code`.
4. **Two touching words keep both boxes.** Test in P3: `test_touching_words_keep_both_boxes`.
5. **The labels stage reads nothing from the LLM passes.** Test in P5:
   `test_labels_stage_reads_no_llm_output` (the stage's declared reads and the import closure).

---

### Task P1: The detector's axis and box outputs, Terra's misses unlabelled, every output measured

**Files:**
- Modify: `pipeline/src/sfzmaps/solver/textnet/model.py`
  - `OUTPUTS`: text 1, kind 4, axis 2 (cos 2θ, sin 2θ), box 4 (distances to the four sides).
  - The 360° direction output is removed.
- Modify: `pipeline/src/sfzmaps/solver/textnet/data.py`
  - `crop_targets` yields `axis2` (cos 2θ, sin 2θ) and `box` (the four distances, in page px, scaled with the crop).
  - It also yields `unboxed`: ink outside every box.
  - Quarter turns rotate both.
- Modify: `pipeline/src/sfzmaps/solver/textnet/train.py`
  - **Loss:** text BCE with unboxed ink weighted by `MISS_WEIGHT`; kind as now; axis (1 − cosine of the doubled angle)
    on text px; box as the IoU loss of the voted rotated box against the true one (EAST's AABB-IoU in the box's frame,
    plus the axis term), on text px.
  - `measure_miss_rate(sheets) -> dict`: the share of the other passes' boxes and Sol's points with no Terra box, by
    kind and map, recorded.
- Modify: `pipeline/src/sfzmaps/solver/textnet/predict.py`: `Outputs` gains `axis` (float16 2×) and `box` (float16 4×);
  `direction` goes.
- Modify: `pipeline/src/sfzmaps/solver/textnet/scores.py` and `trial.py`. Every measure gets a bootstrap interval:
  - axis error (degrees, modulo 180, against Terra's angle) on matched text;
  - box overlap of each text px's voted box with its Terra box (median, and the share over 0.5 and 0.7);
  - kind agreement;
  - words found, and long-line ink called text, as now.
- Modify: `pipeline/README.md` (the detector's outputs and measures).
- Test: `pipeline/tests/test_solver_textnet.py`

**Interfaces:**
- **Produces:**
  - `predict.Outputs(chance, kind, kind_p, axis, box)`;
  - `scores` entries `axis_error`, `box_iou`, `kind_agreement`, each `{value, low, high}`.
- **`MISS_WEIGHT`:** p(unboxed ink is not text), from `measure_miss_rate`, per map.

- [ ] **Step 1: Failing tests.**
  - `test_axis_target_is_the_same_for_a_word_and_its_half_turn`.
  - `test_box_targets_turn_with_the_crop`: the four distances permute under a quarter turn.
  - `test_unboxed_ink_is_weighted_by_the_measured_miss_rate`.
  - `test_measure_miss_rate_counts_other_passes_words_without_terra`.
  - `test_scores_report_axis_box_and_kind_with_intervals`.
  - The recorded-weights guard is re-recorded for the new outputs (`GUARD_*`), as its comment says.
- [ ] **Step 2: Implement.** The tests pass, and the full suite passes.
- [ ] **Step 3: Commit.**

### Task P2: The schedule by every measure, and the reader's detectors

**Files:**
- Modify: `pipeline/src/sfzmaps/solver/textnet/stage.py` and `reader_stage.py`
  - the reader stage's recipe is the one chosen here;
  - its unknown area covers each other pass's word and Sol point with a measured per-word extent, not a single radius;
  - it saves `axis` and `box` on ink and the px next to it;
  - `solver-text` (the walls') keeps its own recipe unless the owner decides otherwise.
- Modify: `pipeline/README.md` (the trials table).
- Test: `pipeline/tests/test_solver_textnet.py`, `pipeline/tests/test_build.py`

**Interfaces:**
- **Consumes:** P1.
- **Produces:** `data/solver/text-reader/<sheet>/{chance.png, kind.npz, axis.npz, box.npz}`, each sheet's px from the
  model that never saw their quarter; `held.json`.

- [ ] **Step 1: The measured loop,** on the validation area and test drawings, ONE trial at a time on the GPU:
  - candidates: the stage recipe at 2,000, 4,000 and 8,000 steps, with the new outputs;
  - every candidate gets all P1 measures with intervals.

  Choose the shortest schedule whose every measure is a tie with or better than the longest's (the spec's rule). Report
  the table.
- [ ] **Step 2: A stage test:** the per-word unknown extent from the measured word shapes, not one radius.
- [ ] **Step 3: Real run** of `solver-text-reader` with the chosen recipe.
  - **Quality first:** the four checks on the training labels (axis within 15° and 45°, box overlap, word ink called
    text, kind), against the old 8,000-step models as reference.
  - **Only then,** one repeat run for determinism.
- [ ] **Step 4: Full suite. Commit.**

### Task P3: Word boxes

**Files:**
- Create: `pipeline/src/sfzmaps/solver/codereader/boxes.py`
- Test: `pipeline/tests/test_codereader.py`

**Interfaces:**
- **Consumes:** P2's outputs.
- **Produces:**
  - `WordBox`: `centre` (page px), `axis` (radians modulo π), `length`, `height`, `kind_p` (4), `p` (the box's
    probability), `alt_of` (index of the box it is an alternative to, or −1), `log_w`.
  - `word_boxes(sheet) -> list[WordBox]`: the votes grouped by agreement, as EAST's locality-aware merge, with weights
    kept, never suppressed.
  - `box_scores(boxes, sheet, area) -> dict`, with intervals:
    - recall of Terra's words by kind (curve over the overlap threshold);
    - the "Terra / other passes / unseen" split of detections.
  - `eye_sample(boxes, n, seed) -> list`: unseen boxes, for P6's page.

- [ ] **Step 1: Failing tests,** synthetic, with measured shapes:
  - `test_one_word_gives_one_box`;
  - `test_touching_words_keep_both_boxes`;
  - `test_a_word_crossed_by_a_line_keeps_its_box`;
  - `test_box_scores_split_terra_other_unseen`.
- [ ] **Step 2: Implement.** The tests pass.
- [ ] **Step 3: Real run on every 2000 sheet.**
  - **Report:** recall by kind on held-out quarters (curve and intervals), the detection split, and seconds per
    sheet.
  - **Picture:** a dense window with boxes drawn (say what it shows).
- [ ] **Step 4: Full suite. Commit.**

### Task P4: The reader on detected boxes

**Files:**
- Create: `pipeline/src/sfzmaps/solver/codereader/reading.py`: the read-along-the-word pass of the 11-16 design,
  sections 3–4, on a box's frame.
- Create: `pipeline/src/sfzmaps/solver/codereader/hypotheses.py`: the per-kind priors.
- Modify: `pipeline/src/sfzmaps/solver/codereader/posterior.py`
  - `evidence` reads a `WordBox` both ways;
  - the pose grid, `PosePrior`, the 2-D placement prior and the end-break term are deleted;
  - Terra's and Sol's inputs are deleted from inference.
- Create: `pipeline/src/sfzmaps/solver/codereader/run.py`: `score_all(boxes_by_sheet, workers) -> Path`, writing
  records.
- Create: `pipeline/src/sfzmaps/solver/codereader/evaluate.py`
  - the owner's answers matched to boxes by position; constants cross-fitted (two folds);
  - top-1, top-3, mean −log p and reliability, with intervals;
  - box-found against box-missed answers;
  - the saved LLM reads' scores on the same answers as the yardstick.
- Modify: `pipeline/src/sfzmaps/cli.py`: `sfzmaps solver read-labels`, `sfzmaps solver reader-eval`.
- Test: `pipeline/tests/test_codereader.py`

**Interfaces:**
- **Consumes:** `WordBox` (P3); Task 2's `Font` and `NoiseModel`; `labels.line_mask`.
- **Produces:**
  - `Reading` (log evidence per kind and string, over temperatures);
  - `posterior(reading, prior, constants) -> dict[(kind, string), float]`, with "none" included;
  - the code posterior per box: `code_posterior(...) -> dict[str, float]`;
  - records in `data/solver/experiments/labels-records/<commit>/`.
- **The priors** (`hypotheses.py`):
  - **codes:** the code list, p(code), and p(form | code) from training counts;
  - **street names:** the names of DataSF / OSM centrelines (`data/ref/streets.parquet`) within reach, weighted by
    distance and angle to the box, plus their printed forms;
  - **block numbers:** the block under the box and its neighbours (`data/solver/lots`);
  - **other text:** a bigram;
  - **kind:** p(kind) from the detector's `kind_p`, measured on training boxes.

  Every kind keeps mass everywhere.

- [ ] **Step 1: Failing tests,** synthetic, with measured numbers:
  - `test_upside_down_and_sideways_codes_read`;
  - `test_line_through_h_does_not_make_it_m`;
  - `test_widely_spaced_code_reads_that_code`;
  - `test_prefix_pays_for_the_ink_it_leaves`;
  - `test_street_name_beside_a_line_is_not_a_code`;
  - `test_block_number_reads_its_block`;
  - `test_lines_only_window_prefers_none`;
  - `test_temperature_grid_gives_each_temperature_as_alone`;
  - `test_posterior_sums_to_one`;
  - `test_inference_reads_no_llm_output`: monkeypatch the saved-reads loaders to raise.
- [ ] **Step 2: Implement.** The tests pass.
- [ ] **Step 3: The full run** (`read-labels`, load near 20).
  - **Report:** minutes and seconds per box.
  - **Evaluation** (`reader-eval`):
    - codes on the owner's answers, with intervals and the LLM yardstick, box-found against box-missed;
    - street names and block numbers against Terra's strings and the centreline / block underneath, as checks.
  - **A page of the disagreements** between the reader and the LLM reads, with the scan (say what it shows).
- [ ] **Step 4: Full suite. Commit.**

### Task P5: The solver's labels from the pipeline

**Files:**
- Modify: `pipeline/src/sfzmaps/solver/labels_stage.py`
  - prints become the boxes whose code posterior has real mass;
  - readings per held-out quarter;
  - placement, street-side and lot-size features from boxes.
- Modify: `pipeline/src/sfzmaps/solver/trust.py`: one reader. Its calibration (the log-odds of its code posterior, as
  features) is learned from the table, held out by quarter as today.
- Modify: `pipeline/src/sfzmaps/solver/placement.py`
  - options from box geometry;
  - `label_fixes.csv` only in training and scoring;
  - the leader-move key bug (old code against `fix_code`) fixed, with its failing test first.
- Modify: `pipeline/src/sfzmaps/build.py`: `solver-labels` reads `solver-text-reader` and the records, not
  `solver-prints` or the saved reads.
- Modify: `pipeline/README.md`.
- Test: `pipeline/tests/test_solver_trust.py`, `test_build.py`, and a new `test_solver_labels_from_boxes.py`.

**Interfaces:**
- **Consumes:** P3's boxes and P4's records.
- **Produces:** the same files the solve reads today (`readings_<q>.npz`, `trust.json`, `streetside_<q>.npz`,
  `lotsize.npz`), from the pipeline. The solve's code is unchanged.

- [ ] **Step 1: Failing tests.**
  - `test_labels_stage_reads_no_llm_output`;
  - `test_leader_moves_keyed_by_fix_code`;
  - `test_readings_held_out_by_quarter`.
- [ ] **Step 2: Implement.** The tests pass.
- [ ] **Step 3: Solve** as an experiment run (`sfzmaps solver solve labels-from-scan --walls data/solver/walls`).
  - **Report:** the six measures, every lot, with intervals, against `final` on the LLM reads, and the biggest new and
    fixed incidents.
  - **A page of the biggest changes** (say what it shows).
  - Reported, not gated (owner).
- [ ] **Step 4: Full suite. Commit.**

### Task P6: The eye check and the report

**Files:**
- Create: `pipeline/src/sfzmaps/solver/codereader/eyecheck.py`: an HTML page of `eye_sample` boxes (P3) and the
  reader's disagreements (P4), each cropped from the scan at full resolution, for the owner to mark text / not text /
  right / wrong.
- Modify: `pipeline/README.md`
- Test: `pipeline/tests/test_codereader.py`

- [ ] **Step 1:** The page. The controller publishes it as an Artifact; the owner's marks are read back. The
  corrected precision and the reader's corrected accuracy are reported, never trained on.
- [ ] **Step 2:** The final report: what the pipeline gives against the LLM-read solve, with every number's interval and
  each scoring set's error rate.
- [ ] **Step 3: Full suite. Commit.**
