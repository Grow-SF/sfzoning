### Task 15: The learned box model (added by the owner, 11-17; built alongside Tasks 6-7, run in Task 8 arm (c))

The owner: "there are things you can learn about boxes and how many words they hold ... Don't just hardcode these
... There is a hidden function that you should be able to learn automatically." So no hand features (ink density,
the ink centroid against the box centre, gaps): a small network learns them from the box's own pixels.

**Files:** Create `pipeline/src/sfzmaps/solver/wordread/boxmodel.py`, `pipeline/scripts/wordread/run_boxmodel.py`;
Test `pipeline/tests/test_wordread_boxmodel.py`.

**Interfaces:**
- **Consumes:** boxes in `wordboxes.COLUMNS` form (today's, Task 6's, or Task 9's candidates); the page ink; the
  detector's per-ink text chance and, when present, its word-head outputs (core chance, tags); Terra's words
  (`textnet/boxes.py` `terra_boxes`); Task 7's weights when present; `textread/crops.py`'s rectification.
- **Produces:**
  - `box_examples(boxes, terra, sheets, quarter, weights=None) -> Examples`: per box, the rectified crop with the
    reader crops' own margin (`crops.py`; no new margin constant). Channels: ink, text chance, and the word-head
    outputs when present. Targets, with no cut:
    - the word count as a distribution: each Terra word contributes its share of ink inside the box, so the count is
      the Poisson-binomial of those shares;
    - per column along the box, whether it lies in the gap between two consecutive held Terra words (exact columns
      from their ends).
    - Plus joined real neighbours: two real Terra word crops on one line, the gap drawn from the measured
      distribution of real same-line gaps, the targets known by construction.
    - Only training-quarter words outside the answered blocks; it raises otherwise.
  - `BoxNet`: plain torch, like the reader's backbone. Outputs: count logits over 0..K (K = the largest count in the
    training examples, measured) and a boundary logit per column.
  - `fit_boxmodel(examples, recipe) -> model` inside `gpu.gpu_turn`; deterministic seeds.
  - `box_logp(model, crops) -> (count_logp, boundary_logp)`, for Task 9's `choice_features`.
- **The keep rule** (written now): on NE over seeds 0 and 1, Task 9's choice with the box model's log-probabilities
  against without:
  - the headline rises with the paired label interval excluding zero;
  - the held-out box numbers' `merged_with_word` count falls (Mission sheet: 293 today);
  - the review page agrees.

  A kept model is confirmed on all four folds at seed 0.
- [ ] **Step 1: Write the failing tests:**
  - the count target is the exact Poisson-binomial of the ink shares;
  - the boundary columns are exactly the gaps between held Terra words;
  - `box_examples` raises on held-quarter or answered-block words;
  - a short CPU fit on synthetic one-word and two-word crops separates them well above chance;
  - joined-neighbour gaps follow the measured distribution.
- [ ] **Step 2: Run them to verify they fail.**
- [ ] **Step 3: Implement.** No training runs in this step (Task 8 arm (c) trains it).
- [ ] **Step 4: Run the tests to verify they pass.**
- [ ] **Step 5: Commit.**

## Global Constraints

- **No hand-set cutoffs (the owner's standing rule; reviewers treat a breach as Important):**
  - nothing that decides is a fixed number: no match, split, join, filter, cap, search window, flag or in/out rule;
  - a percentile (q99, "the 99th") or a value read off a measured histogram and used as a cut is STILL a cutoff;
  - a published method's constant (PAN's 3, DBNet's 0.4, a 1% rule) is not exempt;
  - model each decision as a probability: a likelihood with measured spreads, a fitted logit or prior, a per-item
    weight, or a draw matching a whole measured distribution;
  - a setting that can't be a probability (e.g. a shrink ratio) is screened by the held-out outcome;
  - hard rules only for the physically impossible;
  - report-only measures (IoU 0.5 / 0.3 for the top-line, the review page's reason labels) are display, not decisions;
  - if this brief's text gives a cutoff, stop and report it, don't build it.
- **$0:** no LLM calls. Never print, log, copy or commit `pipeline/.env`, `site/.env.local` or any token.
- **One GPU (mps) process on the machine at a time:** every training or prediction inside `gpu.gpu_turn(True, device)`.
  Load near 20: ≤ 4 workers × 1 thread. Don't run pytest while a training runs (the session takes a GPU turn).
- **Deterministic:** seeded. New rngs are derived from the run seed (`np.random.default_rng([seed, k])`).
- **Held out:** by quarter (`data.Quarter`). The owner's answered blocks (`reader_stage.held_blocks()`) are never
  trained on, never used for composites, scenes or banks, and never used to fit a prior.
- **No answer key in a prior:** the 2000 zoning table's code at a lot is never in a 2000 context pack, and no prior is
  fitted on the owner's answers.
- **Defaults stay byte-identical:** every new recipe or reader option defaults off, so today's training and reading
  are unchanged. The pinned-batch tests and `GUARD_WEIGHTS` must keep passing.
- **No import-closure change for the existing stages:**
  - don't import new modules from `textnet/data.py`, `train.py`, `textread/*` stage paths, `wordboxes.py`;
  - pass new behaviour in as objects or options;
  - check with `uv run pytest -q tests/test_build.py`.
- **Files not to touch:** `build.py`, `dvc.yaml`, `dvc.lock`, `paths.py`, `cli.py`, `pyproject.toml`, `uv.lock`.
  No stage is rebuilt.
- **No commit and no new or edited file under `pipeline/src`, `inputs`, `dvc.yaml`, `pyproject.toml` or `uv.lock`
  while an `sfzmaps build` runs:** `pgrep -fl 'sfzmaps (build|stage)'` must be empty.
- **Experiments:**
  - outputs under `pipeline/data/solver/experiments/wordread/`, never committed;
  - each item is screened on fold NE at seeds 0 and 1, and a kept item is confirmed on all four folds at seed 0;
  - the keep rule is written in the task before its runs;
  - no verdict from one seed, from wrong-lot counts, or changed after the results.
- **The headline:** read right at our boxes over all answered labels of the scored folds, with a missed box counted
  wrong. Matching is T3's rule (`textread/read.py` `holding_box`): the box holding the label's point, and the
  highest-posterior box when several hold it.
- **Licences:** Apache-2.0 code only. Ideas from MMOCR or docTR may be re-expressed; nothing copied from AdelaiDet,
  DeepSolo, MapTD (GPL) or mapKurator.
- **Tests:** synthetic arrays in tests. Real data are injected; a real-data check skips without `pipeline/data`.
  Iterate set:
  `uv run pytest -q tests/test_wordread_*.py tests/test_solver_textnet.py tests/test_solver_textread.py tests/test_build.py`.
  Before a commit: `uv run pytest -q`.
- **Git:** `git add <paths>` and `git commit` as separate plain commands. Messages end with
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` and
  `Claude-Session: https://claude.ai/code/session_01UjTgrnx2j4PhzZErrad25Q`.

## Review Focus

1. **A label point held by two boxes, or by none:** the headline counts it once, by T3's rule, and a missed box is
   wrong, never dropped (Task 1 test).
2. **A Terra word split across two of our boxes, or two Terra words in one of ours:** the box numbers count each Terra
   word once (matched, or missed with reason "split" / "merged"). A merged box counts once as matched, to the
   best-IoU word (Task 1 test).
3. **A context pack with no entry near a box, or a kind with no list:** the posterior falls back to the ink alone
   (escape mass), never to zero probability (Task 10 test).
4. **Word formation on a core that touches the tile edge or the page edge:** boxes are clipped, never wrapped or
   dropped (Task 6 test).
5. **New recipe or reader options left at their defaults:** batches, weights and reads are byte-identical to today
   (Tasks 3, 5 and 12 tests).

