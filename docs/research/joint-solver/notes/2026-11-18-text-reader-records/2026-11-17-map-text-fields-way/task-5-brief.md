### Task 5: Detector word heads and their targets (`textnet` options)

**Files:**
- Modify `pipeline/src/sfzmaps/solver/textnet/model.py` (`OUTPUTS` per recipe), `data.py` (`crop_targets(...,
  words=False)`), `train.py` (`Recipe.heads="all"` gains `"words"`; the loss terms).
- Create `pipeline/src/sfzmaps/solver/wordread/shrink.py`.
- Test `pipeline/tests/test_wordread_heads.py`.

**Interfaces:**
- **Produces:**
  - **Amended (owner, 11-17: no picked core size):** NO shrink ratio. `shrink.prepare(sheets)` fills
    `SheetData.cores` with each word's own labelled rectangle. Touching words are separated by the learned tags, the
    geometry votes (Task 6) and the learned box model (Task 15). `measure_r`, `core` and `unclip` (built in
    87f3d82/da3a149) stay only as a record of the overlap measurement; nothing uses r.
  - **`crop_targets(..., words=True)`** adds:
    - `core` (uint8, all px in a word's rectangle);
    - `geo` (float32 (5, h, w) at core px: distances to the word's top, bottom, start and end edges in crop px, plus
      the reading angle, the semantic edges taken from the word's reading angle);
    - `tag_id` (int32 word id at core px, 0 elsewhere).
  - **`Recipe.heads="words"`** gives `OUTPUTS` = 7 + 1 (core) + 4 (geometry distances) + 4 (the PAN tag). The angle
    reuses channels 5-6.
  - **The loss adds:**
    - for the core: BCE over all known px with hard-negative mining at 3:1, plus Dice;
    - for geometry at core px: IoU loss on the distances, 10 × (1 − cos) on the angle;
    - for the tag: PAN's aggregation (pull to ≤ 0.5) and discrimination (push ≥ 3), each × 0.25.

    The per-ink text head (channels 0-6) is trained exactly as today.
  - `heads="all"` (the default) stays byte-identical.
- [ ] **Step 1: Write the failing tests:**
  - (amended) `prepare` fills each word's core with its own rectangle;
  - `geo` at a core px of a 0° word gives the right four distances;
  - the default recipe's batches and the `GUARD_WEIGHTS` are unchanged;
  - `test_build.py` passes.
- [ ] **Step 2: Run them to verify they fail.**
- [ ] **Step 3: Implement.** (The overlap curve stays recorded in `experiments/wordread/detector/shrink.json`.)
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

