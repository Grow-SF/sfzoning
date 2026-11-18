### Task 10: Context packs and the reader's posterior with them (item 4, `wordread/context.py`)

**Files:** Create `pipeline/src/sfzmaps/solver/wordread/context.py`, `pipeline/scripts/wordread/build_pack.py`; Test
`pipeline/tests/test_wordread_context.py`.

**Interfaces:**
- **Produces:**
  - `Pack` (one JSON or parquet file per map), with:
    - `streets` (name, centreline linestring, EPSG:2227);
    - `blocks` (number, outline);
    - `codes` (the era's code list).
    - It has NO lot codes for the 2000 map; the builder raises if asked to add them.
  - `build_sf_pack(year) -> Pack`, from the DataSF centrelines and blocks the solver already loads (find the
    loaders; `lots/table.parquet` has `block_num`), with `codes` from `codereader.labels.table_codes`' era list.
  - `prior(pack, box_map: dict, kind_p: np.ndarray) -> dict[str, float]`: log prior over the candidate strings near
    the box:
    - street names: by the distance from the box's axis to each centreline and the angle difference;
    - the block number: the block holding the box centre;
    - codes: the era list, flat.

    An `escape` term covers every other string.
  - `fit_prior(words: pd.DataFrame) -> dict`: the weight w and escape mass e, by maximum likelihood on Terra-matched
    words of the other three quarters (never the owner's answers).
  - `posterior(ink_logp: dict[str, float], prior: dict, w, e) -> dict`: log p(s | ink) + w · log prior, with mass e
    on "other". The reader computes `ink_logp` exactly by CTC (`ctc.string_log_probs`) for every candidate string.
- [ ] **Step 1: Write the failing tests:**
  - a string off the list with clearly stronger ink wins;
  - with no pack, the posterior equals the ink-only one;
  - a box with no street within range falls back to the escape mass;
  - `build_sf_pack(2000)` refuses lot codes;
  - `fit_prior` never sees answered-block words (it raises if given one).
- [ ] **Step 2: Run them to verify they fail.**
- [ ] **Step 3: Implement,** then build the SF 2000 pack to `experiments/wordread/packs/sf-2000.parquet`.
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

