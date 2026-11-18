### Task 1: The end-to-end measures (`wordread/measures.py`)

**Files:** Create `pipeline/src/sfzmaps/solver/wordread/__init__.py`, `measures.py`; Test
`pipeline/tests/test_wordread_measures.py`.

**Interfaces:**
- **Consumes:**
  - `textread/read.py`: `holding_box`, `label_scores`, `CodeList`, `boot_interval`;
  - `wordboxes.py`: `poly_area`, `pairs`;
  - `textnet/boxes.py`: `terra_boxes`.
- **Produces:**
  - `headline(boxes: pd.DataFrame, reads: dict, answers: pd.DataFrame, quarters: tuple[str, ...]) -> dict`:
    - keys `n`, `right`, `share`, `ci`, `with_box`, `right_with_box`, and `by_quarter`;
    - `reads` maps a box row to its posterior, as `text-read` stores it;
    - `answers` is the owner's labels, with their sheet, point, truth and quarter.
  - `box_numbers(boxes, terra: pd.DataFrame, reads, iou=(0.5, 0.3)) -> dict`, with keys:
    - `terra`, `ours`, `matched`, `missed`, `ours_unmatched_text`, `ours_unmatched_none`, `matched_to_missed`;
    - `missed_reasons` (a counter of the reason codes from Task 2);
    - per IoU and per quarter.
  - `read_numbers(boxes, terra, reads) -> dict`: on the matched boxes, `same_as_terra`, `differs`, and
    `codes_right_vs_answers`.
  - `compare(now: dict, before: dict) -> dict`: the differences, for "vs last run".
  - `paired_labels(a: pd.DataFrame, b: pd.DataFrame, boot=2000, seed=0) -> dict`:
    - a and b give right or wrong per answered label for two arms;
    - the bootstrap resamples blocks (clusters of labels);
    - returns `diff`, `lo` and `hi`.

- [ ] **Step 1: Write the failing tests**

```python
def test_a_label_with_no_box_counts_wrong():
    r = measures.headline(BOXES_ONE, READS_ONE, ANSWERS_TWO_ONE_UNBOXED, ("ne",))
    assert r["n"] == 2 and r["right"] == 1 and r["with_box"] == 1

def test_a_point_in_two_boxes_goes_to_the_higher_posterior_box():
    r = measures.headline(BOXES_OVERLAP, READS_OVERLAP_SECOND_RIGHT, ANSWERS_ONE, ("ne",))
    assert r["right"] == 1

def test_a_terra_word_split_over_two_boxes_is_missed_with_reason_split():
    b = measures.box_numbers(BOXES_SPLIT, TERRA_ONE, READS_SPLIT)
    assert b[0.5]["missed"] == 1 and b[0.5]["missed_reasons"]["split"] == 1

def test_a_merged_box_matches_one_word_and_misses_the_other():
    b = measures.box_numbers(BOXES_MERGED, TERRA_TWO, READS_MERGED)
    assert b[0.5]["matched"] == 1 and b[0.5]["missed_reasons"]["merged"] == 1

def test_paired_labels_of_identical_arms_is_zero():
    p = measures.paired_labels(ARM, ARM, boot=50)
    assert p["diff"] == 0 and p["lo"] == 0 == p["hi"]
```

- [ ] **Step 2: Run them to verify they fail.** `uv run pytest tests/test_wordread_measures.py -v`. Expected: FAIL
  (no module).
- [ ] **Step 3: Implement** to the interfaces. A Terra word is "split" when two or more of our boxes each meet it at
  IoU ≥ 0.1 and none at ≥ 0.5. It is "merged" when the box that meets it best also meets another Terra word better.
- [ ] **Step 4: Run the tests to verify they pass.** Then write today's baseline to
  `experiments/wordread/baseline/{headline,boxes,reads}.json`, from `data/solver/boxes` and `data/solver/text-read`
  (the headline should come out at 41.1% over 719). Expected: PASS, and a baseline of about 0.411.
- [ ] **Step 5: Commit** (after the pgrep check): the two module files and the test.


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

