### Task 2: The Mission Street review page (`wordread/review.py`, `scripts/wordread/review_page.py`)

**Files:** Create `pipeline/src/sfzmaps/solver/wordread/review.py`, `pipeline/scripts/wordread/review_page.py`; Test
`pipeline/tests/test_wordread_review.py`.

**Interfaces:**
- **Consumes:** Task 1, `textnet/data.py` (page, ink, Terra words), and the fold models' outputs (chance, kind,
  direction).
- **Produces:**
  - `mission_sheet() -> str`: the 2000 sheet holding the longest length of the DataSF centreline named MISSION ST
    inside its footprint, mapped through the warp. Find the centreline loader the walls use (`STREET_CROSSING`).
  - `miss_reason(word_rect, chance, ink, components, level) -> tuple[str, dict]`. The reason is one of:
    - `"below_level"`;
    - `"merged_into_line"`: the component holding its ink has a length > 3× the word's length;
    - `"split"`;
    - `"wrong_size"`;
    - `"outside_map"`.

    The details hold the highest chance on the word's ink, the share of its ink above the level, the swallowing
    component's size, and the reading at Terra's box when it is given.
  - `colour(ours: str, terra: str | None) -> str`: "green" when equal, "purple" when different, "none" with no Terra
    word.
  - `page_data(sheet, boxes, reads, terra, chance, before: dict | None) -> dict`: everything the page draws, with the
    top-line panels from Task 1.
  - `render(data, out: Path) -> Path`: one self-contained HTML page.
    - The full-resolution sheet image: grey ink, red where chance ≥ 0.5 (text), white paper.
    - SVG overlays:
      - every box, with a reading-direction arrow and a tag (kind · chance · angle);
      - the reading in green or purple, with Terra's string in red after a purple one;
      - green-shaded boxes for ours unmatched but read as text;
      - thin grey outlines for ours unmatched and read as "none";
      - yellow-shaded dashed Terra boxes for misses, with the reason and details.

      Shading sits under the red ink and never over it.
    - Tapping a box shows its metadata.
    - The two top-line panels.
    - A line naming the models that drew it. Every px and box comes from the fold model that held out that px's
      quarter.
    - Pan and zoom.
    - It works on a phone, and in both themes.
    - Under 16 MB.

- [ ] **Step 1: Write the failing tests**

```python
def test_colour_rules():
    assert review.colour("RH-2", "RH-2") == "green" and review.colour("RM-2", "RH-2") == "purple"
    assert review.colour("RH-2", None) == "none"

def test_a_word_inside_a_long_line_component_is_merged_into_line():
    r, d = review.miss_reason(WORD, CHANCE_HIGH, INK_WORD_ON_LINE, COMPONENTS_LINE, 0.99)
    assert r == "merged_into_line" and d["component_px"] > 3 * d["word_length_px"]

def test_a_word_below_the_level_says_how_close():
    r, d = review.miss_reason(WORD, CHANCE_062, INK_WORD, COMPONENTS_NONE, 0.99)
    assert r == "below_level" and 0.6 <= d["max_chance"] <= 0.63

def test_shading_never_covers_red_ink(tmp_path):
    out = review.render(PAGE_DATA_TINY, tmp_path / "p.html")
    assert review.red_px_after_overlay(out) == review.red_px_of(PAGE_DATA_TINY)
```

- [ ] **Step 2: Run them to verify they fail.**
- [ ] **Step 3: Implement.** The script `review_page.py` builds the page for the Mission sheet from the latest run, or
  from the baseline. It writes it to `experiments/wordread/review/<run>/review.html`, and keeps `before` from the
  previous run's numbers. The controller publishes it to one Artifact URL and updates it in place.
- [ ] **Step 4: Run the tests, then build the baseline page.** Open the HTML and check the overlays against the agreed
  mockup (`https://claude.ai/artifact/2ytdcmDwbya7t72JMTxg5F`). Expected: PASS, and a page.
- [ ] **Step 5: Commit** `review.py`, `review_page.py` and the test.


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

