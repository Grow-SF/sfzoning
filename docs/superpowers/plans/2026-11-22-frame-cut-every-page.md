# Frame finder: the production cut covers every page

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `sfzmaps frame` cuts every map page with the final model, training sheets included; nothing from `--dev` is reused in the production cuts.

**Architecture:** `run.predict_all` builds a `PageJob` for every page of `data.map_pages`, with the final heads. The copy of `dev/` outputs into the production folders is removed. `--dev` and `--test` stay as they are: scoring only, written under `data/frame/dev/` and `data/frame/test/`, read by nothing in production.

**Tech stack:** Python, `uv run` from `pipeline/`, pytest.

**Spec:** the owner's ruling, 2026-11-22: "Running the cuts should run on everything, even if it's trained on that sheet. Don't reuse the training data for prod cuts. Training data is only for training." It amends `docs/superpowers/specs/2026-11-20-frame-crop-design.md`.

## Global constraints

- Branch `frame-cut-every-page` off `main`, its own PR. Not in PR #70.
- No change to the model, the labels, the fit, `--dev` or `--test` scoring, or the shape decision.
- No `sfzmaps build`. No refit: the saved model stays current (staleness depends on labels, pages and features, not on this code).
- $0, no LLM output.
- The owner merges.

## Review focus

- A training sheet with no `dev/` outputs: a plain run still writes its four files (Task 1's test).
- A training sheet with `dev/` outputs present: the production files are the final heads' cut, not a copy (Task 1's test).
- `dev/scores.json` missing: `pages.json` still lists every page (Task 1's test).
- A reader of `pages.json` that expects `held_out`: none in the repo (`grep -rn held_out pipeline/src/sfzmaps/frame` finds only `run.py`); the key is dropped.

---

### Task 1: `predict_all` cuts every page

**Files:**
- Modify: `pipeline/src/sfzmaps/frame/run.py` (module docstring lines 37-38; `_copy_dev_outputs`; `predict_all`)
- Modify: `pipeline/src/sfzmaps/cli.py` (the `frame` command's help text, if it mentions dev outputs)
- Test: `pipeline/tests/test_frame_run.py` (`test_predict_every_other_page_with_the_fitted_model`)

**Interfaces:**
- Produces: `predict_all(device, out, turn)` unchanged in signature. Under `out/<series>/<sheet>/`: the four `PAGE_FILES` for every map page. `pages.json` rows: `series`, `sheet`, the ink figures, `side_rank`, `dropped_rank`; no `held_out`.

- [ ] **Step 1: Rewrite the test** as `test_predict_cuts_every_page_with_the_fitted_model(fake_world)`:
  - after `run.fit("cpu", out)`, write the fake `dev/` outputs (`b"dev"`) for the labelled sheets as the test does now, but **no** `dev/scores.json`;
  - `run.predict_all("cpu", out)`;
  - for every sid in `set(UNLABELLED) | set(labelled)`: the folder holds exactly the four `PAGE_FILES`, `outside.png` opens as a mode-"1" image (so it is not the `b"dev"` bytes), and `outline.json`'s `"shape"` is not None;
  - `run.sheet_dir(out, ("a", "c0"))` still doesn't exist (not a map page);
  - `pages.json` holds exactly those sids, and no row has a `held_out` key.
- [ ] **Step 2: Run it:** `uv run pytest -q -n 0 tests/test_frame_run.py -k cuts_every_page`. Expected: FAIL (the labelled sheets hold `b"dev"`).
- [ ] **Step 3: Change `predict_all`:** `todo = pages` (every map page); delete the `_copy_dev_outputs` loop and its warning; delete the block that adds rows from `dev/scores.json` and its warning; rows are `{"series": s, "sheet": h, **v}`. Delete `_copy_dev_outputs`. Update the two docstrings to: every map page is decided with the final heads; `--dev` and `--test` outputs are for scoring only.
- [ ] **Step 4: Run** `uv run pytest -q tests/test_frame_run.py tests/test_frame_*.py`. Expected: all pass.
- [ ] **Step 5: Commit** `frame: the production cut covers every page with the final model; nothing copied from --dev`.

### Task 2: Run it on the real pages and look

Run where the frame data already is (features and model, 369 pages): the `hatch-cnn` worktree, `.claude/worktrees/saved-reads`. A fresh worktree would need the features extracted again.

- [ ] **Step 1: Merge** `frame-cut-every-page` into `hatch-cnn`.
- [ ] **Step 2: Check nothing else is on the GPU** (`pgrep -fl 'hatchnet|sfzmaps frame'`), then run `uv run sfzmaps frame --predict-only` from `saved-reads/pipeline`. This is the frame finder's own run over its 369 pages (the last one took 2,011 s for 318 pages), not a pipeline build. Expected: no "no outputs for" lines; `data/frame/1921/` has 16 folders; every series folder has a cut for each of its map pages.
- [ ] **Step 3: Look at 1921's seven training sheets' pictures** (`data/frame/look/1921/<sheet>.png`): `094142210`, `094149543`, `094219795`, `094308402`, `094325427`, `094333266`, `094355582`. Send them to the owner: this is what the final model cuts on sheets it trained on.
- [ ] **Step 4: Run** `uv run sfzmaps frame --test`. Expected: the 22 test sheets' table equals #73's (frame ink left in 0.10%, label-box ink left in 24.2%, map ink cut 0.14%): the test sheets' cuts don't change.

### Task 3: PR

- [ ] **Step 1: Push** `frame-cut-every-page`; open a PR against `main`: what changed, why (the ruling above), that the 51 training sheets' production cuts now come from the final model, the test table from Task 2, and that `Cache shipped` will show stale stages only if `cli.py` changed (say which).
- [ ] **Step 2: Report** the PR link.

## Then

Back to `docs/superpowers/plans/2026-11-22-hatch-reader-frame-cut.md` on `hatch-cnn`: rebuild the 1921 hatch data on the new cuts (77 s last time) before its Task 2.

## Not in this plan

- A per-series option for `sfzmaps frame`.
- #72's faults.
