# Bundle The Explorer's Shared Files Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A `sfzmaps pull` from Drive restores everything the explorer's shared layer needs.

**Architecture:** `share.SHARED_GLOBS` decides what's bundled. It already catches `out/shared/*.pmtiles`, through `out/*/*.pmtiles`, but none of the files that make that archive usable. Add those by name.

**Tech Stack:** Python, pytest.

**Spec:** None. The request is the bug report: `processed-shared.zip` holds only the tiles.

## Global Constraints

- No API calls and no live runs.
- Never commit `pipeline/data/`.
- Only `processed-shared.zip` and `PROCESSED.json` should change in the rebuilt bundle.

## Review Focus

- A build in progress leaves `*.tmp.gpkg` in `out/shared/`. It must not be bundled, which rules out a bare `out/shared/*` glob.
- `*.counts.json` looks like a build detail, but `shared.build` rebuilds without it, so a pull without it would cost a five-minute rebuild.
- `_group` must put these files in `processed-shared.zip`. Grouping by `parts[1]` already gives `shared`.

---

### Task 1: Bundle the shared files

**Files:**
- Modify: `pipeline/src/sfzmaps/share.py` (`SHARED_GLOBS`)
- Test: `pipeline/tests/test_share.py`

**Interfaces:**
- Consumes: `share.bundle`, `share.pull`, and the test file's existing `data`, `_w` and `_fresh_pull` helpers.
- Produces: nothing new.

- [ ] **Step 1: Write the failing test**

`test_pull_restores_the_explorers_shared_files`: write `out/shared/districts.json`, `districts-x.pmtiles`, `districts-x.table.gz`, `districts-x.cards.json.gz`, `districts-x.counts.json` and `districts-x.tmp.gpkg`; bundle and pull into an empty root; assert the first five exist with the same bytes and the `.tmp.gpkg` doesn't.

- [ ] **Step 2: Run it and see it fail**

Run: `uv run pytest tests/test_share.py -k shared_files -q`
Expected: FAIL, `districts.json` missing after the pull.

- [ ] **Step 3: Add to `SHARED_GLOBS`**

`out/shared/districts.json`, `out/shared/*.table.gz`, `out/shared/*.cards.json.gz`, `out/shared/*.counts.json`.

- [ ] **Step 4: Run all the pipeline tests**

Run: `uv run pytest -q`
Expected: no failures.

- [ ] **Step 5: Commit and open a PR**

- [ ] **Step 6: Rebuild the bundle and verify**

Run `sfzmaps bundle` with `SFZMAPS_DATA_DIR` set to the real data. Expected: `processed-shared.zip` holds all five files, every zip matches `PROCESSED.json`, and no other zip's hash changed.
