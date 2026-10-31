# Editor auto-fill Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** An editor button that seeds the automatic refine from the reviewer's hand-placed points and returns a dense point set to review and save.

**Architecture:**
- **Server:** a new FastAPI endpoint in `editor/server.py` runs `refine.refine` + `classify_changes` in memory and parks the result in `autofill.json`.
- **Scoring:** preview and save score against that file when the body says `autofill: true`. Save promotes it into `rough.json` / `refine_raw.json` / `refine.json` before the usual review save.
- **Front end:** vanilla JS in `editor.js` adds the button and the flag. Stale 5 / 12 ft labels are corrected to 7 / 15.

**Tech Stack:** Python 3.12, FastAPI, numpy, pytest + FastAPI TestClient; vanilla JS.

**Spec:** `docs/superpowers/specs/2026-10-31-editor-autofill-design.md`

## Global Constraints

- No change to `refine.py`, `runner.py`, `score.py`, `config.PASS_*`, or the automatic pipeline's behavior.
- No new dependencies. The editor stays bound to 127.0.0.1.
- `AUTOFILL_MIN_POINTS = 4` (module constant in `editor/server.py`).
- The auto-fill endpoint never writes anything except `autofill.json`. Only save promotes it.
- Tests use the synthetic demo (`env` fixture in `pipeline/tests/test_editor_api.py`); no real data or network.
- `cd pipeline && uv run pytest` stays green; all tests go in `pipeline/tests/test_editor_api.py`.
- Commit messages end with:
  ```
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01UjTgribg1ST4QF8gSMrmVR
  ```

## Review Focus

1. **A seed far from the truth** (points placed on the wrong blocks) can drive refine to a degenerate transform. Expected: 422, nothing written. Test: `test_autofill_degenerate_is_422_and_writes_nothing` (Task 1).
2. **`autofill: true` sent after `autofill.json` was deleted** by a save, or never existed. Expected: 409, not a silent score against `refine.json`. Test: `test_preview_autofill_without_file_is_409` (Task 1).
3. **Disabled points among the posted set.** They don't count toward the 4-point minimum, and they are kept in the returned set, still disabled. Test: `test_autofill_counts_enabled_points_only` (Task 1).
4. **Saving an auto-fill, then a later automatic run with `force_auto`.** The automatic chain must reproduce the promoted `refine.json` and not crash on the hand-made `rough.json`. Test: `test_force_auto_after_autofill_save_reproduces_refine` (Task 2).
5. **Saving with `autofill: false` after an auto-fill was abandoned.** Expected: no promotion; `refine.json` unchanged. Test: `test_save_without_autofill_flag_does_not_promote` (Task 2).

---

### Task 1: Auto-fill endpoint and auto-fill-aware preview

**Files:**
- Modify: `pipeline/src/sfzmaps/editor/server.py`
  - module docstring: endpoint list; "5 ft / 12 ft" → "7 ft / 15 ft (config.PASS_MEDIAN_FT / PASS_P95_FT)";
  - `PreviewBody`;
  - `_score_new`;
  - the `preview` route;
  - the new route.
- Test: `pipeline/tests/test_editor_api.py`

**Interfaces:**
- Produces:
  - `AUTOFILL_MIN_POINTS: int = 4`.
  - `PreviewBody.autofill: bool = False`. `SaveBody` inherits it.
  - `_autofill_path(d: Path) -> Path` returns `d / "autofill.json"`.
  - `_score_new(d: Path, t: SheetTransform, change_areas: dict | None, autofill: bool = False) -> SheetScore`.
    - With `autofill=True`: 409 `HTTPException` if `autofill.json` is missing. Otherwise it scores with `runner.score_review` against the file's `result` member.
    - Adapter: write `result` to a temp `*.tmp.json` in `d`, pass that path, and delete it in `finally`. Or any equivalent; this is left to the builder.
  - `POST /api/sheet/{series}/{sheet_id}/autofill`, body `PreviewBody`.
    - Response keys: `gcps` (list of GCP dicts with `residual_ft`), `residuals`, `matches`, `n_matches`, `score`, `preview_png_url`, `preview_bounds`, `preview_size`.
  - The `autofill.json` schema:
    ```
    {"seed":   runner._transform_json(seed, "seed"),
     "raw":    runner._result_json(raw),
     "result": runner._result_json(res),
     "n_matches": int}
    ```
    `n_matches` is the count of `res.matches` with `label == "inlier"`.

- [ ] **Step 1: Write the failing tests**
  - Seed helper: `_seed(demo, n=6)`. It returns `_gcps(demo.true_gcps[:: max(1, len(demo.true_gcps)//n)][:n])`, with every `source` set to `"manual"`.
  - `test_autofill_returns_dense_set_keeping_manual_points`: POST `/autofill` with the 6 seed points. Assert:
    - 200;
    - `len(body["gcps"]) > 25`;
    - exactly 6 returned GCPs have `source == "manual"`, with the posted `px`/`py`;
    - `body["n_matches"] >= 40`;
    - `body["score"]["passed"] is True`;
    - `autofill.json` exists with keys `seed`, `raw`, `result` and `n_matches`;
    - `refine.json`, `transform.json` and `rough.json` (if present) have the same mtime as before the call (capture `stat().st_mtime_ns` first).
  - `test_autofill_needs_four_enabled_points`: 3 seed points → 400, and `"4"` is in `r.json()["detail"]`.
  - `test_autofill_counts_enabled_points_only`: 5 seed points, 2 of them `enabled=False` → 400. With 6 points, 1 disabled → 200, and the disabled one comes back with `enabled is False` and source `manual`.
  - `test_autofill_without_blocks_is_409`: delete `blocks_px.geojson` → 409, and `"run the pipeline"` is in the detail.
  - `test_autofill_degenerate_is_422_and_writes_nothing`: monkeypatch `server.refine_mod.refine` to return a `RefineResult` whose transform maps the page to a 1-ft square:
    - 4 GCPs: page corners → `(0,0)`, `(1,0)`, `(0,1)`, `(1,1)` ft;
    - assert 422 and that `autofill.json` does not exist.
  - `test_preview_autofill_scores_against_autofill_json`: after a successful auto-fill, POST `/preview` with the returned `gcps` and `autofill: true`.
    - Assert the score's `n_scored` equals the auto-fill response's `score["n_scored"]`.
    - Then delete `refine.json` and repeat the `autofill: true` preview. Assert it's still 200 with the same `n_scored`. This proves the score comes from `autofill.json`, not `refine.json`.
  - `test_preview_autofill_without_file_is_409`: `autofill: true` with no `autofill.json` → 409.

- [ ] **Step 2: Run the tests to verify they fail**
  - Run: `cd pipeline && uv run pytest tests/test_editor_api.py -k autofill -v`
  - Expected: FAIL (404 on `/autofill`, or `autofill` ignored).

- [ ] **Step 3: Implement**
  - Import `from sfzmaps import refine as refine_mod` at module level, so the test can monkeypatch `server.refine_mod.refine`.
  - The route:
    1. Count enabled non-auto posted GCPs; fewer than `AUTOFILL_MIN_POINTS` → 400 `"auto-fill needs at least 4 enabled points of your own (auto points don't count)"`.
    2. `blocks_px.geojson` missing → 409 `"no block corners for this sheet; run the pipeline for it first"`.
    3. `seed = _transform_of(` the enabled non-auto posted GCPs `)`; auto points never seed.
    4. Corners: from `blocks_px.geojson`, exactly as `runner.run_sheet.do_refine` builds them (`None` → NaN, reshape `(-1, 2)`).
    5. `raw = refine_mod.refine(corners, _get_ref().corners, seed)`, then `res = refine_mod.classify_changes(raw)`.
    6. `res.transform.check_warpable(runner.page_shape(d / "page.png"))`. On `RuntimeError` → 422 with the message.
    7. Merge: `[GCP dicts of res.transform.gcps]` + `[posted GCPs whose source != "auto"]`.
    8. `t = _transform_of(merged)` and render the preview with `render_preview(d, t)`.
    9. Write `autofill.json` with `runner._write_json`.
    10. Score with `_score_new(d, t, body.change_areas, autofill=True)`.
  - Don't hold `_save_lock`.
  - `preview` passes `body.autofill` through to `_score_new`.

- [ ] **Step 4: Run the tests to verify they pass**
  - Run: `cd pipeline && uv run pytest tests/test_editor_api.py -v`
  - Expected: all PASS, the existing tests included.

- [ ] **Step 5: Commit**
  - `git commit -m "feat(editor): auto-fill endpoint seeds refine from the reviewer's points"`

### Task 2: Save promotes the auto-fill

**Files:**
- Modify: `pipeline/src/sfzmaps/editor/server.py` (`save` route)
- Test: `pipeline/tests/test_editor_api.py`

**Interfaces:**
- Consumes: from Task 1, `_autofill_path`, the `autofill.json` schema, `_score_new(..., autofill=)`, and `SaveBody.autofill`.
- Produces:
  - `_promote_autofill(d: Path) -> dict`. It writes these files, in this order, each with `runner._write_json`:
    1. `rough.json` = `{"transform": seed, "scale_hint_ft_per_px": None, "seeded_from": "editor autofill"}`;
    2. `refine_raw.json` = `raw`;
    3. `refine.json` = `result`.

    It returns the loaded `autofill.json` dict. The caller deletes `autofill.json` after a successful `run_sheet`.
  - `review.json` gains `"autofill": {"seed_points": <posted GCPs with source != "auto" and enabled>, "matches": n_matches}`, only when `body.autofill`.

- [ ] **Step 1: Write the failing tests**
  - `test_save_autofill_promotes_and_matches_preview`:
    1. Auto-fill with 6 seed points.
    2. Save with the returned `gcps`, `autofill: true` and status `edited`.
    3. Assert:
       - 200, and `body["score"]["passed"] is True`;
       - `body["score"]["holdout_p50_ft"] == pytest.approx(body["preview_score"]["holdout_p50_ft"])`;
       - `refine.json` equals `autofill.json`'s former `result` (load it before the save);
       - `refine_raw.json` equals `raw`;
       - `rough.json["seeded_from"] == "editor autofill"`;
       - the mtime order is `rough.json` ≤ `refine_raw.json` ≤ `refine.json`;
       - `autofill.json` is gone;
       - `review.json["autofill"] == {"seed_points": 6, "matches": <n_matches>}`;
       - `review.json["smoothing"] == config.TPS_SMOOTHING`.
  - `test_save_without_autofill_flag_does_not_promote`:
    1. Auto-fill, then save with `autofill: false` and the demo's `true_gcps`.
    2. Assert `refine.json` bytes are unchanged from before the auto-fill, `"autofill" not in review.json`, and `autofill.json` still exists.
  - `test_force_auto_after_autofill_save_reproduces_refine`:
    1. Save an auto-fill as above.
    2. Run `runner.run_sheet(demo.page, demo.ref, force_auto=True)`.
    3. Assert no exception, `transform.json["origin"] == "auto"`, and that the `matches` in `refine.json` equal the promoted `result["matches"]`.

- [ ] **Step 2: Run the tests to verify they fail**
  - Run: `cd pipeline && uv run pytest tests/test_editor_api.py -k "save_autofill or does_not_promote or force_auto_after" -v`
  - Expected: FAIL.

- [ ] **Step 3: Implement**
  - In `save`, score with `_score_new(d, t, body.change_areas, autofill=body.autofill)`. This gives a 409 if the flag is set and the file is missing.
  - Inside `_save_lock` and before writing `review.json`:
    - if `body.autofill`, call `_promote_autofill(d)`;
    - add the `autofill` field to `review`.
  - After `run_sheet` returns, unlink `autofill.json` (`missing_ok=True`).

- [ ] **Step 4: Run the tests to verify they pass**
  - Run: `cd pipeline && uv run pytest -q`
  - Expected: all PASS (the whole pipeline suite).

- [ ] **Step 5: Commit**
  - `git commit -m "feat(editor): saving an auto-fill promotes it into the sheet's refine artifacts"`

### Task 3: Editor button, the 7 / 15 ft labels, and docs

**Files:**
- Modify:
  - `pipeline/src/sfzmaps/editor/static/index.html`
  - `pipeline/src/sfzmaps/editor/static/editor.js`
  - `pipeline/README.md` ("Reviewing and editing")
- Test: `pipeline/tests/test_editor_api.py`

**Interfaces:**
- Consumes: from Task 1, the `/autofill` response; from Tasks 1 and 2, the `autofill` body flag on preview and save.

- [ ] **Step 1: Write the failing test**
  - `test_front_end_has_autofill_and_current_pass_bar`: read the static files from disk (`Path(server.__file__).parent / "static"`). Assert:
    - `id="tool-autofill"` is in `index.html`;
    - the text `Auto-fill from my points` is in `index.html`;
    - `≤ 7 ft` and `≤ 15 ft` are in `index.html`, and `≤ 5 ft` and `≤ 12 ft` are not;
    - the legend reads `&lt; 7 ft` and `&lt; 15 ft`;
    - in `editor.js`: `/autofill` and `autofill: S.autofill` appear, `PASS_P50 = 7` and `PASS_P95 = 15` appear, and `above 5 ft` does not.

- [ ] **Step 2: Run the test to verify it fails**
  - Run: `cd pipeline && uv run pytest tests/test_editor_api.py -k front_end_has_autofill -v`
  - Expected: FAIL.

- [ ] **Step 3: Implement**
  - **`index.html`:**
    - add `<button id="tool-autofill" title="Seed the automatic matcher from your points">Auto-fill from my points</button>` after `#tool-area`;
    - change the header spans to `≤ 7 ft` / `≤ 15 ft` and the legend to `&lt; 7 ft` / `&lt; 15 ft`.
  - **`editor.js` constants:** `TOL_GREEN = 7, TOL_AMBER = 15`, `PASS_P50 = 7, PASS_P95 = 15`. Change the "above 5 ft" toast text to use `PASS_P50`.
  - **`editor.js` state:**
    - add `autofill: false` to `S`;
    - include `S.autofill` in each `pushHistory` entry and restore it in `undo`;
    - reset it to `false` in `openSheet`;
    - send `autofill: S.autofill` in the preview and save bodies.
  - **`editor.js` button:**
    - The button is enabled only when at least 4 points are enabled. Update it wherever the point list changes (`refreshStyles` is called after every edit). Its title explains why when disabled.
    - Click: set `$("busy")` to `"Auto-filling…"` and disable the button. Then `pushHistory()` and POST `/autofill` with `{gcps: cleanGcps(), change_areas: areasFC()}`.
    - On success:
      - replace `S.gcps` with the returned gcps, carrying `_r` from `residual_ft`;
      - set `S.autofill = true` and `S.score = r.score`;
      - call `setOverlay(r)`, `rebuildMarkers()`, `renderScore()`, `edited()` / `updateDirty`;
      - toast `` `Auto-fill: ${r.n_matches} matched corners. Review, then Save.` ``.
    - On error: pop the pushed history entry and toast the server's message.
    - Always clear the busy text.
  - **`pipeline/README.md`:** add a numbered-list item under "Reviewing and editing" covering:
    - the button and that it needs 4 or more points;
    - that it never saves;
    - that saving it fits with `TPS_SMOOTHING` and records `autofill` in `review.json`.

- [ ] **Step 4: Run the tests to verify they pass**
  - Run: `cd pipeline && uv run pytest -q`
  - Expected: all PASS.

- [ ] **Step 5: Check the UI by hand**
  - Run `uv run sfzmaps edit --demo --port 8799` in the background. Load `http://127.0.0.1:8799` with the browser tools and confirm the button renders. Disable points down to 3 and confirm the button disables.
  - Re-enable the points, click it, and confirm the toast, the score and the point count change. Undo and confirm the original points return.
  - Stop the server.
  - If browser tools are unavailable, say so in the report and drive the same flow through the API with curl.

- [ ] **Step 6: Commit**
  - `git commit -m "feat(editor): Auto-fill button; pass bar shown as 7 / 15 ft"`
