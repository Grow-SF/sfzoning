# Editor: auto-fill from my points

Status: design, written autonomously (the owner asked for brainstorm → plan →
subagent-driven development → pull request without check-ins). Every decision
below that the owner did not state is marked **(my call)**.

## Why

The scorer only passes a sheet with at least 40 measured corners
(`score.score(min_scored=40)`). A reviewer who hand-places a dozen good points
gets an accurate fit that still fails with `too_few_matches`, and placing 40+
points by hand is tedious. The owner said so directly: "that's so many
points... why don't YOU add points".

On 2026-10-31 this was done by hand for 1960 sheet 5: the owner's 12 points
became the starting transform of the automatic refine. That gave 1,568
matches, median 2.8 ft, p95 6.5 ft, a pass. The one-off script is
`seed_from_review.py` (scratchpad, not in the repo). This feature puts that
step in the editor as a button.

## Goal

In the control-point editor, a reviewer places a few points, clicks
**Auto-fill from my points**, and gets a dense, automatically matched point
set seeded from their placement. They see the preview and score, and save
as usual.

## Non-goals

- No change to the automatic pipeline (`sfzmaps run`), the pass bar, or the
  cross-edition matcher.
- No CLI command for it **(my call: YAGNI; the editor is where hand points
  exist)**.
- No new vision or API calls. Refine is local geometry only.

## Anti-goals

- The button must never save anything. The reviewer still decides, as with
  every other edit.
- It must not silently throw away the reviewer's points. Their manual points
  stay in the set, and one Undo restores exactly what they had.
- Abandoning an auto-fill (closing the tab, switching sheet) must leave the
  pipeline's automatic artifacts untouched.

## Behavior

### Server: `POST /api/sheet/{series}/{sheet_id}/autofill`

- **Body:** the same as preview: `{gcps, change_areas}`.
- **400** with a plain message when fewer than `AUTOFILL_MIN_POINTS = 4`
  points are enabled **(my call: three fix an affine, and a fourth gives one
  check)**.
- **409** when the sheet has no `blocks_px.geojson`; the message says to run
  the pipeline for that sheet first.
- **Seed:** `_transform_of(body.gcps)`, the same fit preview uses.
- **Refine:** called in memory, exactly as the runner calls it:
  - corners from `blocks_px.geojson`, with `None` turned into NaN, as in
    `runner.do_refine`;
  - `raw = refine.refine(corners, ref.corners, seed)`;
  - `res = refine.classify_changes(raw)`.
- **The reviewer's points stay.** `refine._fit` only adds the seed's
  non-`auto` points when fewer than 10 corners matched, so a dense result
  drops them. The returned set is therefore the result transform's `auto`
  GCPs plus every posted GCP whose source is not `auto` (`manual`, `anchor`,
  `answer_key`), with their source and enabled flag kept. Posted `auto`
  points, from an earlier automatic fit, are replaced by the new ones
  **(my call: they came from a worse fit and the new run supersedes them)**.
- **422**, with nothing written, when `res.transform` fails
  `SheetTransform.check_warpable`.
  - Every other outcome is returned with its score: not converged, a weak
    vote, or too few matches. The reviewer judges it in the preview, as with
    any edit **(my call: the vote status is measured after the seed is
    applied, so it isn't a quality test)**.
- **On success:** writes `autofill.json` in the sheet's work dir, atomically
  (`runner._write_json`):
  - `seed`: `runner._transform_json(seed, "seed")`;
  - `raw`: `runner._result_json(raw)`;
  - `result`: `runner._result_json(res)`.
  - It never touches `rough.json`, `refine_raw.json`, `refine.json`,
    `changes.geojson` or `transform.json`.
- **Returns:**
  - `gcps` (the merged set) and `residuals` under the editor's fit of that
    set (`_transform_of`, smoothing by `runner.smoothing_for`);
  - `matches` (`result`'s);
  - `n_matches` (inlier count);
  - `score` (below);
  - the preview overlay fields (`preview_png_url`, `preview_bounds`,
    `preview_size`), rendered from that fit.
- **Locking:** no lock is held while refine runs (tens of seconds). The
  final file write is atomic.

### Scoring with the auto-fill's matches

- Preview and save bodies gain `autofill: bool = false`.
- When it is true:
  - scoring uses `runner.score_review(t, d / "autofill.json"…)`, which reads
    the `result` member (a small loader adapter is fine);
  - **409** if `autofill.json` is missing.
- The editor's change areas are unchanged by auto-fill. They stay the
  reviewer's drawn polygons. Auto-detected real change is already carried
  as match labels in `result`, and `score_review` excludes those matches.

### Save with `autofill: true`

- Under the existing `_save_lock`, before writing `review.json`, promote
  `autofill.json` into the runner's artifacts, in this order so each is
  newer than its inputs:
  1. `rough.json` = `{"transform": seed, "scale_hint_ft_per_px": null,
     "seeded_from": "editor autofill"}`;
  2. `refine_raw.json` = `raw`;
  3. `refine.json` = `result`.
- Then save as today: `review.json`, rerun the sheet, then rebuild.
  - The runner's review branch writes `changes.geojson` from the review's
    change areas (existing behavior, kept).
  - It scores with `refine.json`, so the saved score equals the preview.
- **Why promote all three:** the automatic chain stays coherent. A later
  `force_auto` run finds `rough.json` and `refine_raw.json` fresh. It reruns
  only `changes` (`transform.json`'s origin is `review`), which reproduces
  `refine.json` from `refine_raw.json`. So "automatic" then means "automatic
  from this seed", which is what the 1960 sheet 5 fix did.
- Delete `autofill.json` after a successful promotion.
- `review.json` gains `"autofill": {"seed_points": <enabled posted points>,
  "matches": <n_matches>}`.
- Smoothing comes from `runner.smoothing_for` as today. The merged set has
  hundreds of points, so this is `TPS_SMOOTHING`.

### Editor UI

- **The button:** "Auto-fill from my points" in the toolbar, after
  "Mark change area". It's disabled when fewer than 4 points are enabled,
  with a title explaining why.
- **While it runs:** the busy indicator reads "Auto-filling…", and the
  button is disabled.
- **On success:**
  - pushes the current state to the undo history and replaces the point
    list with the returned set;
  - sets `S.autofill = true` and shows the returned score and overlay;
  - toasts "Auto-fill: N matched corners. Review, then Save."
- **Later edits** keep `S.autofill = true`, and every preview or save sends
  `autofill: S.autofill`.
- **Undo** to the state before the auto-fill clears it: the history entry
  stores the flag.
- **Opening a sheet** resets it.
- **On failure:** a toast with the server's message; the points are unchanged.

### Pass-bar labels (bundled fix)

The bar is 7 / 15 ft (`config.PASS_MEDIAN_FT`, `PASS_P95_FT`). `editor.js`
already renders limits from the score (`pass_median_limit_ft`,
`pass_p95_limit_ft`); only the stale constants remain. Fix:
- the fallback constants in `editor.js` (`PASS_P50 = 5, PASS_P95 = 12`);
- the hard-coded "above 5 ft" toast text in `editor.js`;
- the static "≤ 5 ft / ≤ 12 ft" spans and the "< 5 ft / < 12 ft" legend in
  `index.html`, changed to 7 / 15;
- the server docstring's 5 / 12.

There's no new payload field **(my call: the score already carries the
limits)**.

## Constraints

- Python 3.12, FastAPI, the existing editor patterns. Vanilla JS in
  `editor.js`; no new dependencies.
- The editor stays bound to 127.0.0.1.
- Tests use the existing synthetic demo sheet (`editor/demo.py`, fixture
  `env` in `tests/test_editor_api.py`), with no real data or network.
  `cd pipeline && uv run pytest` must stay green.
- `pipeline/README.md`, "Reviewing and editing", documents the button.

## Left to the builder

- Exact JSON field order, helper names, and CSS for the button.
- How `score_review` is pointed at `autofill.json`'s `result` member: a
  small adapter or a temp file are both fine.
