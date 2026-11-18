# Task 2 report: the text-vs-line gate

Status: DONE_WITH_CONCERNS. One commit on `solver`: 1bba63c (BASE 8157a97).

## Built
- `pipeline/src/sfzmaps/solver/textnet/linegate.py`:
  - `ChanceSource`, `folder_source(folder, "flat" | "nested")`. A missing file raises FileNotFoundError naming the
    sheet. A file that is not 8-bit gray raises ValueError. The gate checks the page's shape and raises ValueError
    naming the sheet, so it never scores fewer px.
  - `Geo` (Protocol): `quarter`, `block`, `border_weight`. Optional: `answered`, `weight_table`, `block_key`.
  - `CityGeo`: the real geo. Page px go to their map point by `data._world_at`, in bands of `data.BAND` rows. Then
    `grid_px`, which is `grid.to_grid` and a floor.
    - The quarter comes from `grid.quarter_of`, at every px.
    - At ink px only, it gives the block: `C.grown_raster`'s lot, then that lot's `block_num`, numbered in sorted
      order.
    - Also at ink px only, it gives w(d).
  - The weight maths, as pure functions: `ink_shares`, `excess_weight`, `weight_table`, `district_weight`.
  - `gate_sheets(sheets, source, quarters, geo, boot, seed) -> (result, tallies)`. Its keys:
    - `m1a`, `m1a_in_box`, `m1a_near`, `m1a_far`;
    - `m1b_in_box`, `m1b_near`, `m1b_far`, `m1b_by_point` {0-30, 30-100, 100+};
    - `m1c`;
    - `m2`, `m2_by_kind` (all four kinds), `m2_answered`, `overlap_band`;
    - `rules`, `m1a_weight` (the w table), `targets`, `block_key` and the run's settings.
    - Each measure gives {px, called_05, called_09, mean, log_loss, ci_05}. M1a's px are Σw.
  - `gate(source, quarters, geo=None, boot, seed)`. It loads the 13 sheets (`paths.check_sheets`,
    `data.load_sheets`) and uses CityGeo by default. If `data/solver/experiments/textlines/rules.json` exists, it
    adds the learned font's rule table.
  - `write(result, tallies, out)` writes:
    - `scores.json`;
    - `scores.md`: the measures table, the w(d) table and the run rule's accuracy for putText and font.npz;
    - `blocks.npz`: `<measure>.{quarter, block, px, called_05, called_09, chance, log_loss}`, all sums float64, and
      `block_key`.
  - `paired(a, b, measure, boot=1000, seed=0, stat="called_05")`:
    - it returns {diff = b − a, lo, hi, a, b, blocks, …};
    - each draw resamples the blocks once, the same draw for both arms;
    - each arm's value is the mean over its runs of the ratio of sums.
- `pipeline/tests/test_textnet_linegate.py`: 10 tests.
  - The brief's 5 tests.
  - A part-covering chance stops the gate naming the sheet (Review Focus 1).
  - Only the asked quarters are scored.
  - M1a weighs px by w.
  - The sign of `paired`: b − a = −1 when b calls nothing.
  - The weight maths on drawn profiles and cracks: the ±d averaging, the excess formula, the nearest class's table,
    and 0 past J.
  - The real-data round trip, which is skipped without `pipeline/data`.
  - The `crossed_word` fixture is "RH-1" drawn with putText, with a 4-px line across the page through the letters.
    It asserts that the run rule's M2 base equals the base from its own masks; at this line row it does.
- No edits to `data.py`, `train.py` or any other file. `linegate` is in no stage's import closure: test_build passes,
  and dvc.yaml is unchanged.

## TDD
- RED: `uv run pytest -q tests/test_textnet_linegate.py` → `ImportError: cannot import name 'linegate'`, 1 error.
- GREEN: the same command → `10 passed` (the real-data test ran locally).
- The iterate set (`test_solver_textnet`, `test_codereader`, `test_solver_textread`, `test_build`,
  `test_textnet_*`), `-n 4`: 169 passed, 3 xfailed.
- Full suite `uv run pytest -q -n 3`: 1477 passed, 3 xfailed, before the commit.
- `pgrep -fl 'sfzmaps (build|stage)'` was empty before I created the src file and before the commit.

## The w(d) table (all 2000 cracks; d in grid px of 4 ft; J = 20)
Cracks counted: touching lots, 363,745 district cracks and 4,648,072 other both-known cracks. Across a street,
470,339 and 2,506,980. These match the red team's counts.

| d | touch f_D | touch f_L | touch w | across f_D | across f_L | across w |
|---|---|---|---|---|---|---|
| 0 | 0.894 | 0.153 | 0.828 | 0.816 | 0.216 | 0.735 |
| 1 | 0.874 | 0.153 | 0.825 | 0.808 | 0.217 | 0.732 |
| 2 | 0.789 | 0.152 | 0.808 | 0.752 | 0.220 | 0.708 |
| 3 | 0.616 | 0.153 | 0.752 | 0.593 | 0.225 | 0.620 |
| 4 | 0.429 | 0.157 | 0.633 | 0.397 | 0.230 | 0.421 |
| 5 | 0.291 | 0.165 | 0.432 | 0.274 | 0.240 | 0.127 |
| 6 | 0.229 | 0.173 | 0.245 | 0.264 | 0.263 | 0.005 |
| 7 | 0.203 | 0.174 | 0.141 | 0.321 | 0.307 | 0.041 |
| 8 | 0.191 | 0.172 | 0.097 | 0.408 | 0.352 | 0.137 |
| 9 | 0.183 | 0.167 | 0.086 | 0.447 | 0.358 | 0.199 |
| 10 | 0.179 | 0.166 | 0.074 | 0.433 | 0.331 | 0.235 |
| 11 | 0.180 | 0.169 | 0.061 | 0.349 | 0.279 | 0.201 |
| 12 | 0.183 | 0.174 | 0.045 | 0.282 | 0.242 | 0.143 |
| 13 | 0.189 | 0.179 | 0.049 | 0.228 | 0.220 | 0.037 |
| 14-20 | 0.19-0.22 | 0.17-0.19 | 0.07-0.15 | 0.20-0.21 | 0.21 | 0.000 |

f_D and f_L at d = 0, 2, 3, 4, 6, 8 match the red team's `prof_check.py` numbers exactly.

How d is measured: each district crack is drawn at the px where `prof` reads offset 0, the floor of the crack's
middle. A Euclidean distance transform over that, rounded, gives d. So on a straight border, the px at |offset| d are
at distance d. Where the touching and across classes tie, the touching table is used.

## Round trip on real data (page → grid by CityGeo's path → page by `page.sheet_px` / `to_pixel`)
- The test sheet (p02): all 10,000 random ink px land within 1 px. 93.7% land at 0 px and 6.3% at 1 px. PASS.
- All 13 sheets, 10,000 on-grid ink px each (scratch `roundtrip.py`):
  - on p02, p03, p04, p07 and p11, the largest error is 1 px;
  - on the others, a few px land more than 1 px away: from 0.01% to 0.6% of px, at most 7 px, on p06.
- Those misses are almost all outside the printed frame (scratch `roundtrip_outliers.py`):

  | sheet | ink px in the frame | in the frame, > 1 px off | outside the frame, > 1 px off |
  |---|---|---|---|
  | p06 | 7,489 | 0 | 62 |
  | p10 | 8,443 | 1 (2 px off) | 29 |
  | p01 | 7,165 | 0 | 6 |

  The page margins sit near the bottom title block. There the 32-px placement samples differ from the exact
  transform by 5-17 ft (the exact `to_world`/`to_pixel` round trip is 0).

## Run time and smoke checks ($0, CPU, one thread)
- `gate` on all 13 sheets and four quarters takes 63.0 s for the held-out detectors (`data/solver/text-reader/`,
  nested). It takes 58.7 s for the all-sheets detector (`data/solver/text/chance/`, flat).
  - That time covers `border_paths` (about 15 s), the weight grid, `load_sheets` and the rule accuracy (about 5 s).
  - Peak RSS is 6.96 GB per process.
  - The outputs are in `pipeline/data/solver/experiments/textlines/task2-smoke/{r328,all-sheets}/`, not committed.
- The all-sheets detector:
  - m1b_in_box called_05 is 0.804 [0.671, 0.879]. That is near the red team's ~0.82, as Task 3 expects.
  - m1a_in_box is 0.677, m1a_far 0.012, m1c 0.036 and m2 0.983.
- The held-out detectors (r328):
  - m1b_in_box 0.975, m1a_in_box 0.990, m1a_far 0.464, m1c 0.366, m2 0.999.
  - These are high everywhere near labels, which fits `detector-overcall/results.md`: the 328-px discs hold 95% of
    the ink outside words.
- `paired(r328, all-sheets)`:
  - m1a_in_box −0.313 [−0.329, −0.296] over 2,729 blocks;
  - m1b_in_box −0.170 [−0.290, −0.103];
  - m2 −0.015 [−0.023, −0.011];
  - about 0.03 s each.
- The owner-answered labels: 668 words, every one an exact Terra word on its sheet. m2_answered has 448,244 px.

## Decisions beyond the brief
1. **Block and weight only inside the sheet's crop on the merged page.**
   - The crop is the frame quad in `data/solver/page/frames.json`, minus the cut boxes (`page.crop_mask`).
   - Elsewhere the block is −1 and w is 0: the margins and notes are not the grid ink w was measured on, and the
     round trip is poor there.
   - The quarter still covers every px, so M1b, M1c and M2 keep the margins (no exclusions, F9).
   - Sheets not on the merged page (other series) get block −1 and w 0 everywhere, so Task 5's 1990-96 sheets
     report no M1a.
2. **`m2_answered`** is M2's base inside the words of labels an owner's answer joined (`codereader.labels`,
   `held_out` with a Terra box). It reaches the gate through an optional `Geo.answered`.
   - `CityGeo.answered` imports `codereader.labels` lazily. That is outside the "may import" list. `linegate` is in
     no stage closure.
   - With a geo that lacks `answered`, `m2_answered` is None ("not measured").
3. **`gate` returns `(result, tallies)`,** like `gate_sheets`, not a bare dict: `write` needs both.
4. **`paired`'s unit is (quarter, block).** It uses the blocks with px in every run of both arms, so an NE-only run
   against an all-quarter baseline is compared on NE blocks only. It takes an optional `stat` (default called_05).
   diff = b − a: negative means the second arm calls less text.
5. **Log-loss targets:** not text for M1a/b/c; text for M2 and the overlap band. The overlap band mixes line and glyph
   ink, so its log loss has no right target; `scores.md` says "text".
6. **px off the grid or outside the crop** form one bootstrap unit per quarter (block −1). That is conservative.

## Concerns
1. **M1a-in-box is not line ink alone, so a perfect detector does not score 0** (scratch `m1a_parts.py`, all 13
   sheets).
   - 31% of M1a-in-box's weight comes from d ≥ 7: the profile's tail, not the line's own ink. Touching w stays at
     0.05-0.15 out to d = 20. Across a street, w rises again to 0.14-0.24 at d = 8-12, where the block edges are.
   - 44% of the in-box weight sits on M2's base: ink clear of the run rule's lines. That is glyph ink, plus the
     stair-step district lines the run rule misses (recall 8%).
   - Over all of M1a, 80% of the weight is at d ≤ 3 and 9% at d ≥ 7.
   - So paired differences point the right way, but they are diluted.
   - Options for the controller: fit J where w meets noise, or report M1a by d band. Both are measured choices; the
     memory rules out guessed cutoffs.
2. **The round trip passes on p02 (the brief's test) but not on every sheet over all ink.** The misses are almost
   all outside the frame, where M1a now reads nothing (decision 1). In the frame I saw one miss of 2 px in about
   23,000 px on three sheets.
3. **Memory is about 7 GB per gate process.** Four gates at once would need about 28 GB of the 64 GB.
4. **The fixture depends on where Hershey draws the glyphs.** The line row (10 below the glyphs' top) was chosen by
   a search where the run rule's M2 base is exact. The fixture asserts this, so a change in OpenCV's rendering fails
   loudly; it does not drift silently.
5. **The learned-font table comes from `rules.json`**, which a script writes. `gate` reads it if it is there; it is
   not recomputed (the font needs `codereader.glyphs`).

## Files
- `/Users/sbuss/workspace/sfzoning/.claude/worktrees/saved-reads/pipeline/src/sfzmaps/solver/textnet/linegate.py`
- `/Users/sbuss/workspace/sfzoning/.claude/worktrees/saved-reads/pipeline/tests/test_textnet_linegate.py`
- Smoke outputs (not committed):
  `/Users/sbuss/workspace/sfzoning/.claude/worktrees/saved-reads/pipeline/data/solver/experiments/textlines/task2-smoke/`
- Scratch checks:
  - `/private/tmp/claude-501/-Users-sbuss-workspace-sfzoning/eb18d2fc-e22a-48a6-afbc-57c31cd3e134/scratchpad/t2/roundtrip.py`
  - `/private/tmp/claude-501/-Users-sbuss-workspace-sfzoning/eb18d2fc-e22a-48a6-afbc-57c31cd3e134/scratchpad/t2/roundtrip_outliers.py`
  - `/private/tmp/claude-501/-Users-sbuss-workspace-sfzoning/eb18d2fc-e22a-48a6-afbc-57c31cd3e134/scratchpad/t2/m1a_parts.py`
  - `/private/tmp/claude-501/-Users-sbuss-workspace-sfzoning/eb18d2fc-e22a-48a6-afbc-57c31cd3e134/scratchpad/t2/answered_check.py`
  - `/private/tmp/claude-501/-Users-sbuss-workspace-sfzoning/eb18d2fc-e22a-48a6-afbc-57c31cd3e134/scratchpad/t2/smoke.py`
  - `/private/tmp/claude-501/-Users-sbuss-workspace-sfzoning/eb18d2fc-e22a-48a6-afbc-57c31cd3e134/scratchpad/t2/smoke_all.py`

---

# Fix round 1 (commit 7fc310d)

## What changed
1. **M1a's weight ends where the line's own excess ends.** `weight_cut(excess) -> (cut, far_level)` sets the cut per
   class:
   - the far-field level is the median excess e(d) = f_D(d) − f_L(d) over the outer third of d = 0..J (the last
     ceil((J+1)/3) values, d = 14..20);
   - the cut is the first d past e's peak at which e ≤ the far-field level, or at which e stops falling
     (e(d+1) > e(d));
   - w is unchanged below the cut and 0 from it on.
   - The rule is stated in the module docstring, in `weight_cut`'s docstring and in `scores.md`.
   - **The "stops falling" clause goes beyond the ruling's literal rule; it was needed.** The across-street far-field
     level is negative (−0.0111), so e falls to it only at d = 17, past the bump, and the literal rule would have
     kept the bump. The excess bottoms out at d = 6 (+0.0014) before the block outlines' bump (rising to +0.10 at
     d = 10). A rise after the peak means another feature's ink, so the weight ends there. The rule has no tuned
     constant.
2. **The weight beyond the cut is reported apart.** `m1a_context` and `m1a_in_box_context` use the same w formula
   for cut ≤ d ≤ J. They are a diagnostic, marked "in no verdict" in `scores.md`. `m1a`, `m1a_in_box`, `m1a_near` and
   `m1a_far` are now the core.
   - The geo supplies it through an optional `Geo.context_weight(sd)`. With a geo that lacks it, the context keys
     are None.
   - `CityGeo` now keeps the grid distance d (uint8) and the nearest crack's class (bool) instead of a float weight
     grid. It looks both tables up at each sheet's ink px (`district_distance`, `weight_at`; `district_weight` is
     their composition).
3. **`scores.md` has a "What each measure misses" section**, one line per family:
   - M1a: weight 0 on the margins, cut boxes and other series, and beyond the cut. Ink within the cut is weighted by
     the city average, so label ink touching a border carries some weight.
   - m1a_context: a diagnostic.
   - M1b: stair-step 0.08 and dashed 0.02 recall; a few glyph px called line.
   - M1c: counts as not-text the other passes' boxes (`sd.other`), the text Terra missed near Sol points, and the
     margin text.
   - M2: counts as text the stair-step and dashed lines inside boxes; leaves out glyph ink within 1 px of a run-rule
     line.
   - The overlap band: its target is mixed.

   The lines are also in `scores.json` under `misses`.
4. **`paired` raises ValueError when px differ** (`np.allclose`) between runs on any common (quarter, block). The
   message names both folders and the unit.

## TDD
- RED: the three new tests run against the module as committed in 1bba63c (scratch `red_check.py`) → 3 failed.
  - `weight_cut` was missing (AttributeError).
  - The context measures were missing.
  - `paired` did not raise.
- GREEN: `tests/test_textnet_linegate.py` → 13 passed.
  - New: `test_the_weight_beyond_the_cut_is_reported_apart`.
  - New: `test_paired_stops_on_runs_scored_on_different_px`.
  - New: `test_the_weight_ends_where_the_lines_own_excess_ends`. It covers a fall to the far-field level, a dip
    before a bump, and `weight_table`'s split of w and w_context.
- The iterate set (`-n 3`): 172 passed, 3 xfailed. `test_build` is included.
- Full suite (`-n 4`): 1480 passed, 3 xfailed.
- `pgrep -fl 'sfzmaps (build|stage)'` was empty before the commit.

## Cut points (all 2000 cracks)
- **Touching lots:** far-field level +0.0239; cut at d = 8, so the weight covers d ≤ 7 (w 0.828 … 0.141). The
  context runs over d = 8..20 (w 0.045-0.146).
- **Across a street:** far-field level −0.0111; cut at d = 6, so the weight covers d ≤ 5 (w 0.735 … 0.127). The
  context runs over d = 6..13 (0.005, 0.041, 0.137, 0.199, 0.235, 0.201, 0.143, 0.037), then 0.
- **The across-street bump at d = 8-12 is entirely in the context.** The core w is 0 there, as checked.

## Smoke: both sources in one process, one CityGeo (scratch `smoke2.py`)
- The held-out detectors (r328, nested) took 62.0 s.
- The all-sheets detector (flat) took 34.8 s, because the CityGeo was already built.
- Peak RSS was 7.02 GB.
- The outputs replaced `pipeline/data/solver/experiments/textlines/task2-smoke/{r328,all-sheets}/`.

| measure (called ≥ 0.5) | weighted px | r328 | all-sheets |
|---|---|---|---|
| m1a_in_box (core) | 130,151 | 0.9874 [0.9828, 0.9916] | 0.5794 [0.5581, 0.6002] |
| m1a_in_box_context | 55,198 | 0.9961 [0.9944, 0.9975] | 0.9075 [0.8994, 0.9148] |
| m1a (core) | 3,175,748 | 0.6683 [0.6507, 0.6857] | 0.0462 [0.0435, 0.0489] |
| m1a_context | 310,129 | 0.5884 [0.5649, 0.6077] | 0.1933 [0.1827, 0.2026] |
| m1b_in_box | 1,098,605 | 0.9747 | 0.8045 |
| m2 | 4,121,157 | 0.9986 | 0.9833 |

- Paired (all-sheets − r328) on `m1a_in_box`: −0.408 [−0.429, −0.388] over 1,842 blocks. Before the cut it was
  −0.313; the reviewer measured −0.414 on d ≤ 6.
- `m1a_in_box_context`: −0.089 [−0.097, −0.081].
- The context is called at glyph rates by the all-sheets detector (0.91 in box). That confirms it is mostly label
  ink, now out of the verdict measure.

## Remaining concerns
- **The core within the cut still weights label ink touching a border by the city average.** About 44% of the old
  in-box weight sat on M2's base. Some of that is stair-step line ink, which is M1a's point, so the core's floor for
  a perfect detector is above 0 but much lower than before. `scores.md` states it.
- **The cut rule's "stops falling" clause would cut early if the excess wobbled near its peak.** On these
  360-470 k cracks per class the excess is strictly falling until the cut, so this is not seen.
