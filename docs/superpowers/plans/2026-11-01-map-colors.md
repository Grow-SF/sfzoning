# Map Colours Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recolour the zoning map: a seven-step red-to-green housing scale that includes mixed-use, light and dark blue commercial, brown industrial.

**Architecture:** The pipeline's `category_for` gains the new categories, and rebuilt outputs carry them. The site's palette, legend and chart read the same category names. New outputs publish under new file names, through `site/lib/published.json`.

**Tech Stack:** Python (pipeline, uv, pytest), Next.js, MapLibre, vitest, Playwright.

**Spec:** `docs/superpowers/specs/2026-11-01-map-colors-design.md`

## Global Constraints

- Categories, in this order, everywhere: `single_family, two_unit, three_unit, homes_low, homes_mid, homes_high, homes_top, commercial_light, commercial_dense, industrial, public, unrestricted, unknown`. `multi_family`, `mixed` and `commercial` go away.
- No paid reads: use `sfzmaps districts --cache-only` or `--republish` only.
- Never print or copy `pipeline/.env`, `site/.env.local` or any token. Never commit `pipeline/data/`.
- Never overwrite a published file; publish through `site/scripts/publish-tiles.mjs`.
- vadimg/sfzoning has no license: a cross-check only, copy nothing.
- Test our behaviour only: no trivial assertions, no tests of third-party libraries or MapLibre internals.

## Review Focus

- A code form on an old map (`R-1-D`, `RM3NC3`, `NO BEACH`, `C-3-O(SD)`) lands on the intended step, not `unknown` or a default.
- `MUO`, `MUR`, `MUG`, `UMU` share one research row but differ: MUO is light blue, the rest top green.
- A rule text with a density in it ("1 per 400 sq ft (6 on a 2,500 sq ft lot)", "1 per 3,000") is read by its density and not its parenthetical.
- Unread limits never silently look verified: each is listed by `tests` or a report, and in the PR.
- A deployed site reading the old pointer keeps working after the new files are published.

---

### Task 1: New categories in the pipeline

**Files:**
- Modify: `pipeline/src/sfzmaps/districts/categories.py`, `shared.py`, `assemble.py` (`COLORS`), `today.py` if it names categories
- Test: `pipeline/tests/test_districts_rules.py` and others naming old categories

**Interfaces:**
- Produces: `category_for(code: str, year: int) -> str`, same signature, returning the Global Constraints names; `CATEGORIES` in that order.

- [ ] **Step 1: Write failing table test** `test_map_category_table` with a parametrized list of `(code, year, expected)`. Include exactly:
  `RH-1 2020 single_family; RH-1(D) 2020 single_family; R-1 1960 single_family; 1R 1948 single_family; RH-1(S) 2020 two_unit; RH-2 2020 two_unit; R-2 1960 two_unit; RH-3 2020 three_unit; RM-1 2020 homes_low; RM-2 2020 homes_low; NC-1 2020 homes_low; NC-2 2020 homes_low; NC-3 2020 homes_low; NC-S 2020 homes_low; RC-1 1990 homes_low; RC-2 1990 homes_low; RM-3 2020 homes_mid; R-3 1960 homes_mid; RC-3 2020 homes_mid; C-1 1960 homes_mid; RM-4 2020 homes_high; R-4 1960 homes_high; RC-4 2020 homes_high; SLR 2000 homes_high; R-5 1960 homes_top; 2R 1948 homes_top; NCT-3 2020 homes_top; RTO 2020 homes_top; UMU 2020 homes_top; MUR 2020 homes_top; MUG 2020 homes_top; RH-DTR 2020 homes_top; C-3-O 2020 commercial_dense; C-3-R 2020 commercial_dense; C-3-G 2020 commercial_dense; C-3-S 2020 commercial_dense; C-3 1960 commercial_dense; C-M 2020 commercial_dense; C-2 2020 commercial_light; SSO 2000 commercial_light; WMUO 2020 commercial_light; MB-O 2020 commercial_light; MUO 2020 commercial_light; CMUO 2020 commercial_light; C 1948 commercial_dense; M-1 2020 industrial; PDR-1-G 2020 industrial; LI 1948 industrial; HI 1948 industrial; P 2020 public; U 1948 unrestricted`.
  Ruling on 1921's C: dark blue, because it held downtown.
- [ ] **Step 2: Run it.** `cd pipeline && uv run pytest tests/test_districts_rules.py -k map_category -v`. Expected: FAIL.
- [ ] **Step 3: Implement.** Explicit dict per canonical code for blue zones and exceptions, then parse the rule's first `1 per N` for residential and mixed rows: N >= 500 is `homes_low`, 250-499 `homes_mid`, 150-249 `homes_high`, under 150 or form-based, "no limit" or "unlimited" `homes_top`. Per-lot caps of 1, 2, 3 stay as they are. RH-1's "1 per 3,000 with CU" stays `single_family`. Don't change `zoning-districts.csv`'s `category` column.
- [ ] **Step 4: Check unread limits.** For each of `CCB, CRNC, CVR, RED, RED-MX, RSD, SPD, WMUG, RCD, RTO-C, RED`, the per-agreement zones and named NCDs, consult the SF Planning Code at codelibrary.amlegal.com. Record the step and source in `docs/research/map-category-notes.md`. If it can't be confirmed, use the closest comparable zone and mark it "unverified" there. Named NCDs without a confirmed limit are `homes_low`, unverified.
- [ ] **Step 5: Update the other pipeline tests** that name old categories to test the same behaviour under the new names; update `COLORS` to the Task 2 hues approximately (it's the review page only).
- [ ] **Step 6: Run** `cd pipeline && uv run pytest`. Expected: all pass.
- [ ] **Step 7: Commit** `feat(pipeline): map categories for the red-to-green scale`.

### Task 2: Palette, legend, chart on the site

**Files:**
- Modify: `site/lib/palette.ts`, `site/lib/stats.ts`, `site/components/Legend.tsx`, `site/components/legend.module.css`
- Test: `site/tests/palette.test.ts`, `site/tests/districts.test.ts`

**Interfaces:**
- Consumes: Task 1's names.
- Produces: `HOMES_RAMP` (the seven steps), `OTHER_USES` (`commercial_light, commercial_dense, industrial, public, unrestricted, unknown`), `CATEGORY_COLORS`, `CATEGORY_LABELS`.

- [ ] **Step 1: Write failing tests.** In `palette.test.ts`, replace the "darker with every step" test with: `the scale runs from a red to a deep green` (first hex has red the largest channel; last has green the largest and OKLab L under 0.45); `steps are evenly spaced` (over normal vision on opaque colours, each adjacent ΔE within 25% of the mean, ignoring none). Keep the existing per-pair bars unchanged. Replace the commercial-over-ink test with `commercial_dense over black ink does not read as homes_top over paper`, at least 15. Add `threePlusShare` tests: the shares use `three_unit` and all `homes_*`; total also adds `single_family` and `two_unit`.
- [ ] **Step 2: Run.** `cd site && npx vitest run tests/palette.test.ts tests/districts.test.ts`. Expected: FAIL.
- [ ] **Step 3: Find colours.** Starting from `#DB2D27 #EB9632 #F1EC9D #9DC471 #3F9C48 #18682F #00381C`, light blue near `#9CC8EA`, dark blue near `#1F4E9C`, brown near `#7A5230`, grey-tan near `#CFCBBF`, and a purple for unrestricted, search for the smallest changes meeting the tests. Keep the search script out of the repo. If the existing bars can't all be met with 13 categories, return DONE_WITH_CONCERNS with the best numbers: adjacent steps and steps-vs-uses first. Do not lower a bar unless told to.
- [ ] **Step 4: Implement.** Labels: `One home, Two homes, Three homes, 3-4 homes, About 6 homes, About 12 homes, 20+ or no limit, Light commercial, Dense commercial, Industrial, Public, Unrestricted, Not read yet`. The legend head reads "Homes allowed on a typical lot"; ticks `1, 2, 3, 3-4, 6, 12, 20+`. Update the palette file's header comment with the real minimums.
- [ ] **Step 5: Run** `npm test`, `npx tsc --noEmit`, `npm run lint`. Expected: pass, with e2e left to Task 3.
- [ ] **Step 6: Commit** `feat(site): red-to-green palette and new legend`.

### Task 3: Rebuild, publish, wire up

**Files:**
- Modify: `pipeline/src/sfzmaps/districts/shared.py` (`ARCHIVE_FORMAT` or `MANIFEST_VERSION` if needed), `site/lib/published.json`, `site/lib/shared.ts` if the version changes
- Test: `site/tests/districts.test.ts`

- [ ] **Step 1:** Read `shared.py`, `site/lib/shared.ts` and `publish-tiles.mjs`. Confirm that the deployed site's pointer name is not overwritten: `/shared/districts.json` must keep its old content for the live build, or the live site would reject unknown categories and drop to scans. Add a test `old pointers are refused without breaking` only if behaviour changes. Raise `MANIFEST_VERSION` and `ARCHIVE_FORMAT` as their comments require.
- [ ] **Step 2:** Rebuild every series with `--republish`, then the shared files. Confirm from logs that no API calls were made. Compare each series' parcels count with the old `stats.json`: totals must match.
- [ ] **Step 3:** Publish under new names only. List the uploaded names; confirm that the old pointer, table and archives still return their old bytes.
- [ ] **Step 4:** Check numeric claims. For every number or comparison in `site/content/story.ts` and the chart (`threePlusShare`), compare before and after. Write results to the task report. Any claim that changed goes to Task 4.
- [ ] **Step 5:** Run e2e: `NEXT_PUBLIC_TILES_BASE_URL=https://zwgxwpoxqzfauqfg.public.blob.vercel-storage.com npm run test:e2e`. Fix tests that name old categories. Expected: no failures.
- [ ] **Step 6: Commit** `feat: publish the map under the new categories`.

### Task 4: Story text and a look at the map

**Files:**
- Modify: `site/content/story.ts`, `site/README.md`, `pipeline/README.md` where categories are named

- [ ] **Step 1:** Rewrite every step that names a colour or changed claim so it matches the screen. Only colours and claims change; no new advocacy. Steps known to name colours: lines 41, 59, 63, 78, 97, 120, 139, 188, 207, 286. Note in particular 1921's "pink streets" Commercial, now blue, and "violet where homes and shops mix".
- [ ] **Step 2:** Render each story step in its views on the preview or a local server with published tiles, and look at them, including "both" over the solid-black districts of 1921 and 1948. Save a contact sheet to the scratchpad and report what was seen.
- [ ] **Step 3:** Run all tests from `AGENTS.md`; commit `docs(story): describe the new colours`.
