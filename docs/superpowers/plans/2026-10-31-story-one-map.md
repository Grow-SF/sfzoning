# One Map Approach Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The story and explorer draw districts one way, and a story step pans while cross-fading with the old map holding wherever the new one isn't ready.

**Architecture:** `StoryMap` always draws through `addLookupLayer`; per-year vector code leaves the site. Story scans gain a hold (old stays under new until new has loaded), and the next step's destination is fetched ahead through the same pmtiles client MapLibre uses.

**Tech Stack:** Next.js, TypeScript, MapLibre GL JS 6.16.0, pmtiles 4.5.0, Vitest, Playwright.

**Spec:** `docs/superpowers/specs/2026-10-31-story-one-map-design.md`

## Global Constraints

- `maplibre-gl` stays pinned at 6.16.0. Only `site/lib/lookupLayer.ts` may read MapLibre internals.
- No tests of MapLibre internals and no trivial assertions. Test what we decide and what a person sees.
- Never overwrite a published file. Nothing is deleted from Blob. No pipeline changes.
- Reduced-motion readers get no fades or eased moves, as today.
- Explorer year changes stay instant with no scan hold.
- Don't print or commit secrets. Don't commit `pipeline/data/`. The worktree has no data folder: local e2e needs the tile server, so use published tiles (below).
- Commits end with: `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` and `Claude-Session: https://claude.ai/code/session_01UjTgrnx2j4PhzZErrad25Q`.

E2E command, from `site/`: `NEXT_PUBLIC_TILES_BASE_URL=https://zwgxwpoxqzfauqfg.public.blob.vercel-storage.com npm run test:e2e`. Stop any dev server on port 3100 first.

## Review Focus

- Very fast scrolling past several steps: the map ends on the last step, with exactly one scan left visible (Task 2 test).
- A hold interrupted by another step: no scan stays stuck visible (Task 2 unit test).
- A scan that never arrives: colours still show, the hold ends at the limit, no stuck state (Task 2 test).
- Scrolling back up: steps work in reverse, with the preload targeting the step ahead in scroll direction's neighbour on both sides (Task 3 unit test).
- Preload failure or a missing archive: no notice and no thrown error (Task 3 unit test).

---

### Task 1: One district path

**Files:**
- Modify: `site/components/StoryMap.tsx`, `site/components/Story.tsx`, `site/lib/districts.ts`, `site/lib/shared.ts`, `site/lib/tiles.ts`, `site/lib/eras.ts`, `site/lib/palette.ts`, `site/scripts/publish-tiles.mjs`, `site/README.md`
- Test: `site/tests/explorer.spec.ts`, `site/tests/districts.test.ts`, `site/tests/lib.test.ts`

**Interfaces:**
- Produces: `loadSharedDistricts(): Promise<SharedDistricts | null>`, memoized. `useDistrictAvailability()` answers from it.

An untested draft of this task is at `/private/tmp/claude-501/-Users-sbuss-workspace-sfzoning/eb18d2fc-e22a-48a6-afbc-57c31cd3e134/scratchpad/one-map-draft.patch`. Read it for guidance; verify everything it does. Apply nothing blindly.

- [ ] **Step 1: Change the e2e tests first**

Replace "an era without published districts falls back..." with `"without the districts, the map shows the scan with a note instead of failing"`: route `**/shared/districts*.json` to 404, go to step `1921-sunset`, expect `classification-coming` to contain "coming", no `legend`, no `.map-notice`. Delete "going back to a year just viewed reuses its tiles" (the sweep test covers it). Point `tilesExist` skips at `shared/districts.json`. Drop `openLot`'s per-year branch. Add a test to `tests/story.spec.ts`, `"the story draws its colours from the shared tiles"`: at `1921-sunset`, once idle, requests matching `/\/districts\/[^/]*-districts\.pmtiles/` total zero, and requests matching `/shared\/districts-[^/]+\.pmtiles/` are above zero.

- [ ] **Step 2: Run them and see the new story test fail**

Run: `npx playwright test tests/story.spec.ts -g "shared tiles" --project=desktop` with the published base. Expected: FAIL, per-year requests found.

- [ ] **Step 3: Remove the per-year path**

From `StoryMap.tsx`: `distSrc`, `distFill`, `distLine`, `addDistricts`, `keptEras`, `lineWidthExpression`, `fillOpacity`, the `webglcontextlost` listener, the per-year click branch, and the district warm logic. Load shared districts in both maps. Elsewhere remove `probeDistricts`, `districtPath`, `districtSource`, `rememberEra`, `KEPT_ERAS`, `fillColorExpression`, `noteLostContext` and the storage flag, with their tests. `loadStats` fetches `districtStatsUrl` directly. `publish-tiles.mjs` publishes `stats.json` per series plus the shared files. Update the README's shared section and any stale "probes at runtime" or per-era text.

- [ ] **Step 4: Run all checks**

Run: `npx tsc --noEmit && npm run lint && npm test`, then the e2e command. Expected: no failures. `grep -rE "dist-\$\{|-districts\.pmtiles" components lib` finds nothing.

- [ ] **Step 5: Commit**

### Task 2: Story motion and the scan hold

**Files:**
- Create: `site/lib/scanHold.ts`
- Modify: `site/components/StoryMap.tsx`
- Test: `site/tests/lib.test.ts`, `site/tests/story.spec.ts`

**Interfaces:**
- Consumes: `lookupRef.current.show(look, ms)` from `lib/lookupLayer.ts`.
- Produces: `HOLD_LIMIT_MS = 8000`; `type Hold = { held: string[]; since: number }`; `heldScans(hold: Hold | null, wanted: string[], loaded: (series: string) => boolean, faded: boolean, now: number): Hold | null`, which returns the scans to keep under the new one, or null when none are left.

- [ ] **Step 1: Write the failing unit tests**

`heldScans`: keeps held scans while `wanted` isn't loaded; keeps them while loaded but `faded` is false; returns null when loaded and faded; returns null at `since + HOLD_LIMIT_MS` even if never loaded; drops a held scan that's now in `wanted`.

- [ ] **Step 2: Run them to see them fail**

Run: `npm test -- lib.test`. Expected: FAIL, no `scanHold` module.

- [ ] **Step 3: Implement `heldScans`, and use it in `paint`**

In the story only (`!interactive`), when the shown scans change and motion isn't reduced: previously visible scans join the hold and keep their opacity, the new scan moves above them with `map.moveLayer` and fades in over `PAN_MS`, and a timer rechecks `heldScans` every 200ms with `map.isSourceLoaded`, clearing when it returns null. A new step merges into the existing hold and restarts its limit. Colours fade over `PAN_MS = 1400`, which also replaces the literal in `easeTo`. The explorer's view changes keep `FADE_MS`, and its year changes stay instant. Hide a released scan as `paint` does today.

- [ ] **Step 4: Write e2e tests**

In `story.spec.ts`: `"scrolling fast through the story leaves only the last step's scan showing"` (jump across five steps with no wait; once idle plus 2s, query `getLayoutProperty` of `scan-*` layers on `window.__map` and expect visible ones to be only the last step's scan). `"a scan that never arrives ends with colours and no stuck hold"` (route `**/1948/1948-paper.pmtiles` to 404; go to `1948-sunset`; after 9s the `legend` is visible, and the previous scan's layer is hidden). In `explorer.spec.ts`, the existing instant-year test must still pass.

- [ ] **Step 5: Run checks, commit**

Run: `npx tsc --noEmit && npm run lint && npm test`, then the e2e command. Expected: no failures.

### Task 3: Load the next destination ahead

**Files:**
- Create: `site/lib/preload.ts`
- Modify: `site/components/StoryMap.tsx`, `site/components/Story.tsx`
- Test: `site/tests/lib.test.ts`, `site/tests/network.spec.ts`

**Interfaces:**
- Consumes: the `Protocol` instance in `loadMaplibre`, which this task keeps in a module variable.
- Produces: `tilesCovering(camera: {center: [number, number]; zoom: number}, width: number, height: number, tileSize: 256 | 512, maxZoom: number): {z: number; x: number; y: number}[]`; `preload(protocol, urls: string[], tiles): Promise<void>`, which never rejects; `MapState.preload?: { camera; scanSeries: string[] }`, replacing `MapState.prefetch`.

- [ ] **Step 1: Write the failing unit tests**

`tilesCovering`: at z14.2, 512px tiles, 1440×900, it returns z14 tiles that include the camera's centre tile and cover all four corners; 256px tiles use z one higher; zoom above `maxZoom` clamps to `maxZoom` and covers the same ground. `preload` resolves when a fetch fails. For scrolling back, `Story` is tested through e2e below.

- [ ] **Step 2: Run them to see them fail**

Run: `npm test -- lib.test`. Expected: FAIL, no `preload` module.

- [ ] **Step 3: Implement**

`preload` gets each archive's `PMTiles` by adding it to the protocol once (`protocol.add`, so MapLibre reuses it and its header cache), then calls `getZxy` for each tile. `maxZoom` comes from the archive header. `Story` sends the camera of the next step and the previous step (both directions), with the current era's scans plus that step's scans. `StoryMap` skips the "allowed" view's scan as `scanOpacity` does, and fetches shared tiles once per destination. Remove the near-zero-opacity warm-up: `WARM`, and the prefetch branch of `scanOpacity`. Add 1 viewport of margin only if the filmstrip in Task 4 shows it's needed.

- [ ] **Step 4: Update `network.spec.ts`**

Its allowed-scan sets follow the new rule: current era's scans, plus the neighbouring steps' scans. Add `"the next step's shared tiles are already fetched before the reader scrolls there"`: at a step, wait for idle, record shared-tile requests during the transition to the next step, and expect fewer than half of an unwarmed first visit to the same camera in a fresh page.

- [ ] **Step 5: Run checks, commit**

Run: `npx tsc --noEmit && npm run lint && npm test`, then the e2e command. Expected: no failures.

### Task 4: Measure, look, and open the PR (controller)

- [ ] **Step 1: Re-run the filmstrip** with `scratchpad/film.cjs` on fast and 4G, and compare with the spec's numbers: finished at 1.40-1.47s and 1.44-2.73s. Look at the contact sheets for blank blocks.
- [ ] **Step 2: Measure memory** with both maps loaded, and record it against the expected "up to about twice 49MB".
- [ ] **Step 3: Check story text** against what's on screen at each step (AGENTS.md).
- [ ] **Step 4: Run pipeline tests, full site checks and e2e, then a whole-branch review.**
- [ ] **Step 5: Push and open a PR** with the e2e output, for a phone check on the preview. Merging waits for the owner.
