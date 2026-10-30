# Scrollytelling Site (Step Three) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A mobile-first Next.js scrollytelling page where scrolling advances through SF's zoning eras on a sticky map, showing the original scans and "what it allowed", ending in an explorer.

**Architecture:** A Next.js App Router app in `site/`. One client component owns a MapLibre map with all era layers pre-declared (scan rasters plus district vectors via the `pmtiles` protocol); per-step state only changes opacity, camera and filters. A small IntersectionObserver hook drives the steps. The content is typed TS. Tiles come from `TILES_BASE_URL`; in dev that's a local Range-capable server over `pipeline/data/out`.

**Tech Stack:** Next.js (latest stable, App Router, TypeScript), maplibre-gl, pmtiles, Vitest + Testing Library, Playwright, plain CSS modules with design tokens.

**Spec:** `docs/superpowers/specs/2026-10-30-scrollytelling-site-design.md`

## Global Constraints

- Mobile-first. Test at 390×844 and 1440×900. Native scroll only, no scroll-jacking. Map gestures are disabled during the story and enabled in the explorer.
- No runtime third-party basemap, fonts CDN or tracking. Fonts are self-hosted via `next/font`; the basemap is our own `out/ref/blocks.pmtiles`.
- `prefers-reduced-motion`: instant transitions, no flyTo animation.
- Credit "© Cartography Associates, David Rumsey Map Collection" is visible whenever 1948 imagery is visible.
- Story copy is a DRAFT, marked as such in a visible dev-only banner and in `content/story.ts` comments. Every era step cites a source from `docs/research/sources.md`.
- Tiles: `NEXT_PUBLIC_TILES_BASE_URL` (default `http://127.0.0.1:8787`). `scripts/serve-tiles.mjs` serves `../pipeline/data/out` with HTTP Range and CORS on port 8787. No tiles in git.
- The district layers degrade gracefully: if an era's districts PMTiles is missing, the "allowed" view shows the scan with a "classification coming" note. No crash.
- Use the `frontend-design` skill guidance for visual design: distinctive, editorial, legible; an accessible palette that's colour-blind-safe for the 10 story categories (single_family, two_unit, three_unit, multi_family, mixed, commercial, industrial, public, unrestricted, unknown).

## Review Focus

1. A small phone (360 px wide) with a long step body: text must stay readable over the map (card max-height plus internal scroll, never clipped). Test in S2 with Playwright.
2. A fast scroll that skips several steps: the map must land on the final step's state, without stale intermediate animations. Test in S2 (hook unit test).
3. An unreachable tile server: the page still renders the text and shows a friendly map-error notice. Test in S1.
4. Era layers where both views are visible at once ("both"): the district fill is semi-transparent over the scan, the legend shows both, and the credit appears if the era is 1948. Test in S4.
5. Keyboard-only navigation through the story and the explorer controls (slider, toggle): visible focus, operable. Test in S4/S5.

---

### Task S1: App scaffold, tiles server, map shell

**Files:**
- Remove the root Vite starter: `index.html`, `src/`, `public/vite.svg`, root `vite.config.js`, `eslint.config.js`, and root `package.json`/`package-lock.json`. Keep the README, with its content replaced by a pointer to `site/` and `pipeline/`.
- Create `site/` (`create-next-app` with TypeScript, App Router, ESLint, no Tailwind), `site/scripts/serve-tiles.mjs`, `site/lib/tiles.ts`, `site/lib/eras.ts`, `site/components/StoryMap.tsx`, `site/app/page.tsx`, `site/vitest.config.ts`, `site/playwright.config.ts`, `site/tests/`.

**Interfaces:**
- `lib/eras.ts`: `type Era = { id: string; label: string; year: number; scanSeries: string[]; districtSeries?: string; sources: string[] }`, plus `ERAS: Era[]`.
  - Order: 1921, 1929, 1935, 1948, 1960, 1964, 1968, 1972, 1975, 1980, 1980–90 (zoning series), 1990–96, 1996–2000, 2000s, today.
  - Height and special-use series aren't shown as scans in the story [call].
- `lib/tiles.ts`:
  - `tileUrl(path: string): string` → `pmtiles://${BASE}/${path}`;
  - `scanSource(series)`;
  - `districtSource(series)`;
  - `BASEMAP = "ref/blocks.pmtiles"`.
- `components/StoryMap.tsx`: a client component exposing `apply(state: MapState)`, where `type MapState = { era: string; view: "scan"|"allowed"|"both"; camera: {center:[number,number]; zoom:number; bearing?:number}; highlight?: GeoJSON }`.
  - Adds the basemap and every era's scan raster source/layer once, at opacity 0.
  - `apply` cross-fades opacities (duration 0 under reduced motion) and calls `easeTo`/`jumpTo`.
  - On load error, renders a notice (Review Focus 3).

- [ ] Tests:
  - Vitest:
    - `tileUrl` builds pmtiles URLs from env;
    - `ERAS` ids are unique and sorted by year.
  - Playwright `smoke.spec.ts`: the page loads at both viewports and a `<canvas>` exists.
  - With the tile server down, the notice appears and the text still renders.
- [ ] Run `node site/scripts/serve-tiles.mjs` plus `npm run dev`. Screenshot 1948 at zoom 13 into the scratch dir and view it. Commit `feat(site): Next.js scaffold, tile server, map shell`.

### Task S2: Scroll engine and layout

**Files:** `site/lib/useScrollSteps.ts`, `site/components/Story.tsx`, `site/components/StepCard.tsx`, `site/app/page.module.css`, `site/content/story.ts` (with 3 placeholder steps), `site/content/schema.ts`; tests.

**Interfaces:**
- `useScrollSteps(refs: RefObject<HTMLElement>[], opts?) -> { active: number; progress: number }`. Uses IntersectionObserver with a threshold at the viewport middle. When several steps change in one frame, only the last active state is reported (Review Focus 2).
- `content/schema.ts`: `type Step = { id: string; era: string; view: View; camera: Camera; title: string; body: Block[]; sources: string[]; highlight?: GeoJSON }`, where `Block = { kind: "p" | "quote" | "stat"; text: string; href?: string }`. Also `validateStory(steps, eras): string[]` (errors).
- `Story.tsx` renders a sticky map plus a scrolling column of `StepCard`s, and calls `map.apply(...)` on active change.
  - Desktop (≥ 900 px): text column on the left, 38% width.
  - Mobile: cards overlay the bottom of the map, max-height 55vh, with internal scroll (Review Focus 1).

- [ ] Tests:
  - Vitest:
    - the hook reports the last of rapid changes (mock IntersectionObserver);
    - `validateStory` catches an unknown era.
  - Playwright:
    - scrolling through all steps sets `data-active-step` on each in turn;
    - at 360×740 the longest step's card is fully readable via its own scroll.
- [ ] Commit `feat(site): scroll engine and responsive layout`.

### Task S3: Story content (draft) and sourcing

**Files:** `site/content/story.ts` (full draft), `site/components/Sources.tsx`, `site/components/Credits.tsx`, `site/app/about/page.tsx` (methods and sources).

- [ ] Write ~10–14 steps following the spec's arc. Copy comes from `docs/research/zoning-history.md`, with specific numbers only where the research cites them:
  - 1921 First vs Second Residential;
  - 1960 first density caps;
  - 1970–75 downzonings;
  - 1972 height limits;
  - 1978 RH/RM with "~180,000 fewer units" (Final EIR p.122);
  - 1985 Downtown Plan;
  - today.
  Every step cites its source. Choose cameras by looking at the scans: tile screenshots of candidate areas, e.g. the Sunset, Richmond, Mission and downtown. Credit for 1948 per the global constraints.
- [ ] The about page covers how the maps were aligned (from the step-one spec and README, in plain language), the accuracy bar, the review status, and sources.
- [ ] Tests: `validateStory(STORY, ERAS)` returns []; every source id resolves; the 1948 steps show the credit (Playwright).
- [ ] Commit `feat(site): draft story and sources`.

### Task S4: "What it allowed" layers, legend, explorer, stats

**Files:** `site/lib/palette.ts`, `site/components/Legend.tsx`, `site/components/Explorer.tsx`, `site/components/ParcelPopup.tsx`, `site/components/StatsChart.tsx`, `site/lib/rules.ts` (rules JSON loader), and `site/scripts/build-rules.mjs` (converts `docs/research/zoning-districts.csv` → `site/public/data/rules.json`); modify StoryMap.

**Interfaces:**
- `palette.ts`: `CATEGORY_COLORS: Record<Category,string>` and `CATEGORY_LABELS`.
- StoryMap adds per-era district vector sources (`districtSource`) with a fill layer coloured via a `match` expression on `category` and a thin outline on `districts`. A missing source is caught → "classification coming" (global constraints).
- `Explorer`: an era slider (keyboard-operable range input with year labels), a view toggle, and map gestures enabled. Clicking a parcel opens `ParcelPopup` with the era, code, plain-language rules (from rules.json: allowed_uses, max_units_per_lot, height_limit_ft) and confidence.
- `StatsChart`: an SVG chart of the share of residential parcels allowing 3+ units, by era, from each `out/<series>/districts/stats.json`. If absent, the chart hides itself.

- [ ] Tests:
  - Vitest: palette covers every category; `build-rules` output schema; the rules lookup by (era year, code).
  - Playwright:
    - the explorer slider changes the era via the keyboard (Review Focus 5);
    - "both" view on 1948 shows the legend and credit (Review Focus 4);
    - clicking a parcel opens the popup (skip if the districts tiles are absent).
- [ ] Commit `feat(site): allowed-use layers, legend, explorer, stats`.

### Task S5: Polish, performance, accessibility

**Files:** as needed; `site/README.md`.

- [ ] Prefetch the next step's era tiles (warm sources by setting opacity 0.001 one step ahead).
- [ ] Lazy-init the map below the hero.
- [ ] Check focus styles, contrast, and the `<html lang>`.
- [ ] Add OG/meta tags.
- [ ] Lighthouse mobile run on the production build (`next build && next start`), recording the scores.
- [ ] Playwright screenshots of every step at both viewports, saved under the scratch dir and spot-checked by eye.
- [ ] `site/README.md`: dev setup (tile server, env var) and the later Vercel deployment notes (tiles to object storage, set `NEXT_PUBLIC_TILES_BASE_URL`).
- [ ] Commit `feat(site): polish, performance, accessibility`.
