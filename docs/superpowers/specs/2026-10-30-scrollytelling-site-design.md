# sfzoning.com scrollytelling site: design (step three)

Date: 2026-10-30
Status: written autonomously overnight at the owner's request. The owner
asked for:
- a NYT-style feature where scrolling advances through the years and shows
  the changes;
- "figure out how to best show the maps";
- mobile friendly;
- Vercel deployment eventually, not now;
- a preference for Next.js.
Everything else is marked **[call]**.

## Why

San Francisco's housing shortage was zoned in. The archive of zoning maps
from 1921 to today, now aligned to real streets (step one) and turned into
"what was allowed where" (step two), lets a reader *see* it happen: in 1921,
apartments were legal on most residential land. Over the following decades,
especially 1960–1978, most of the city was rezoned to allow only one or two
homes per lot. The site tells that story by scrolling, with the real historic
maps as evidence.

## Goals

1. **A single long-form page.** A sticky full-screen map, with narrative steps
   that scroll over it. Each step sets:
   - the **year**;
   - the **layer**: the original scan, the "what was allowed" district
     colours, or both;
   - the **camera**: center, zoom, bearing;
   - optional **highlight areas** with callouts.
2. **Two ways to read every era [call]:**
   - "**The map as drawn**": the georeferenced scan over a quiet modern
     basemap;
   - "**What it allowed**": parcels coloured by story category (single-family,
     2-unit, apartments, mixed, commercial, industrial, public). The same
     palette is used for every era so changes pop.
   Steps cross-fade between eras and between the two views. A small, always-
   visible legend explains the colours.
3. **A finale explorer:**
   - a year slider/scrubber across all eras;
   - the view toggle;
   - tap or click a parcel to see that era's code and plain-language rules
     (from the research table);
   - each era's statistic, e.g. "share of residential land where apartments
     were legal", shown as a small chart driven by `stats.json`.
4. **Mobile-first.**
   - On phones the map fills the screen and step text arrives in cards that
     slide over its lower part.
   - On desktop the text column sits to the left of a large map.
   - Touch gestures on the map are disabled during the story, so the page
     scrolls, and enabled in the explorer.
5. **Honest sourcing.** Each era step links its primary source, from
   `docs/research/sources.md`. Credits include "© Cartography Associates, David
   Rumsey Map Collection" wherever 1948 imagery shows (a requirement).
6. **Fast.**
   - Vector and raster PMTiles via the `pmtiles` protocol.
   - Only the current and next steps' layers are loaded, and the next step's
     tiles are prefetched.
   - The scan layers cap at the source resolution (z18).
   - Good Lighthouse scores on mobile.

## Non-goals (for now)

- Deployment and hosting of the ~1 GB of tiles. Locally, tiles are served from
  `pipeline/data/out` via a symlink. A `TILES_BASE_URL` environment variable
  points elsewhere later (Vercel Blob / R2) **[call]**.
- Final copy. Draft copy is written from `docs/research/zoning-history.md`
  with citations, clearly marked as a draft for the owner to edit.
- A CMS. Content lives in a typed file, `site/content/story.ts`.

## Anti-goals

- No third-party basemap or tracking dependency at runtime **[call]**. The
  basemap is our own quiet vector layer (DataSF blocks, streets and
  shoreline as PMTiles), so the site is self-contained and the old maps stay
  the visual focus.
- No scroll-jacking. Native scrolling drives the steps; the map reacts.
- No janky mobile map. Avoid heavy WebGL work during scroll, and fall back to
  instant transitions under `prefers-reduced-motion`.

## How (decided) [calls]

- **App:**
  - Next.js (App Router, TypeScript) in `site/`;
  - the unused Vite starter at the repo root is removed;
  - plain CSS modules with design tokens;
  - built with the `frontend-design` guidance for typography and colour.
- **Map:** MapLibre GL JS with the `pmtiles` protocol. All layers are
  declared once and opacity is toggled per step; nothing is re-added.
- **Scrolling:** a small `useScrollSteps` hook on IntersectionObserver
  (scrollama-style) reports the active step and progress within it. Tested
  with Vitest.
- **Data contract:** the site reads, per series:
  - `out/<series>/<series>.pmtiles` (scan);
  - `out/<series>/manifest.json` (credit, bounds);
  - `out/<series>/districts/<series>-districts.pmtiles`;
  - `stats.json`;
  - plus `out/ref/*`.

  A small manifest of eras (`site/content/eras.ts`) lists which series form
  each era, their display years and their sources.
- **Content:**
  - `site/content/story.ts`: an array of steps, each with `id`, `era`,
    `view: 'scan' | 'allowed' | 'both'`, `camera`, `highlight?`, a `title`, a
    `body` (MDX-free rich text: paragraphs and links) and `sources`.
  - Draft story arc:
    1. today;
    2. 1921, the first ordinance, when apartments were broadly allowed;
    3. the 1920s–40s amendments;
    4. 1960, the first density caps;
    5. the early-1970s downzonings and the 1972 height limits;
    6. 1978, the RH/RM rezoning (~180,000 fewer legal units, per the Final
       EIR);
    7. 1985, the Downtown Plan;
    8. the 2000s;
    9. today and recent reform;
    10. the explorer.
- **Testing:**
  - Vitest for hooks and content validation, e.g. every step references an
    existing era and layer;
  - Playwright for screenshots of key steps at 390×844 (iPhone) and
    1440×900, plus a scroll-through smoke test that every step activates.
- **Dev tiles:** `site/public/tiles` → a symlink to `pipeline/data/out`,
  git-ignored. Next's dev server must serve Range requests for PMTiles. If it
  doesn't, `scripts/serve-tiles.mjs` serves `pipeline/data/out` with Range
  support on a local port, and `TILES_BASE_URL` points to it.

## Left to the builder

Visual design details within the brief, the palette (accessible,
colour-blind-safe, and consistent across eras), animation timing, exact
camera positions, and chart form.
