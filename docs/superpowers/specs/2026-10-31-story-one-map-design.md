# One Map Approach For The Story And Explorer

## Why

The story shows loading gaps between steps. It still draws from per-year vector archives, while the explorer draws from the shared tiles and colour table (`site/lib/lookupLayer.ts`). The owner doesn't want two independent map approaches.

Measured on the story before this work, against published tiles, with a 1.4s camera move:

- Fast connection: picture finished at 1.40-1.47s, with gaps during the move.
- Simulated 4G: finished at 1.44-2.73s, with blank blocks visible mid-move. `1960-not-a-downzoning` is mostly blank at 0.67s.
- Scrolling back refetched tiles at most steps.

Causes: the story warms the next era at the current camera, not the next step's camera; colours and scans arrive separately; hiding a vector layer discards its tiles.

## What the reader sees (the owner's decisions)

- There is one map approach. The per-year map path is removed from the site, including the explorer's fallback.
- Moving between story steps is one motion: the map pans while the old year cross-fades into the new one.
- Where the new picture isn't ready, the old map stays on screen rather than a blank gap.
- In the explorer, a year change stays instant, with no fade (already shipped; the owner asked for this). A smaller lot wins where lots overlap (also shipped).

## Goals

1. `StoryMap` draws all districts through `addLookupLayer`, in the story and the explorer. No per-year vector source or layer remains in `site/`.
2. Story transitions: colours cross-fade over the same 1.4s as the camera move.
3. Scans, in the story only: the old scan stays underneath while the new one loads, and the new one fades in above it as its tiles are ready. The old scan goes once the new source has finished loading and the fade is done. If the new scan has really failed, an 8s limit ends the hold, so a slow scan never counts as failed at ordinary 4G speeds (up to 2.73s measured).
4. The next step's destination is loaded ahead of time: the shared tiles, the next era's scan and the current era's scan, so the old map exists where the pan arrives. This costs up to two scan viewports per era change. It replaces the near-zero-opacity warm-up layer, which warmed the current camera.
5. Story recordings improve against the numbers above on both connections, with no blank-block frames in the recorded transitions on the fast connection.

## Non-goals

- No change to story text, cameras or step order, unless a changed picture makes text wrong (AGENTS.md: check).
- No change to the pipeline's per-year outputs; review pages and stats use them. Nothing is deleted from Blob.
- No change to how scans are stored or drawn in their own right; the paper layer stays faithful (AGENTS.md).
- No WebGPU, no new map library.

## Anti-goals

- A blank block or a flash of bare basemap during a transition when an older picture was on screen.
- A second code path for "story" versus "explorer" drawing of districts. They differ only in settings such as fade length.
- Tests of MapLibre internals or trivial assertions. Test what we decide and what a person sees.
- Overwriting a published file. Any publish goes through `site/scripts/publish-tiles.mjs`.

## Behaviour decisions

Decided in conversation, as the builder's defaults the owner has seen:

- Missing or unreadable shared files: the map shows scans with the existing "colours are coming" note. There is no per-year fallback.
- `useDistrictAvailability` answers from the shared pointer, not by probing per-year archives. An era not in the pointer shows its scan and the note.
- The chart still reads each series' `stats.json` through `districtStatsUrl`.
- `publish-tiles.mjs` stops publishing per-year district archives, because the site no longer reads them; it still publishes each `stats.json` and the shared files.
- The explorer doesn't hold old scans: a year change there stays instant, because its purpose is comparing years. The hold is a story setting, not a separate drawing path. (Author's call.)
- A brief view of bare basemap is acceptable only where neither scan has loaded and the preload hasn't finished, such as after a very fast scroll. (Author's call.)
- A lost WebGL context no longer sets a "use per-year" flag. `lookupLayer` already rebuilds its program and table on a restored context.
- If a place has never loaded there's no sharp old map to show. MapLibre's coarser parent tile is acceptable, when loaded.

## Constraints

- `maplibre-gl` is pinned at 6.16.0. `lookupLayer.ts` is the only file that may read MapLibre internals.
- The story and explorer are separate map instances on one page. The table and pointer must be fetched once and shared (`loadSharedDistricts` memoized). Each map holds its own GPU copy; total map memory is expected to be up to about twice 49MB.
- Reduced-motion users get no fades or eased moves, as today.
- Published tiles are at `https://zwgxwpoxqzfauqfg.public.blob.vercel-storage.com`. The e2e run with that base is required before a PR (AGENTS.md).

## Left to the builder

- How to load the next step's destination. Fetching through the pmtiles client for the tiles covering the next camera is a sensible start; another map instance isn't, because of its memory and GPU cost.
- How to detect a scan is ready (`map.isSourceLoaded` is public) and how to order layers (`map.moveLayer` is public).
- Code structure inside `StoryMap.tsx`.

## How we'll know

- `npm run lint`, `npx tsc --noEmit`, `npm test`, pipeline tests, and both e2e modes pass.
- An e2e test checks that a story transition has no frame where a colour area goes bare once the old colours were drawn, and that a step whose scan never arrives still ends with its colours and no stuck state.
- Re-run the filmstrip comparison on fast and 4G-like connections, and look at the contact sheets.
- The owner checks on a phone via the PR preview.
