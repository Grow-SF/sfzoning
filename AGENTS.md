# Notes for AI coding agents

Read `CONTRIBUTING.md` first; it covers setup, tests and the pull request
flow. `pipeline/README.md` and `site/README.md` cover each part in depth. This
file lists what an agent is most likely to get wrong here.

## Running things

Whenever work could run in parallel, run it in parallel by default: for example
one `sfzmaps districts --series S --republish` per series, and separate test
suites side by side. Keep steps that depend on each other in order (`--shared`
needs every series first). Playwright stays at one worker; see `site/README.md`.

## Spending money

The pipeline's label and district readers call the OpenAI API, at real cost
to whoever owns the key. Every answer is cached in `pipeline/data/cache/llm/`.

- Don't start a run that makes live calls unless the person asked for it.
  Check first with `uv run sfzmaps estimate --year S` (no API calls).
- Prefer `sfzmaps districts --cache-only` or `--republish`, which never call
  the API.
- Set `SFZMAPS_MAX_USD` before any live run; the run stops cleanly at that
  spend.
- Changing a prompt, image size or model invalidates cached answers and means
  paying again.

## Secrets

The OpenAI key lives in `pipeline/.env` (git-ignored). Never print, log,
cache, copy or commit it. The same goes for `site/.env.local` and any Vercel
or Blob token.

## Data and tiles

- Never commit `pipeline/data/`. It's gigabytes, and it's shared through the
  processed bundle instead.
- Published tiles are served from Vercel Blob. Don't overwrite a published
  `.pmtiles` in place: readers with the page open break mid-session. Upload
  a changed archive under a new path, or accept that open tabs need a reload.
- A production build of the site must never point at local tiles
  (`scripts/check-tiles-env.mjs` enforces this; don't bypass it).

## Faithfulness to the historic maps

- The paper layer shows the scans as scanned. Don't upscale them, vectorize
  them, or draw anything that isn't in the scan. The only edits are the cream
  tint on the 1-bit scans, erasing printed border rules and binding shadows,
  and plain paper filling gaps between sheets.
- Zoning boundaries come from modern lot lines (DataSF parcels), never from
  tracing the scan.
- Where a street really moved, keep the historic position; don't warp the
  scan to fit today's streets.

## Credits and licenses

Keep the attribution "Based on sfzoning.com by Steven Bacio for GrowSF
(https://growsf.org)", the David Rumsey credit wherever 1948 imagery shows,
and the license notices. See `NOTICE`. Don't add dependencies under licenses
incompatible with AGPL-3.0.

## Checking your work

- Tests: `cd pipeline && uv run pytest`; in `site/`, `npm run lint`,
  `npx tsc --noEmit` and `npm test`. GitHub runs these on every pull request.
- If you changed the site, also run the end-to-end tests, which GitHub doesn't
  run: in `site/`,
  `NEXT_PUBLIC_TILES_BASE_URL=https://zwgxwpoxqzfauqfg.public.blob.vercel-storage.com npm run test:e2e`
  (stop any running dev server first). Paste the end of the output, with the
  pass and fail counts and any failures, into the pull request description.
  Don't claim they passed without running them.
- Passing tests aren't enough for map rendering. Seams, stray lines and
  missing ink have all slipped past the tests here. After changing warping,
  stitching or tiling, render the affected area and look at it; check the
  sheet seams and the solid-black districts.
- Story text describes what's on screen. If you change a step's view or
  colors, check the text still matches.
