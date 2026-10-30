# sfzoning.com: the scrollytelling site

A Next.js (App Router) site that tells the story of San Francisco's zoning on the city's own
historic zoning maps, then lets readers explore every map. The story copy is a **draft**
(`content/story.ts`); a dev-only banner says so.

Maps are MapLibre GL with [PMTiles](https://github.com/protomaps/PMTiles) archives produced by
the pipeline (`../pipeline/data/out`). There is no third-party basemap, font CDN or tracking:
the basemap is our own `ref/blocks.pmtiles` and fonts are self-hosted by `next/font`.

## Development

```sh
cd site
npm install            # also copies the MapLibre worker into public/maplibre/
npm run tiles          # tile server on http://127.0.0.1:8787 (serves ../pipeline/data/out)
npm run dev            # http://localhost:3000 (builds public/data/rules.json first)
```

- `scripts/serve-tiles.mjs` is a static server with HTTP Range and CORS, the two things
  PMTiles needs. `TILES_DIR` and `PORT` override its directory and port.
- `NEXT_PUBLIC_TILES_BASE_URL` points the site at the tiles (default `http://127.0.0.1:8787`).
  It is inlined at build time, so set it before `next build`. A production build
  (`VERCEL` set or `NODE_ENV=production`) stops in `prebuild` if it is unset or not https;
  `ALLOW_LOCAL_TILES=1` lets a local production build use the local tile server.
- `NEXT_PUBLIC_DRAFT` keeps the draft banner up and adds `noindex, nofollow` unless it is
  exactly `false` (`lib/draft.ts`).
- The site picks up whatever the pipeline has published. Each era's district tiles are probed
  (`lib/districts.ts`); an era without them shows its scan with a "district colours are
  coming" note instead of failing.

### Tests

```sh
npm test               # vitest: content, palette colour-blind checks, lib helpers
npm run test:e2e       # playwright, mobile (390×844) and desktop (1440×900)
SITE_PORT=3217 npm run test:e2e   # use your own dev server port if 3100 is another checkout's
```

Playwright runs with **one worker** (`playwright.config.ts`): every spec drives a WebGL map in
software rendering, and two workers starved each other of CPU, which made the story-scroll spec
flaky. Map specs can wait on the map's idle hook: `.map-wrap[data-idle="true"]` is set on
MapLibre's `idle` event and cleared on `movestart`/`dataloading`.

`node scripts/shot.mjs <outDir> [baseUrl]` screenshots the hero, every story step, the
explorer (all three views) and the about page at both viewports.

## Production build

```sh
NEXT_PUBLIC_TILES_BASE_URL=https://tiles.example.org npm run build
npm start
# or, against the local tile server:
ALLOW_LOCAL_TILES=1 npm run build
```

Optional: `NEXT_PUBLIC_SITE_URL` (default `https://sfzoning.com`) makes Open Graph URLs absolute.
The share image is `app/opengraph-image.jpg` (today's zoning, screenshotted from the site).

## Deploying (later; nothing is deployed yet)

The site deploys to Vercel as a normal Next.js app. The tiles do **not** go in git or in the
Vercel build (about 1.2 GB); they go to object storage that supports HTTP Range requests, and
the site reads them from there.

### 1. Upload the tiles

Upload, preserving paths relative to `pipeline/data/out/`:

- `**/*.pmtiles` (scans, `*/districts/*-districts.pmtiles`, `ref/blocks.pmtiles`)
- `**/manifest.json`
- `**/stats.json` (the explorer's chart reads `<series>/districts/stats.json`)

Skip the pipeline's scratch directories (`tuning/`, `par/`, `final/`) and everything else
(`*.fgb`, `*.vrt`, `cut/`, `review.html`, logs). For example, with Cloudflare R2 or S3 via
`rclone` (a remote named `tiles` already configured):

```sh
cd pipeline/data/out
rclone copy . tiles:sfzoning-tiles \
  --include "*.pmtiles" --include "manifest.json" --include "stats.json" \
  --exclude "tuning/**" --exclude "par/**" --exclude "final/**" \
  --header-upload "Cache-Control: public, max-age=86400" --progress
```

or with the AWS CLI:

```sh
aws s3 sync pipeline/data/out s3://sfzoning-tiles \
  --exclude "*" --include "*.pmtiles" --include "*manifest.json" --include "*stats.json" \
  --exclude "tuning/*" --exclude "par/*" --exclude "final/*" \
  --cache-control "public, max-age=86400"
```

Treat uploaded archives as immutable: a `.pmtiles` file overwritten while a reader has the
page open breaks that reader's Range reads (locally, district colours vanished for a session
while the pipeline rewrote them). Upload a changed archive under a new path, or accept
that open tabs need a reload. Re-run the same command whenever the pipeline publishes more eras; the site needs no rebuild
to show a new era's districts (it probes for them at runtime).

Serve `.pmtiles` as `application/octet-stream` (or `application/vnd.pmtiles`) **without**
`Content-Encoding: gzip`: a CDN that compresses the archive breaks Range reads.

### 2. CORS on the bucket

The browser fetches tiles cross-origin with a `Range` header, so the bucket needs:

```json
[
  {
    "AllowedOrigins": ["https://sfzoning.com", "https://*.vercel.app"],
    "AllowedMethods": ["GET", "HEAD"],
    "AllowedHeaders": ["Range", "If-Match", "If-None-Match"],
    "ExposeHeaders": ["Content-Range", "Content-Length", "ETag", "Accept-Ranges"],
    "MaxAgeSeconds": 86400
  }
]
```

Check it from a shell:

```sh
curl -sI -H "Origin: https://sfzoning.com" -H "Range: bytes=0-16383" \
  https://tiles.example.org/ref/blocks.pmtiles   # expect 206 and Access-Control-Allow-Origin
```

### 3. Point the site at it

In the Vercel project settings set `NEXT_PUBLIC_TILES_BASE_URL` to the bucket's public URL
(a custom domain such as `https://tiles.sfzoning.com`, no trailing slash needed) for
Production and Preview, then redeploy (the variable is inlined at build time). Before
publishing, remove the draft status: the story copy needs the owner's review.

## Lighthouse (mobile)

Lighthouse 12.8.2, mobile emulation (Moto G Power, slow 4G, 4× CPU throttle), headless
Chrome, against `next build && next start` with tiles from the local tile server, 2026-10-30:

| Page     | Performance | Accessibility | Best practices | SEO | FCP   | LCP   | TBT    | CLS |
| -------- | ----------- | ------------- | -------------- | --- | ----- | ----- | ------ | --- |
| `/`      | 88 (87–88)  | 100           | 96             | 100 | 1.2 s | 3.5 s | 190 ms | 0   |
| `/about` | 91          | 100           | 100            | 100 | 0.9 s | 3.5 s | 30 ms  | 0   |

```sh
npx lighthouse@12.8.2 http://127.0.0.1:3000/ --form-factor=mobile --chrome-flags="--headless=new"
```

Notes:

- Before the map was deferred, `/` scored 44 for performance (LCP 7.7 s, TBT 2.7 s): MapLibre
  started during hydration and its WebGL work held the headline's paint. It now loads after
  the page's `load` event and an idle callback (`components/StoryMap.tsx`).
- The remaining LCP is the headline waiting on its web font (Newsreader, self-hosted).
- Best practices is 96 on `/` only because the browser logs 404s for the district tiles of
  eras the pipeline hasn't published yet (1921, 1929 and 1935 at the time of the run). They
  go away as those eras are published.
- Headless Chrome renders WebGL in software, so the map costs more here than on a phone's GPU.
