import { defineConfig } from "@playwright/test";

// Tile server is started by the test config so the happy path has real tiles. Set
// NEXT_PUBLIC_TILES_BASE_URL to the published tiles to test without a local copy: then no local
// tile server is started (contributors without pipeline/data/out run the suite this way).
// The "tiles down" test aborts every request to the tile server with page.route().
// NEXT_PUBLIC_E2E=1 turns on the map test hooks (window.__map/__maps, data-idle); production
// builds leave them out. A reused dev server must have been started with it too.
// SITE_PORT lets parallel checkouts run their own dev server instead of reusing another's.
const port = Number(process.env.SITE_PORT ?? 3100);
const remoteTiles = Boolean(process.env.NEXT_PUBLIC_TILES_BASE_URL);

export default defineConfig({
  testDir: "./tests",
  testMatch: /.*\.spec\.ts/,
  timeout: 90_000,
  // Every spec drives a WebGL map in software (SwiftShader). Two workers starve each other's
  // main thread (and the dev server's compiler) of CPU, so scroll-driven IntersectionObserver
  // callbacks and map loads miss their deadlines: the story-scroll spec flaked at 2 workers.
  // One worker is deterministic and still finishes in a few minutes.
  workers: 1,
  fullyParallel: false,
  retries: 0,
  expect: { timeout: 15_000 },
  use: { baseURL: `http://127.0.0.1:${port}` },
  projects: [
    { name: "mobile", use: { viewport: { width: 390, height: 844 } } },
    { name: "desktop", use: { viewport: { width: 1440, height: 900 } } },
  ],
  webServer: [
    ...(remoteTiles
      ? []
      : [{ command: "node scripts/serve-tiles.mjs", url: "http://127.0.0.1:8787/ref/blocks.pmtiles", reuseExistingServer: true }]),
    { command: `npm run dev -- -p ${port}`, url: `http://127.0.0.1:${port}`, reuseExistingServer: true, timeout: 120_000, env: { NEXT_PUBLIC_E2E: "1" } },
  ],
});
