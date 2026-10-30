import { defineConfig } from "@playwright/test";

// Tile server is started by the test config so the happy path has real tiles.
// The "tiles down" test points the page at a dead port via ?tiles= override.
// SITE_PORT lets parallel checkouts run their own dev server instead of reusing another's.
const port = Number(process.env.SITE_PORT ?? 3100);

export default defineConfig({
  testDir: "./tests",
  testMatch: /.*\.spec\.ts/,
  timeout: 60_000,
  use: { baseURL: `http://127.0.0.1:${port}` },
  projects: [
    { name: "mobile", use: { viewport: { width: 390, height: 844 } } },
    { name: "desktop", use: { viewport: { width: 1440, height: 900 } } },
  ],
  webServer: [
    { command: "node scripts/serve-tiles.mjs", url: "http://127.0.0.1:8787/ref/blocks.pmtiles", reuseExistingServer: true },
    { command: `npm run dev -- -p ${port}`, url: `http://127.0.0.1:${port}`, reuseExistingServer: true, timeout: 120_000 },
  ],
});
