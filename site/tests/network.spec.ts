import { expect, test, type Page } from "@playwright/test";
import { centerStep as goToStep } from "./helpers";
import { ERAS } from "../lib/eras";
import { STORY } from "../content/story";

const TILES = new URL(process.env.NEXT_PUBLIC_TILES_BASE_URL ?? "http://127.0.0.1:8787");
const SCAN_SERIES = new Set(ERAS.flatMap((e) => e.scanSeries));

/** Records every request for a scan series' archive (`<series>/<series>-paper.pmtiles`). */
function recordScanRequests(page: Page) {
  const log: string[] = [];
  page.on("request", (req) => {
    const u = new URL(req.url());
    if (u.origin !== TILES.origin) return;
    const [series, file, ...rest] = u.pathname.split("/").filter(Boolean);
    if (!rest.length && SCAN_SERIES.has(series) && file === `${series}-paper.pmtiles`) log.push(series);
  });
  return log;
}

const counts = (log: string[]) => log.reduce<Record<string, number>>((c, s) => ((c[s] = (c[s] ?? 0) + 1), c), {});

async function settled(page: Page) {
  await expect(page.locator("[data-active-step] .map-wrap")).toHaveAttribute("data-idle", "true", { timeout: 30_000 });
}


/** Pan the story map so every visible layer has to fetch new tiles, then wait for it. */
async function panAndSettle(page: Page) {
  await page.evaluate(() => {
    type M = import("maplibre-gl").Map;
    (window as unknown as { __map: M }).__map.panBy([260, 180], { duration: 0 });
  });
  await page.waitForTimeout(300);
  await settled(page);
}

/** Scans the story may fetch at step i: the current era's, plus the neighbouring steps' (preloaded). */
const allowedScans = (i: number) => {
  const eras = [STORY[i - 1], STORY[i], STORY[i + 1]].filter(Boolean).map((s) => s.era);
  return new Set(eras.flatMap((id) => ERAS.find((e) => e.id === id)!.scanSeries));
};

test("the opening (today) step loads no scans but the next step's", async ({ page }, info) => {
  expect(STORY[0].era).toBe("today");
  const allowed = allowedScans(0);
  const log = recordScanRequests(page);
  await page.goto("/");
  await settled(page);
  await page.waitForTimeout(1500);
  await settled(page);
  info.annotations.push({ type: "today-step scan requests", description: JSON.stringify(counts(log)) });
  console.log(`[${info.project.name}] today step scan requests:`, log.length, counts(log));
  // Today has no scan; nothing but the neighbouring steps' scans may be fetched.
  expect(log.filter((s) => !allowed.has(s))).toEqual([]);

  log.length = 0;
  await panAndSettle(page);
  expect(log.filter((s) => !allowed.has(s))).toEqual([]);
});

test("a 1948 step loads only its own scan and the neighbouring eras'", async ({ page }) => {
  const i = STORY.findIndex((s) => s.id === "1948-sunset");
  const allowed = allowedScans(i);
  const log = recordScanRequests(page);
  await page.goto("/");
  await settled(page);
  await goToStep(page, "1948-sunset");
  // Let the camera ease and the previous era fade out (and hide), then pan to force fresh tiles.
  await page.waitForTimeout(2200);
  await settled(page);
  log.length = 0;
  await panAndSettle(page);
  expect(log.length).toBeGreaterThan(0);
  expect(log.filter((s) => !allowed.has(s))).toEqual([]);
  expect(log).toContain("1948");
});

/**
 * Records requests for the shared district tiles and for one scan series' archive (the scan of the
 * era being moved to; other scans are the following steps' preloads), and when the last was made.
 */
function recordTileRequests(page: Page, series: string) {
  let count = 0;
  let last = Date.now();
  page.on("request", (req) => {
    const u = new URL(req.url());
    if (u.origin !== TILES.origin) return;
    if (!/\/shared\/districts-[^/]+\.pmtiles$/.test(u.pathname) && !u.pathname.endsWith(`/${series}/${series}-paper.pmtiles`)) return;
    count++;
    last = Date.now();
  });
  return {
    count: () => count,
    reset: () => ((count = 0), (last = Date.now())),
    /** Resolves once no such request has started for `ms`. */
    quiet: async (ms = 1000) => {
      while (Date.now() - last < ms) await page.waitForTimeout(100);
    },
  };
}

test("the next step's shared tiles are already fetched before the reader scrolls there", async ({ page, browser }) => {
  const from = "1960-density-caps";
  const to = "1960-not-a-downzoning";
  const series = "1960";
  // Shared-tile and scan requests made while moving to `to`, once the page has gone quiet.
  const visit = async (p: Page, via?: string) => {
    const log = recordTileRequests(p, series);
    await p.goto("/");
    await settled(p);
    if (via) await goToStep(p, via);
    await log.quiet();
    await settled(p);
    log.reset();
    await goToStep(p, to);
    await log.quiet();
    await settled(p);
    return log.count();
  };
  const ahead = await visit(page, from);
  const context = await browser.newContext();
  const cold = await visit(await context.newPage());
  await context.close();
  console.log("tile requests going to", to, { ahead, cold });
  expect(cold).toBeGreaterThan(0);
  expect(ahead).toBeLessThan(cold / 2);
});

test("a scan failing outside the current era raises no notice; the current era's does", async ({ page }) => {
  await page.route("**/1948/1948-paper.pmtiles", (r) => r.fulfill({ status: 404, body: "not found" }));
  await page.goto("/");
  const notice = page.locator("[data-active-step] .map-notice");
  // 1921-mission preloads 1948, whose archive is missing: not this step's problem.
  await goToStep(page, "1921-mission");
  await page.waitForTimeout(2500);
  await expect(notice).toHaveCount(0);
  await goToStep(page, "1948-sunset");
  await expect(notice).toContainText("could not be loaded");
  await goToStep(page, "1960-density-caps");
  await expect(notice).toHaveCount(0);
});
