import { expect, test, type Page } from "@playwright/test";
import { scrollToStep as centerStep } from "./helpers";
import { STORY } from "../content/story";


test("scrolling through the story activates every step in turn", async ({ page }) => {
  await page.goto("/");
  const story = page.locator("[data-active-step]");
  await expect(story).toHaveAttribute("data-active-step", STORY[0].id);
  for (const s of STORY) {
    await centerStep(page, s.id);
    await expect(story).toHaveAttribute("data-active-step", s.id);
    await expect(page.locator(`[data-step="${s.id}"]`)).toHaveAttribute("data-active", "true");
  }
  // And back up again.
  for (const s of [...STORY].reverse()) {
    await centerStep(page, s.id);
    await expect(story).toHaveAttribute("data-active-step", s.id);
  }
});

test("the story draws its colours from the shared tiles", async ({ page }) => {
  const perYear: string[] = [];
  const shared: string[] = [];
  page.on("request", (req) => {
    if (/\/districts\/[^/]*-districts\.pmtiles/.test(req.url())) perYear.push(req.url());
    if (/\/shared\/districts-[^/]+\.pmtiles/.test(req.url())) shared.push(req.url());
  });
  await page.goto("/");
  await centerStep(page, "1921-sunset");
  await expect(page.locator("[data-active-step] .map-wrap")).toHaveAttribute("data-idle", "true", { timeout: 60_000 });
  expect(perYear).toEqual([]);
  expect(shared.length).toBeGreaterThan(0);
});

test.describe("phone trigger", () => {
  test.use({ viewport: { width: 390, height: 844 } });

  test("the map switches as the new card comes up past 70% of the screen, not before", async ({ page }, info) => {
    test.skip(info.project.name !== "mobile", "phone layout only");
    await page.goto("/");
    const story = page.locator("[data-active-step]");
    const wrap = story.locator(".map-wrap");
    const putCardTopAt = (id: string, frac: number) =>
      page.evaluate(
        ([sid, f]) => {
          const r = document.querySelector(`[data-step="${sid}"] article`)!.getBoundingClientRect();
          window.scrollBy(0, r.top - innerHeight * (f as number));
        },
        [id, frac] as const,
      );
    for (const [prev, next] of [
      ["today-opening", "1921-sunset"],
      ["1921-mission", "1948-sunset"],
    ]) {
      const prevEra = STORY.find((s) => s.id === prev)!.era;
      const nextEra = STORY.find((s) => s.id === next)!.era;
      // The new card is on screen, its top below the trigger line: still the previous step.
      await putCardTopAt(next, 0.8);
      await page.waitForTimeout(400);
      await expect(story).toHaveAttribute("data-active-step", prev);
      await expect(wrap).toHaveAttribute("data-era", prevEra);
      const top = await page.locator(`[data-step="${next}"] article`).evaluate((el) => el.getBoundingClientRect().top);
      expect(top).toBeLessThan(844); // the card really is in view
      // Its top crosses 70%: the step, and the map, change.
      await putCardTopAt(next, 0.65);
      await expect(story).toHaveAttribute("data-active-step", next);
      await expect(wrap).toHaveAttribute("data-era", nextEra);
    }
  });
});

test.describe("small phone", () => {
  test.use({ viewport: { width: 360, height: 740 } });

  test("the longest step's card is fully readable via its own scroll", async ({ page }, info) => {
    test.skip(info.project.name !== "mobile", "phone layout only");
    await page.goto("/");
    const id = await page.$$eval("[data-step]", (els) => {
      let best = els[0];
      for (const el of els) {
        const h = (e: Element) => e.querySelector("article > div")!.scrollHeight;
        if (h(el) > h(best)) best = el;
      }
      return best.getAttribute("data-step")!;
    });
    await centerStep(page, id);
    await expect(page.locator("[data-active-step]")).toHaveAttribute("data-active-step", id);

    const scroller = page.locator(`[data-step="${id}"] article > div`);
    const m = await scroller.evaluate((el) => {
      const card = el.parentElement!.getBoundingClientRect();
      return { top: card.top, bottom: card.bottom, sh: el.scrollHeight, ch: el.clientHeight, vh: innerHeight };
    });
    // The card box itself sits wholly on screen, below the era ruler, within 55% of the viewport.
    expect(m.top).toBeGreaterThanOrEqual(56);
    expect(m.bottom).toBeLessThanOrEqual(m.vh);
    expect(m.bottom - m.top).toBeLessThanOrEqual(m.vh * 0.55 + 1);
    expect(m.sh).toBeGreaterThan(m.ch); // long enough to need its own scroll

    // Scroll the card itself (wheel over it); the last line must come into the card's view.
    await scroller.hover();
    for (let i = 0; i < 10; i++) await page.mouse.wheel(0, 200);
    await expect
      .poll(() =>
        scroller.evaluate((el) => {
          const last = el.querySelector("footer li:last-child")!.getBoundingClientRect();
          const box = el.getBoundingClientRect();
          return last.bottom <= box.bottom + 1 && last.top >= box.top;
        }),
      )
      .toBe(true);
  });
});

const visibleScans = (page: Page) =>
  page.evaluate(() => {
    const m = (window as unknown as { __map: import("maplibre-gl").Map }).__map;
    return m
      .getStyle()
      .layers.filter((l) => l.id.startsWith("scan-") && m.getLayoutProperty(l.id, "visibility") === "visible")
      .map((l) => l.id);
  });

test("scrolling fast through the story leaves only the last step's scan showing", async ({ page }) => {
  await page.goto("/");
  await expect(page.locator("[data-active-step]")).toHaveAttribute("data-active-step", STORY[0].id);
  const ids = ["1921-sunset", "1948-sunset", "1960-density-caps", "1972-height-limits", "1975-downzonings"];
  for (const id of ids) await centerStep(page, id);
  const last = STORY.find((s) => s.id === ids[ids.length - 1])!;
  await expect(page.locator("[data-active-step]")).toHaveAttribute("data-active-step", last.id);
  await expect(page.locator("[data-active-step] .map-wrap")).toHaveAttribute("data-idle", "true", { timeout: 60_000 });
  await page.waitForTimeout(2000);
  expect(await visibleScans(page)).toEqual([`scan-${last.era}`]);
});

/** The share of the story map's pixels that are strongly coloured (the district colours, not the scan's ink). */
async function colouredShare(page: Page) {
  const shot = (await page.locator("[data-active-step] .map-wrap").screenshot()).toString("base64");
  return await page.evaluate(async (b64) => {
    const img = new Image();
    img.src = `data:image/png;base64,${b64}`;
    await img.decode();
    const c = document.createElement("canvas");
    c.width = img.width;
    c.height = img.height;
    const g = c.getContext("2d")!;
    g.drawImage(img, 0, 0);
    const d = g.getImageData(0, 0, c.width, c.height).data;
    let n = 0;
    for (let i = 0; i < d.length; i += 4) if (Math.max(d[i], d[i + 1], d[i + 2]) - Math.min(d[i], d[i + 1], d[i + 2]) > 60) n++;
    return n / (d.length / 4);
  }, shot);
}

const delayed = (ms: number) => async (route: import("@playwright/test").Route) => {
  await new Promise((r) => setTimeout(r, ms));
  await route.continue().catch(() => {});
};

/** A route that never answers: the request stays open, unlike a 404, which ends a preload. */
const never = () => {};

const activeWrap = (page: Page) => page.locator("[data-active-step] .map-wrap");

async function settleAt(page: Page, id: string) {
  await centerStep(page, id);
  await expect(page.locator("[data-active-step]")).toHaveAttribute("data-active-step", id);
  await expect(activeWrap(page)).toHaveAttribute("data-idle", "true", { timeout: 60_000 });
}

test("a scan that never arrives holds the old one up to the limit, then shows colours", async ({ page }) => {
  await page.route("**/1948/1948-paper.pmtiles", never);
  await page.goto("/");
  await settleAt(page, "1921-mission");
  await centerStep(page, "1948-sunset");
  await expect(page.locator("[data-active-step]")).toHaveAttribute("data-active-step", "1948-sunset");
  await page.waitForTimeout(3000);
  expect(await visibleScans(page), "the old scan is held").toContain("scan-1921");
  await page.waitForTimeout(6500); // past the 8s limit, and the 300ms exit
  expect(await visibleScans(page), "the limit ended the hold").not.toContain("scan-1921");
  await expect(page.locator("[data-active-step]").getByTestId("legend")).toBeVisible();
  expect(await colouredShare(page)).toBeGreaterThan(0.02);
});

test("a new camera move on the same scans restarts the hold's limit", async ({ page }) => {
  await page.route("**/1960/1960-paper.pmtiles", never);
  await page.goto("/");
  await settleAt(page, "1948-sunset");
  await centerStep(page, "1960-density-caps");
  await expect(page.locator("[data-active-step]")).toHaveAttribute("data-active-step", "1960-density-caps");
  await page.waitForTimeout(6000);
  await centerStep(page, "1960-not-a-downzoning");
  await expect(page.locator("[data-active-step]")).toHaveAttribute("data-active-step", "1960-not-a-downzoning");
  await page.waitForTimeout(3500); // ~9.5s since the hold began, ~3.5s since the camera moved again
  expect(await visibleScans(page), "still held, counting from the last move").toContain("scan-1948");
  await page.waitForTimeout(5500);
  expect(await visibleScans(page), "released by the limit").not.toContain("scan-1948");
});

test("the Rumsey credit stays while the 1948 scan is held under 1960, and goes with it", async ({ page }) => {
  await page.route("**/1960/1960-paper.pmtiles", never);
  await page.goto("/");
  await settleAt(page, "1948-sunset");
  const credit = page.locator("[data-active-step]").getByText("David Rumsey Map Collection, David Rumsey Map Center, Stanford Libraries", { exact: true });
  await expect(credit).toBeVisible();
  await centerStep(page, "1960-density-caps");
  await expect(activeWrap(page)).toHaveAttribute("data-era", "1960");
  await page.waitForTimeout(3000);
  expect(await visibleScans(page)).toContain("scan-1948");
  await expect(credit).toBeVisible();
  await expect(credit).toHaveCount(0, { timeout: 10_000 });
  expect(await visibleScans(page)).not.toContain("scan-1948");
});

test("a preload that failed doesn't count as loaded, and doesn't break that scan afterwards", async ({ page }) => {
  // Every read of the 1960 scan fails while the neighbours preload.
  let failed = 0;
  await page.route("**/1960/1960-paper.pmtiles", (route) => (failed++, route.abort()));
  await page.goto("/");
  await settleAt(page, "1948-sunset");
  await expect.poll(() => failed, { timeout: 30_000 }).toBeGreaterThan(0);
  await page.waitForTimeout(500);
  await page.unroute("**/1960/1960-paper.pmtiles");
  await page.route("**/1960/1960-paper.pmtiles", delayed(3000));
  await centerStep(page, "1960-density-caps");
  await expect(activeWrap(page)).toHaveAttribute("data-era", "1960");
  await page.waitForTimeout(2200); // the camera has stopped; the scan has not arrived
  expect(await visibleScans(page), "held, since the preload failed").toContain("scan-1948");
  await expect
    .poll(() => page.evaluate(() => (window as unknown as { __map: import("maplibre-gl").Map }).__map.isSourceLoaded("scan-1960")), { timeout: 45_000 })
    .toBe(true);
  await expect(page.locator("[data-active-step] .map-notice")).toHaveCount(0);
});

test("the old scan stays under a new one whose tiles are slow", async ({ page }) => {
  await page.route("**/1948/1948-paper.pmtiles", delayed(6000));
  await page.goto("/");
  await centerStep(page, "1921-mission");
  await expect(page.locator("[data-active-step]")).toHaveAttribute("data-active-step", "1921-mission");
  await page.waitForTimeout(3000);
  await centerStep(page, "1948-sunset");
  await page.waitForTimeout(2000);
  const old = await page.evaluate(() => {
    const m = (window as unknown as { __map: import("maplibre-gl").Map }).__map;
    const o = m.getPaintProperty("scan-1921", "raster-opacity");
    return { visibility: m.getLayoutProperty("scan-1921", "visibility"), solid: typeof o !== "number" || o > 0 };
  });
  expect(old).toEqual({ visibility: "visible", solid: true });
});

test("going from colours to a scan alone keeps the colours until the scan arrives", async ({ page }) => {
  await page.route("**/1980-1990-zoning/*paper.pmtiles", delayed(6000));
  await page.goto("/");
  await centerStep(page, "1978-legacy");
  await expect(page.locator("[data-active-step]")).toHaveAttribute("data-active-step", "1978-legacy");
  await expect(page.locator("[data-active-step] .map-wrap")).toHaveAttribute("data-idle", "true", { timeout: 60_000 });
  await page.waitForTimeout(1500);
  await centerStep(page, "1985-downtown-plan");
  await page.waitForTimeout(3500); // the camera has stopped; the scan has not arrived
  const coloured = await colouredShare(page);
  expect(coloured).toBeGreaterThan(0.02);
});

test("when the next scan is preloaded the colours fade during the pan, not after it", async ({ page }) => {
  let lastTile = Date.now();
  page.on("request", (req) => {
    if (/\.pmtiles$/.test(new URL(req.url()).pathname)) lastTile = Date.now();
  });
  await page.goto("/");
  await centerStep(page, "1960-density-caps");
  await centerStep(page, "1972-height-limits");
  await centerStep(page, "1975-downzonings");
  await centerStep(page, "1978-rezoning");
  await centerStep(page, "1978-legacy");
  // Idle, then the neighbours' preload (which starts at idle) has finished: no tile request for 1s.
  await page.waitForTimeout(300);
  await expect(page.locator("[data-active-step] .map-wrap")).toHaveAttribute("data-idle", "true", { timeout: 60_000 });
  while (Date.now() - lastTile < 1000) await page.waitForTimeout(100);
  const before = await colouredShare(page);
  expect(before).toBeGreaterThan(0.02);
  await centerStep(page, "1985-downtown-plan");
  await page.waitForTimeout(900); // the screenshot itself takes a few hundred ms: ~1.3s into the pan
  const during = await colouredShare(page);
  expect(during).toBeLessThan(before / 2);
});
