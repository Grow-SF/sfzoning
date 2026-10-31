import { expect, test, type Page } from "@playwright/test";
import { centerStep } from "./helpers";
import { ALLOWED_FILL_OPACITY, CATEGORIES, CATEGORY_COLORS, CATEGORY_LABELS } from "../lib/palette";
import { MAP_COLORS } from "../lib/theme";

const TILES = process.env.NEXT_PUBLIC_TILES_BASE_URL ?? "http://127.0.0.1:8787";
const CREDIT = "David Rumsey Map Collection, David Rumsey Map Center, Stanford Libraries";


async function tilesExist(page: Page, path: string) {
  const r = await page.request.get(`${TILES}/${path}`, { headers: { Range: "bytes=0-6" } }).catch(() => null);
  return !!r && r.ok();
}

test("the explorer's era slider is keyboard-operable", async ({ page }) => {
  await page.goto("/");
  const explorer = page.getByTestId("explorer");
  await explorer.scrollIntoViewIfNeeded();
  const slider = explorer.getByRole("slider", { name: "Year" });
  const era = explorer.getByTestId("explorer-era");
  await expect(era).toHaveText("1948");
  await slider.focus();
  await page.keyboard.press("ArrowRight");
  await expect(era).toHaveText("1960");
  await expect(slider).toHaveAttribute("aria-valuetext", "1960");
  await page.keyboard.press("End");
  await expect(era).toHaveText("Today");
  // Today has no scan, so only "What it allowed" can be chosen.
  await expect(explorer.getByRole("radio", { name: "Original map" })).toBeDisabled();
  await page.keyboard.press("Home");
  await expect(era).toHaveText("1921");
  await page.keyboard.press("ArrowLeft");
  await expect(era).toHaveText("1921");
});

test("going back to a year just viewed reuses its tiles instead of rebuilding them", async ({ page }) => {
  test.skip(!(await tilesExist(page, "1960/districts/1960-districts.pmtiles")), "1960 district tiles not published");
  const log: string[] = [];
  page.on("request", (req) => {
    // Either 1948's own archive or the shared one, whichever this device and deployment use.
    if (/\/(1948\/districts\/1948-districts|shared\/districts-[^/]+)\.pmtiles$/.test(req.url())) log.push(req.url());
  });
  await page.goto("/");
  const explorer = page.getByTestId("explorer");
  await explorer.scrollIntoViewIfNeeded();
  const wrap = explorer.locator(".map-wrap");
  await expect(wrap).toHaveAttribute("data-idle", "true", { timeout: 30_000 });
  const slider = explorer.getByRole("slider", { name: "Year" });
  const era = explorer.getByTestId("explorer-era");

  await slider.focus();
  await page.keyboard.press("ArrowRight");
  await expect(era).toHaveText("1960");
  // Long enough for 1948 to finish fading out (hiding it here is what used to drop its tiles).
  await page.waitForTimeout(1500);
  await expect(wrap).toHaveAttribute("data-idle", "true", { timeout: 30_000 });

  log.length = 0;
  await page.keyboard.press("ArrowLeft");
  await expect(era).toHaveText("1948");
  await page.waitForTimeout(500);
  await expect(wrap).toHaveAttribute("data-idle", "true", { timeout: 30_000 });
  expect(log).toEqual([]);
});

test("with the shared archive, every year switches without any tile request", async ({ page }) => {
  test.skip(!(await tilesExist(page, "shared/districts.json")), "shared archive not published");
  const log: string[] = [];
  page.on("request", (req) => {
    if (/districts.*\.pmtiles$/.test(req.url())) log.push(req.url());
  });
  await page.goto("/");
  const explorer = page.getByTestId("explorer");
  await explorer.scrollIntoViewIfNeeded();
  const wrap = explorer.locator(".map-wrap");
  await expect
    .poll(() => page.evaluate(() => (window as unknown as { __maps?: { getSource(id: string): unknown }[] }).__maps?.some((m) => m.getSource("dist-shared")) ?? false), { timeout: 30_000 })
    .toBe(true);
  await expect(wrap).toHaveAttribute("data-idle", "true", { timeout: 60_000 });
  const era = explorer.getByTestId("explorer-era");
  await explorer.getByRole("slider", { name: "Year" }).focus();
  await page.keyboard.press("Home");
  await expect(era).toHaveText("1921");
  log.length = 0;
  for (let i = 0; i < 14; i++) await page.keyboard.press("ArrowRight");
  await expect(era).toHaveText("Today");
  await page.waitForTimeout(500);
  await expect(wrap).toHaveAttribute("data-idle", "true", { timeout: 30_000 });
  // Today's colours are in the archive too, and nothing was fetched or rebuilt on the way.
  expect(log.filter((u) => !/1948\/1948|-paper/.test(u))).toEqual([]);
});

test("the 1948 'both' step shows the legend and the Rumsey credit", async ({ page }) => {
  test.skip(!(await tilesExist(page, "1948/districts/1948-districts.pmtiles")), "1948 district tiles not published");
  await page.goto("/");
  await centerStep(page, "1948-sunset");
  const story = page.locator("[data-active-step]");
  await expect(story.locator(".map-wrap")).toHaveAttribute("data-view", "both");
  await expect(story.getByTestId("legend")).toBeVisible();
  await expect(story.getByText(CREDIT, { exact: true })).toBeVisible();
});

test("on phones the story's key is a chip that opens the full key", async ({ page }, info) => {
  test.skip(info.project.name !== "mobile", "phone layout only");
  test.skip(!(await tilesExist(page, "1960/districts/1960-districts.pmtiles")), "1960 district tiles not published");
  await page.goto("/");
  await centerStep(page, "1960-density-caps");
  const legend = page.locator("[data-active-step]").getByTestId("legend");
  const chip = legend.getByRole("button", { name: "Key" });
  await expect(chip).toHaveAttribute("aria-expanded", "false");
  await expect(legend.getByText("Homes allowed per lot")).toBeHidden();
  await chip.click();
  await expect(legend.getByRole("button", { name: "Hide key" })).toHaveAttribute("aria-expanded", "true");
  await expect(legend.getByText("Homes allowed per lot")).toBeVisible();
  // The chip stays small: it must not cover the map the way the full key did.
  await legend.getByRole("button").click();
  const box = await legend.boundingBox();
  expect(box!.width * box!.height).toBeLessThan(390 * 844 * 0.03);
});

test("on phones the story camera centres each step above the card", async ({ page }, info) => {
  test.skip(info.project.name !== "mobile", "phone layout only");
  await page.goto("/");
  await centerStep(page, "1960-density-caps");
  await expect(page.locator("[data-active-step] .map-wrap")).toHaveAttribute("data-idle", "true");
  const r = await page.evaluate(() => {
    type M = import("maplibre-gl").Map;
    const m = (window as unknown as { __map: M }).__map;
    const p = m.project(m.getCenter());
    const card = document.querySelector('[data-step="1960-density-caps"] article')!.getBoundingClientRect();
    return { y: p.y, cardTop: card.top, padding: m.getPadding() };
  });
  expect(r.padding.bottom).toBeGreaterThan(300);
  expect(r.y).toBeLessThan(r.cardTop);
});

test("an era without published districts falls back to the scan with a note", async ({ page }) => {
  await page.route("**/1921/districts/**", (r) => r.fulfill({ status: 404, body: "not found" }));
  await page.goto("/");
  await centerStep(page, "1921-sunset");
  const story = page.locator("[data-active-step]");
  await expect(story.getByTestId("classification-coming")).toContainText("coming");
  await expect(story.getByTestId("legend")).toHaveCount(0);
  await expect(page.locator(".map-notice")).toHaveCount(0);
});

/** The published table and lot cards, read as the site reads them; null without shared files. */
const SHARED_DATA = `async (tiles) => {
  const r = await fetch(tiles + "/shared/districts.json");
  if (!r.ok) return null;
  const m = await r.json();
  const gz = async (path) => new Response((await fetch(tiles + "/" + path)).body.pipeThrough(new DecompressionStream("gzip")));
  const values = new Uint8Array(await (await gz(m.table)).arrayBuffer());
  const cards = await (await gz(m.cards)).json();
  const era = (series) => m.eras.indexOf(series);
  return { categoryOf: (series, lot) => { const v = values[era(series) * m.lots + lot]; return v === 255 ? null : m.categories[v]; },
           codeOf: (series, lot) => { const c = cards.eras[series].code[lot]; return c === null ? null : cards.strings[c]; } };
}`;

/** The explorer's map is created only once its data has loaded. */
const explorerMap = (page: Page) =>
  page.waitForFunction(
    () => (window as unknown as { __maps?: import("maplibre-gl").Map[] }).__maps?.some((m) => m.getContainer().closest("[data-testid=explorer]")),
    null,
    { timeout: 30_000 },
  );

/** The colour at a viewport point, as it appears on screen. */
async function pixel(page: Page, x: number, y: number): Promise<number[]> {
  const png = await page.screenshot({ clip: { x: Math.round(x), y: Math.round(y), width: 1, height: 1 } });
  return page.evaluate(async (b64) => {
    const bitmap = await createImageBitmap(await (await fetch(`data:image/png;base64,${b64}`)).blob());
    const g = new OffscreenCanvas(1, 1).getContext("2d")!;
    g.drawImage(bitmap, 0, 0);
    return Array.from(g.getImageData(0, 0, 1, 1).data.slice(0, 3));
  }, png.toString("base64"));
}

const rgb = (hex: string) => [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16));
/** A category's colour as the "allowed" view paints it over a block. */
const painted = (hex: string) => rgb(hex).map((c, i) => c * ALLOWED_FILL_OPACITY + rgb(MAP_COLORS.block)[i] * (1 - ALLOWED_FILL_OPACITY));
const distance = (a: number[], b: number[]) => Math.max(...a.map((v, i) => Math.abs(v - b[i])));

/** Scroll to the explorer, pick a view, fly to downtown and click a classified 1948 lot. */
async function openLot(page: Page, view: "What it allowed" | "Both") {
  await page.goto("/");
  const explorer = page.getByTestId("explorer");
  await explorer.scrollIntoViewIfNeeded();
  await explorer.getByText(view, { exact: true }).click();
  await expect(explorer.getByTestId("legend")).toBeVisible();
  await explorerMap(page);

  // Fly the explorer map to downtown, where the 1948 sheet has been read, and find a lot.
  const pt = await page.evaluate(async ({ tiles, sharedData }) => {
    type M = import("maplibre-gl").Map;
    const maps = (window as unknown as { __maps: M[] }).__maps;
    const map = maps.find((m) => m.getContainer().closest("[data-testid=explorer]"))!;
    const data = await (0, eval)(sharedData)(tiles);
    map.getContainer().scrollIntoView({ block: "center" });
    map.jumpTo({ center: [-122.404, 37.792], zoom: 16 });
    for (let t = 0; t < 100; t++) {
      await new Promise((r) => setTimeout(r, 200));
      const layer = map.getLayer("dist-shared-lots") ? "dist-shared-lots" : "dist-1948-fill";
      if (!map.loaded() || !map.getLayer(layer)) continue;
      const c = map.getCanvas();
      const w = c.clientWidth;
      const h = c.clientHeight;
      const r = c.getBoundingClientRect();
      for (let y = h * 0.15; y < h * 0.6; y += 12)
        for (let x = w * 0.1; x < w * 0.9; x += 12) {
          // Only points where the map itself takes the click (not the legend or a note).
          if (document.elementFromPoint(r.left + x, r.top + y) !== c) continue;
          const hits = map.queryRenderedFeatures([x, y], { layers: [layer] });
          if (layer === "dist-1948-fill") {
            const code = hits.map((f) => f.properties?.code).find(Boolean);
            if (code) return { x: r.left + x, y: r.top + y, code: String(code) };
            continue;
          }
          const lots: number[] = hits.map((f) => Number(f.properties?.i)).sort((a, b) => a - b);
          const lot = lots.find((n) => data.categoryOf("1948", n) !== null);
          const code = lot === undefined ? null : data.codeOf("1948", lot);
          // Well inside the lot: an edge pixel can be the district line's colour.
          const inside = [[-5, 0], [5, 0], [0, -5], [0, 5]].every(([dx, dy]) =>
            map.queryRenderedFeatures([x + dx, y + dy], { layers: [layer] }).some((f) => Number(f.properties?.i) === lot));
          if (code && inside) return { x: r.left + x, y: r.top + y, code: String(code) };
        }
    }
    return null;
  }, { tiles: TILES, sharedData: SHARED_DATA });
  expect(pt, "a classified 1948 lot in view").not.toBeNull();
  await page.mouse.move(pt!.x, pt!.y);
  await page.waitForTimeout(900); // the view's fade
  const before = view === "What it allowed" ? await pixel(page, pt!.x, pt!.y) : null;
  await page.mouse.click(pt!.x, pt!.y);
  const card = explorer.getByTestId("parcel-popup");
  await expect(card).toBeVisible();
  return { explorer, card, code: pt!.code, before };
}

test("a lot's colour on screen is the use its card names", async ({ page }) => {
  // The per-year files draw stacked condo lots over themselves, which darkens them.
  test.skip(!(await tilesExist(page, "shared/districts.json")), "shared files not published");
  const { card, before } = await openLot(page, "What it allowed");
  const text = (await card.textContent()) ?? "";
  const category = CATEGORIES.find((c) => text.includes(CATEGORY_LABELS[c]));
  expect(category, `a use named in: ${text}`).toBeDefined();
  const nearest = [...CATEGORIES].sort((a, b) => distance(painted(CATEGORY_COLORS[a]), before!) - distance(painted(CATEGORY_COLORS[b]), before!))[0];
  expect(nearest, `on screen: ${before}, card: ${text}`).toBe(category);
});

test("a lot an incomplete map leaves out draws no colour and opens no card", async ({ page }) => {
  test.skip(!(await tilesExist(page, "shared/districts.json")), "shared files not published");
  await page.goto("/");
  const explorer = page.getByTestId("explorer");
  await explorer.scrollIntoViewIfNeeded();
  await explorer.getByText("What it allowed", { exact: true }).click();
  const wrap = explorer.locator(".map-wrap");
  await explorerMap(page);
  // Find a lot 1948 colours and 1929 doesn't, and centre it close up.
  const pt = await page.evaluate(async ({ tiles, sharedData }) => {
    type M = import("maplibre-gl").Map;
    const map = (window as unknown as { __maps: M[] }).__maps.find((m) => m.getContainer().closest("[data-testid=explorer]"))!;
    const data = await (0, eval)(sharedData)(tiles);
    map.getContainer().scrollIntoView({ block: "center" });
    const left = (n: number) => data.categoryOf("1929", n) === null && data.categoryOf("1948", n) !== null;
    const find = async () => {
      for (let t = 0; t < 150; t++) {
        await new Promise((r) => setTimeout(r, 200));
        if (!map.loaded() || !map.getLayer("dist-shared-lots")) continue;
        const c = map.getCanvas();
        for (let y = c.clientHeight * 0.1; y < c.clientHeight * 0.9; y += 6)
          for (let x = c.clientWidth * 0.3; x < c.clientWidth * 0.9; x += 6) {
            const lots = map.queryRenderedFeatures([x, y], { layers: ["dist-shared-lots"] }).map((f) => Number(f.properties?.i));
            if (lots.length === 1 && left(lots[0])) return map.unproject([x, y]);
          }
      }
      return null;
    };
    const at = await find();
    if (!at) return null;
    map.jumpTo({ center: at, zoom: 17 });
    await new Promise((r) => setTimeout(r, 300));
    while (!map.loaded()) await new Promise((r) => setTimeout(r, 100));
    const c = map.getCanvas();
    const b = c.getBoundingClientRect();
    return { x: b.left + c.clientWidth / 2, y: b.top + c.clientHeight / 2 };
  }, { tiles: TILES, sharedData: SHARED_DATA });
  expect(pt, "a lot 1929's map leaves out").not.toBeNull();
  await expect(wrap).toHaveAttribute("data-idle", "true", { timeout: 30_000 });
  await page.waitForTimeout(900);
  const in1948 = await pixel(page, pt!.x, pt!.y);
  expect(distance(in1948, rgb(MAP_COLORS.block)), "1948 colours it").toBeGreaterThan(20);

  await explorer.getByRole("slider", { name: "Year" }).focus();
  await page.keyboard.press("Home");
  await page.keyboard.press("ArrowRight");
  await expect(explorer.getByTestId("explorer-era")).toHaveText("1929");
  await page.waitForTimeout(900);
  const in1929 = await pixel(page, pt!.x, pt!.y);
  expect(distance(in1929, rgb(MAP_COLORS.block)), "1929 shows the bare block").toBeLessThan(8);
  await page.mouse.click(pt!.x, pt!.y);
  await page.waitForTimeout(300);
  await expect(explorer.getByTestId("parcel-popup")).toHaveCount(0);
});

test("a year looks the same when you come back to it", async ({ page }) => {
  test.skip(!(await tilesExist(page, "shared/districts.json")), "shared files not published");
  await page.goto("/");
  const explorer = page.getByTestId("explorer");
  await explorer.scrollIntoViewIfNeeded();
  await explorer.getByText("What it allowed", { exact: true }).click();
  const wrap = explorer.locator(".map-wrap");
  await explorerMap(page);
  await expect(wrap).toHaveAttribute("data-idle", "true", { timeout: 60_000 });
  await page.waitForTimeout(900);
  const canvas = explorer.locator(".maplibregl-canvas");
  const first = await canvas.screenshot();
  await explorer.getByRole("slider", { name: "Year" }).focus();
  await page.keyboard.press("ArrowRight");
  await page.waitForTimeout(100);
  const right_away = await canvas.screenshot();
  await page.waitForTimeout(900);
  const next = await canvas.screenshot();
  expect(next.equals(first), "another year looks different").toBe(false);
  expect(right_away.equals(next), "the new year appears at once, with no blend of the two").toBe(true);
  await page.keyboard.press("ArrowLeft");
  await page.waitForTimeout(900);
  expect((await canvas.screenshot()).equals(first), "the same year looks the same").toBe(true);
});

test("clicking a parcel in the explorer opens its lot card", async ({ page }) => {
  test.skip(!(await tilesExist(page, "1948/districts/1948-districts.pmtiles")), "1948 district tiles not published");
  const { explorer, card, code } = await openLot(page, "What it allowed");
  await expect(card.getByRole("heading", { level: 3 })).toHaveText(code);
  await expect(card).toContainText("1948");
  await expect(card).toContainText("Homes per lot");
  await expect(card).toContainText("Map reading");
  await expect(card.getByRole("button", { name: "Close lot details" })).toBeFocused();
  await page.keyboard.press("Escape");
  await expect(card).toHaveCount(0);
  // Focus goes back to the map the lot was clicked on, not to the page body.
  await expect(explorer.locator(".maplibregl-canvas")).toBeFocused();
});

test("closing the lot card with its button returns focus to the map", async ({ page }) => {
  test.skip(!(await tilesExist(page, "1948/districts/1948-districts.pmtiles")), "1948 district tiles not published");
  const { explorer, card } = await openLot(page, "What it allowed");
  await card.getByRole("button", { name: "Close lot details" }).click();
  await expect(card).toHaveCount(0);
  await expect(explorer.locator(".maplibregl-canvas")).toBeFocused();
});

test("with a lot open on 1948 'both', the Rumsey credit stays in sight", async ({ page }) => {
  test.skip(!(await tilesExist(page, "1948/districts/1948-districts.pmtiles")), "1948 district tiles not published");
  const { explorer } = await openLot(page, "Both");
  await expect(explorer.locator(".map-wrap")).toHaveAttribute("data-view", "both");
  const credit = explorer.getByText(CREDIT, { exact: true });
  await expect(credit).toBeVisible();
  // Not just present: nothing (the lot card, the key) is drawn over it.
  const covered = await credit.evaluate((el) => {
    const r = el.getBoundingClientRect();
    const pts = [
      [r.left + 2, r.top + 2],
      [r.right - 2, r.top + 2],
      [r.left + 2, r.bottom - 2],
      [r.right - 2, r.bottom - 2],
      [(r.left + r.right) / 2, (r.top + r.bottom) / 2],
    ];
    return pts.filter(([x, y]) => !el.contains(document.elementFromPoint(x, y))).length;
  });
  expect(covered).toBe(0);
});

test.describe("touch screens", () => {
  test.use({ hasTouch: true, isMobile: true });

  test("the explorer map needs two fingers to pan, so one finger scrolls the page", async ({ page }, info) => {
    test.skip(info.project.name !== "mobile", "phone only");
    await page.goto("/");
    await page.getByTestId("explorer").scrollIntoViewIfNeeded();
    await expect(page.locator(".explorer-map canvas")).toBeAttached({ timeout: 30_000 });
    const state = await page.evaluate(() => {
      type M = import("maplibre-gl").Map;
      const maps = (window as unknown as { __maps: M[] }).__maps;
      const ex = maps.find((m) => m.getContainer().closest("[data-testid=explorer]"))!;
      return { coarse: matchMedia("(pointer: coarse)").matches, coop: ex.cooperativeGestures.isEnabled() };
    });
    expect(state).toEqual({ coarse: true, coop: true });
  });
});

test("a mouse-driven explorer map pans with one pointer", async ({ page }, info) => {
  test.skip(info.project.name !== "desktop", "desktop only");
  await page.goto("/");
  await page.getByTestId("explorer").scrollIntoViewIfNeeded();
  await expect(page.locator(".explorer-map canvas")).toBeAttached({ timeout: 30_000 });
  const coop = await page.evaluate(() => {
    type M = import("maplibre-gl").Map;
    const maps = (window as unknown as { __maps: M[] }).__maps;
    return maps.find((m) => m.getContainer().closest("[data-testid=explorer]"))!.cooperativeGestures.isEnabled();
  });
  expect(coop).toBe(false);
});
