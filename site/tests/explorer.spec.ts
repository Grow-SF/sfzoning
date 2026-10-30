import { expect, test, type Page } from "@playwright/test";

const TILES = process.env.NEXT_PUBLIC_TILES_BASE_URL ?? "http://127.0.0.1:8787";
const CREDIT = "© Cartography Associates, David Rumsey Map Collection";

async function centerStep(page: Page, id: string) {
  await page.evaluate((sid) => {
    const r = document.querySelector(`[data-step="${sid}"] article`)!.getBoundingClientRect();
    window.scrollBy(0, r.top + r.height / 2 - innerHeight / 2);
  }, id);
  await expect(page.locator("[data-active-step]")).toHaveAttribute("data-active-step", id);
}

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

test("clicking a parcel in the explorer opens its lot card", async ({ page }) => {
  test.skip(!(await tilesExist(page, "1948/districts/1948-districts.pmtiles")), "1948 district tiles not published");
  await page.goto("/");
  const explorer = page.getByTestId("explorer");
  await explorer.scrollIntoViewIfNeeded();
  await explorer.getByText("What it allowed", { exact: true }).click();
  await expect(explorer.getByTestId("legend")).toBeVisible();

  // Fly the explorer map to downtown, where the 1948 sheet has been read, and find a lot.
  const pt = await page.evaluate(async () => {
    type M = import("maplibre-gl").Map;
    const maps = (window as unknown as { __maps: M[] }).__maps;
    const map = maps.find((m) => m.getContainer().closest("[data-testid=explorer]"))!;
    map.getContainer().scrollIntoView({ block: "center" });
    map.jumpTo({ center: [-122.404, 37.792], zoom: 16 });
    for (let t = 0; t < 100; t++) {
      await new Promise((r) => setTimeout(r, 200));
      if (!map.loaded() || !map.getLayer("dist-1948-fill")) continue;
      const c = map.getCanvas();
      const w = c.clientWidth;
      const h = c.clientHeight;
      const r = c.getBoundingClientRect();
      for (let y = h * 0.15; y < h * 0.6; y += 12)
        for (let x = w * 0.1; x < w * 0.9; x += 12) {
          // Only points where the map itself takes the click (not the legend or a note).
          if (document.elementFromPoint(r.left + x, r.top + y) !== c) continue;
          const f = map.queryRenderedFeatures([x, y], { layers: ["dist-1948-fill"] })[0];
          if (f?.properties?.code) {
            return { x: r.left + x, y: r.top + y, code: String(f.properties.code) };
          }
        }
    }
    return null;
  });
  expect(pt, "a classified 1948 lot in view").not.toBeNull();
  await page.mouse.click(pt!.x, pt!.y);
  const card = explorer.getByTestId("parcel-popup");
  await expect(card).toBeVisible();
  await expect(card.getByRole("heading", { level: 3 })).toHaveText(pt!.code);
  await expect(card).toContainText("1948");
  await expect(card).toContainText("Homes per lot");
  await expect(card).toContainText("Map reading");
  await page.keyboard.press("Escape");
  await expect(card).toHaveCount(0);
});
