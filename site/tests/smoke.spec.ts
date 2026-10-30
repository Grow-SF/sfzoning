import { expect, test, type Page } from "@playwright/test";
import { STORY } from "../content/story";
import { SOURCES } from "../content/sources";

async function centerStep(page: Page, id: string) {
  await page.evaluate((sid) => {
    const r = document.querySelector(`[data-step="${sid}"]`)!.getBoundingClientRect();
    window.scrollBy(0, r.top + r.height / 2 - innerHeight / 2);
  }, id);
  await expect(page.locator("[data-active-step]")).toHaveAttribute("data-active-step", id);
}

test("page loads with a map canvas and story text", async ({ page }) => {
  await page.goto("/");
  await expect(page.locator("canvas").first()).toBeAttached();
  await expect(page.getByRole("heading", { name: /How San Francisco was zoned/ })).toBeVisible();
});

test("tile server down: notice appears and text still renders", async ({ page }) => {
  await page.route("http://127.0.0.1:8787/**", (r) => r.abort());
  await page.goto("/");
  await expect(page.locator("[data-active-step] .map-notice")).toContainText("could not be loaded");
  await expect(page.getByRole("heading", { name: /How San Francisco was zoned/ })).toBeVisible();
});

test("every 1948 step shows the Rumsey credit, and other steps do not", async ({ page }) => {
  // The story map's credit; the explorer below has its own map and credit.
  const credit = page.locator("[data-active-step]").getByText("© Cartography Associates, David Rumsey Map Collection", { exact: true });
  await page.goto("/");
  await expect(credit).toHaveCount(0);
  const steps1948 = STORY.filter((s) => s.era === "1948");
  expect(steps1948.length).toBeGreaterThan(0);
  for (const s of steps1948) {
    await centerStep(page, s.id);
    await expect(credit).toBeVisible();
  }
  const other = STORY.find((s) => s.era !== "1948")!;
  await centerStep(page, other.id);
  await expect(credit).toHaveCount(0);
});

test("step cards link their sources", async ({ page }) => {
  await page.goto("/");
  for (const s of STORY) {
    for (const id of s.sources) {
      await expect(page.locator(`[data-step="${s.id}"] footer li[data-source="${id}"]`)).toHaveText(SOURCES[id].short);
    }
  }
});

test("the about page covers methods, accuracy, review status and sources", async ({ page }) => {
  await page.goto("/about");
  await expect(page.getByRole("heading", { level: 1, name: "How the maps were made" })).toBeVisible();
  await expect(page.getByRole("heading", { name: /lined up with today/ })).toBeVisible();
  await expect(page.getByRole("heading", { name: "How accurate they are" })).toBeVisible();
  await expect(page.getByTestId("review-status")).toContainText("draft");
  await expect(page.locator("#src-finalenvironment2719sanf")).toContainText("Final Environmental Impact Report");
  await expect(page.getByText("© Cartography Associates, David Rumsey Map Collection").first()).toBeVisible();
});
