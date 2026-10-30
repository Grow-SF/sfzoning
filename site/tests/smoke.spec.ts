import { expect, test } from "@playwright/test";

test("page loads with a map canvas and story text", async ({ page }) => {
  await page.goto("/");
  await expect(page.locator("canvas").first()).toBeAttached();
  await expect(page.getByRole("heading", { name: /How San Francisco was zoned/ })).toBeVisible();
});

test("tile server down: notice appears and text still renders", async ({ page }) => {
  await page.route("http://127.0.0.1:8787/**", (r) => r.abort());
  await page.goto("/");
  await expect(page.locator(".map-notice")).toContainText("could not be loaded");
  await expect(page.getByRole("heading", { name: /How San Francisco was zoned/ })).toBeVisible();
});

test("the 1948 step shows the Rumsey credit", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByText("David Rumsey Map Collection")).toHaveCount(0);
  await page.locator('[data-step="1948-lot-rules"]').scrollIntoViewIfNeeded();
  await page.evaluate(() => {
    const r = document.querySelector('[data-step="1948-lot-rules"]')!.getBoundingClientRect();
    window.scrollBy(0, r.top + r.height / 2 - innerHeight / 2);
  });
  await expect(page.getByText("David Rumsey Map Collection")).toBeVisible();
});
