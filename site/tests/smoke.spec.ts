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

test("1948 shows imagery and the Rumsey credit", async ({ page }) => {
  await page.goto("/?era=1948&zoom=13");
  await expect(page.getByText("David Rumsey Map Collection")).toBeVisible();
});
