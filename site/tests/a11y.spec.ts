import { expect, test, type Page } from "@playwright/test";
import { centerStep as toStep } from "./helpers";

const CREDIT = "David Rumsey Map Collection, David Rumsey Map Center, Stanford Libraries";


const FOCUSABLE = "a[href], button, input, select, textarea, [tabindex]:not([tabindex='-1'])";

test("no role=img or aria-hidden container holds focusable content", async ({ page }) => {
  await page.goto("/");
  await page.getByTestId("explorer").scrollIntoViewIfNeeded();
  await expect(page.locator(".explorer-map canvas")).toBeAttached({ timeout: 30_000 });
  await expect(page.getByTestId("stats-chart")).toBeVisible();
  const bad = await page.evaluate((sel) => {
    const out: string[] = [];
    for (const el of document.querySelectorAll("[role=img], [aria-hidden=true]")) {
      if (el.querySelector(sel)) out.push(el.outerHTML.slice(0, 120));
    }
    return out;
  }, FOCUSABLE);
  expect(bad).toEqual([]);
  await expect(page.locator(".explorer-map .map")).toHaveAttribute("role", "region");
});

test("the stats chart is described in text and backed by a table", async ({ page }) => {
  await page.goto("/");
  const chart = page.getByTestId("stats-chart");
  await chart.scrollIntoViewIfNeeded();
  await expect(chart).toBeVisible();
  await expect(chart.locator("svg").first()).toHaveAttribute("aria-hidden", "true");
  const summaryId = await chart.getAttribute("aria-describedby");
  await expect(page.locator(`#${summaryId}`)).toContainText("Every era's value is in the table");
  await expect(chart.getByRole("table")).toBeAttached();
  expect(await chart.getByRole("row").count()).toBeGreaterThan(1);
});

test("inactive story cards keep AA text contrast on desktop", async ({ page }, info) => {
  test.skip(info.project.name !== "desktop", "desktop layout only");
  await page.goto("/");
  await toStep(page, "1921-sunset");
  // Measure the cards before and after it (inactive), every text element in them.
  const worst = await page.evaluate(() => {
    const parse = (c: string) => (c.match(/[\d.]+/g) ?? []).map(Number);
    const lum = ([r, g, b]: number[]) => {
      const f = (v: number) => ((v /= 255) <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4);
      return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b);
    };
    const bg = parse(getComputedStyle(document.body).backgroundColor);
    let min = Infinity;
    let what = "";
    for (const card of document.querySelectorAll("[data-step] article:not([data-active])")) {
      for (const el of card.querySelectorAll<HTMLElement>("p, h2, blockquote, a, span, li")) {
        if (!el.textContent?.trim()) continue;
        let alpha = 1;
        for (let n: HTMLElement | null = el; n; n = n.parentElement) alpha *= Number(getComputedStyle(n).opacity);
        const [r, g, b, a = 1] = parse(getComputedStyle(el).color);
        const k = alpha * a;
        const fg = [r, g, b].map((v, i) => v * k + bg[i] * (1 - k));
        const L1 = lum(fg);
        const L2 = lum(bg);
        const ratio = (Math.max(L1, L2) + 0.05) / (Math.min(L1, L2) + 0.05);
        if (ratio < min) {
          min = ratio;
          what = `${el.tagName}.${el.className}: ${el.textContent.slice(0, 30)}`;
        }
      }
    }
    return { min, what };
  });
  expect(worst.min, worst.what).toBeGreaterThanOrEqual(4.5);
});

test.describe("narrow desktop", () => {
  test.use({ viewport: { width: 960, height: 800 } });

  test("the story key and the Rumsey credit don't overlap at 960px", async ({ page }, info) => {
    test.skip(info.project.name !== "desktop", "desktop layout only");
    await page.goto("/");
    await toStep(page, "1948-sunset");
    const story = page.locator("[data-active-step]");
    const legend = story.getByTestId("legend");
    const credit = story.getByText(CREDIT, { exact: true });
    await expect(legend).toBeVisible();
    await expect(credit).toBeVisible();
    const a = (await legend.boundingBox())!;
    const b = (await credit.boundingBox())!;
    const overlap = a.x < b.x + b.width && b.x < a.x + a.width && a.y < b.y + b.height && b.y < a.y + a.height;
    expect(overlap, JSON.stringify({ legend: a, credit: b })).toBe(false);
    if (process.env.SHOT_DIR) {
      await expect(story.locator(".map-wrap")).toHaveAttribute("data-idle", "true", { timeout: 30_000 });
      await page.waitForTimeout(900);
      await page.screenshot({ path: `${process.env.SHOT_DIR}/desktop-960-1948-legend-credit.png` });
    }
  });
});
