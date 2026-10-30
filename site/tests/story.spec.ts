import { expect, test, type Page } from "@playwright/test";
import { STORY } from "../content/story";

async function centerStep(page: Page, id: string) {
  await page.evaluate((sid) => {
    const el = document.querySelector(`[data-step="${sid}"]`)!;
    const r = el.getBoundingClientRect();
    window.scrollTo(0, window.scrollY + r.top + r.height / 2 - window.innerHeight / 2);
  }, id);
}

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
