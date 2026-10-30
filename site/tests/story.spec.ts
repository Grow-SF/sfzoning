import { expect, test, type Page } from "@playwright/test";
import { STORY } from "../content/story";

async function centerStep(page: Page, id: string) {
  await page.evaluate((sid) => {
    const el = document.querySelector(`[data-step="${sid}"] article`)!;
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
