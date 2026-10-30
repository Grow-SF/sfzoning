import { expect, type Page } from "@playwright/test";

/**
 * Scroll a story step into its reading position: its section filling the screen (on phones
 * the card then rests at the bottom). A step starts when its card's top crosses 70% of the
 * viewport, so if a short card would rest below that line, scroll on until it sits at 60%.
 */
export async function scrollToStep(page: Page, id: string) {
  await page.evaluate((sid) => {
    const section = document.querySelector(`[data-step="${sid}"]`)!;
    const r = section.getBoundingClientRect();
    window.scrollBy(0, r.top + r.height / 2 - innerHeight / 2);
    const top = section.querySelector("article")!.getBoundingClientRect().top;
    if (top > innerHeight * 0.65) window.scrollBy(0, top - innerHeight * 0.6);
  }, id);
}

/** scrollToStep, then wait for the story to report that step active. */
export async function centerStep(page: Page, id: string) {
  await scrollToStep(page, id);
  await expect(page.locator("[data-active-step]")).toHaveAttribute("data-active-step", id);
}
