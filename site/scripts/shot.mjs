// Dev helper: screenshot the hero, every story step, the explorer (each view) and the about
// page at phone and desktop sizes.
// Usage: node scripts/shot.mjs <outDir> [baseUrl] [only-prefix]
//   (site running, tiles on :8787 or a published base)
// Waits for downloads to stop, then for the camera move and fades.
import fs from "node:fs";
import { chromium } from "@playwright/test";

const out = process.argv[2];
const base = process.argv[3] ?? "http://127.0.0.1:3100";
const only = process.argv[4];
fs.mkdirSync(out, { recursive: true });

/** Resolve once the map has stopped downloading and finished moving. */
async function settle(p) {
  await p.waitForTimeout(1500); // the 1.4s camera move
  await p.waitForLoadState("networkidle", { timeout: 30_000 }).catch(() => console.warn("map did not settle"));
  await p.waitForTimeout(900); // paint-property fades
}

const b = await chromium.launch({ args: ["--use-gl=swiftshader", "--enable-unsafe-swiftshader"] });
for (const [n, w, h] of [["mobile", 390, 844], ["desktop", 1440, 900]]) {
  if (only && !only.startsWith(n)) continue;
  const p = await b.newPage({ viewport: { width: w, height: h } });
  await p.goto(base);
  await settle(p);
  await p.screenshot({ path: `${out}/${n}-00-hero.png` });
  const ids = await p.$$eval("[data-step]", (els) => els.map((e) => e.getAttribute("data-step")));
  for (const [i, id] of ids.entries()) {
    await p.evaluate((sid) => {
      const el = document.querySelector(`[data-step="${sid}"] article`);
      const r = el.getBoundingClientRect();
      window.scrollTo(0, window.scrollY + r.top + r.height / 2 - window.innerHeight / 2);
    }, id);
    await p.waitForFunction((sid) => document.querySelector("[data-active-step]")?.getAttribute("data-active-step") === sid, id);
    await p.waitForTimeout(1600); // camera ease
    await settle(p);
    await p.screenshot({ path: `${out}/${n}-${String(i + 1).padStart(2, "0")}-${id}.png` });
  }
  // Mobile legend chip, opened.
  if (n === "mobile") {
    const idx = ids.indexOf("1960-density-caps");
    await p.evaluate(() => {
      const el = document.querySelector('[data-step="1960-density-caps"] article');
      const r = el.getBoundingClientRect();
      window.scrollTo(0, window.scrollY + r.top + r.height / 2 - window.innerHeight / 2);
    });
    await p.waitForTimeout(1600);
    await settle(p);
    const chip = p.locator("[data-active-step] [data-testid=legend] button").first();
    if (await chip.count()) {
      await chip.click();
      await p.waitForTimeout(300);
      await p.screenshot({ path: `${out}/${n}-${String(idx + 1).padStart(2, "0")}b-legend-open.png` });
      await chip.click();
    }
  }
  const explorer = p.getByTestId("explorer");
  await explorer.scrollIntoViewIfNeeded();
  await p.evaluate(() => document.querySelector("[data-testid=explorer]").scrollIntoView({ block: "start" }));
  await p.waitForTimeout(1500);
  for (const view of ["Both", "Original map", "What it allowed"]) {
    await explorer.getByText(view, { exact: true }).click();
    await p.evaluate(() => document.querySelector("[data-testid=explorer] canvas")?.scrollIntoView({ block: "center" }));
    await settle(p, "explorer");
    await p.screenshot({ path: `${out}/${n}-80-explorer-${view.replace(/\W+/g, "-").toLowerCase()}.png` });
  }
  await p.goto(`${base}/about`);
  await p.waitForTimeout(800);
  await p.screenshot({ path: `${out}/${n}-90-about.png` });
  await p.screenshot({ path: `${out}/${n}-91-about-full.png`, fullPage: true });
  await p.close();
}
await b.close();
