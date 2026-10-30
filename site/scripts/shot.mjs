// Dev helper: screenshot the hero and every story step at phone and desktop sizes.
// Usage: node scripts/shot.mjs <outDir> [baseUrl]   (dev server running, tiles on :8787)
import fs from "node:fs";
import { chromium } from "@playwright/test";

const out = process.argv[2];
const base = process.argv[3] ?? "http://127.0.0.1:3100";
fs.mkdirSync(out, { recursive: true });

const b = await chromium.launch({ args: ["--use-gl=swiftshader", "--enable-unsafe-swiftshader"] });
for (const [n, w, h] of [["mobile", 390, 844], ["desktop", 1440, 900]]) {
  const p = await b.newPage({ viewport: { width: w, height: h } });
  await p.goto(base);
  await p.waitForTimeout(4000);
  await p.screenshot({ path: `${out}/${n}-0-hero.png` });
  const ids = await p.$$eval("[data-step]", (els) => els.map((e) => e.getAttribute("data-step")));
  for (const [i, id] of ids.entries()) {
    await p.evaluate((sid) => {
      const el = document.querySelector(`[data-step="${sid}"]`);
      const r = el.getBoundingClientRect();
      window.scrollTo(0, window.scrollY + r.top + r.height / 2 - window.innerHeight / 2);
    }, id);
    await p.waitForFunction((sid) => document.querySelector("[data-active-step]")?.getAttribute("data-active-step") === sid, id);
    await p.waitForTimeout(5000); // camera ease + tiles
    await p.screenshot({ path: `${out}/${n}-${i + 1}-${id}.png` });
  }
  await p.close();
}
await b.close();
