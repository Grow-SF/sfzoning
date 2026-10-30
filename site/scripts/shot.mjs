// Dev helper: screenshot 1948 at zoom 13. Usage: node scripts/shot.mjs <outDir>  (dev server on :3100, tiles on :8787)
import { chromium } from "@playwright/test";

const out = process.argv[2];
const b = await chromium.launch({ args: ["--use-gl=swiftshader", "--enable-unsafe-swiftshader"] });
for (const [n, w, h] of [["desktop", 1440, 900], ["mobile", 390, 844]]) {
  const p = await b.newPage({ viewport: { width: w, height: h } });
  await p.goto("http://127.0.0.1:3100/?era=1948&zoom=13");
  await p.waitForTimeout(6000);
  await p.screenshot({ path: `${out}/1948-z13-${n}.png` });
}
await b.close();
