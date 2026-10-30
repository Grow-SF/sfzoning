// Regenerates app/opengraph-image.png (1200×630; convert to .jpg with `sips -s format jpeg`): the opening step's map of today's zoning,
// screenshotted from the running site. Usage: node scripts/og-image.mjs [baseUrl]
import path from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "@playwright/test";

const base = process.argv[2] ?? "http://127.0.0.1:3000";
const out = path.join(path.dirname(fileURLToPath(import.meta.url)), "../app/opengraph-image.png");

const b = await chromium.launch({ args: ["--use-gl=swiftshader", "--enable-unsafe-swiftshader"] });
// Desktop layout: the map pane is the right 62% of the window, so 1936 px wide gives a
// 1200 px map. The share image is the map alone, without the ruler or key.
const p = await b.newPage({ viewport: { width: 1936, height: 760 } });
await p.goto(base);
await p.addStyleTag({ content: "[aria-label=Eras], [data-testid=legend], .draft-banner { display: none !important; }" });
await p.waitForFunction(() => document.querySelector(".map-wrap")?.getAttribute("data-idle") === "true", null, { timeout: 60_000 });
await p.waitForTimeout(1000);
const box = await p.locator(".map-wrap").first().boundingBox();
await p.screenshot({ path: out, clip: { x: Math.round(box.x + (box.width - 1200) / 2), y: 70, width: 1200, height: 630 } });
await b.close();
console.log("wrote", out);
