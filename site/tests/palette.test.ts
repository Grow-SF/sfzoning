import { describe, expect, it } from "vitest";
import { deltaE, hexToRgb, over, type RGB, type Vision } from "@/lib/colorcheck";
import {
  ALLOWED_FILL_OPACITY,
  BOTH_FILL_OPACITY,
  BOTH_SCAN_WEIGHT,
  CATEGORIES,
  CATEGORY_COLORS,
  type Category,
} from "@/lib/palette";
import { MAP_COLORS } from "@/lib/theme";

// Every pair of map colours must stay apart for readers with protanopia or deuteranopia,
// as the fills are actually seen: over the paper basemap ("allowed"), and in "both" over
// paper and over the scans' solid black ink (at the scan weight StoryMap uses there).

const block = hexToRgb(MAP_COLORS.block);
const ink = over([0, 0, 0], BOTH_SCAN_WEIGHT, block);
const VISIONS: Vision[] = ["normal", "protan", "deutan"];

type Case = { name: string; ground: RGB; alpha: number; min: Record<Vision, number>; withUnknown: boolean };
const CASES: Case[] = [
  { name: "allowed, over paper", ground: block, alpha: ALLOWED_FILL_OPACITY, min: { normal: 15, protan: 9, deutan: 9 }, withUnknown: true },
  { name: "both, over paper", ground: block, alpha: BOTH_FILL_OPACITY, min: { normal: 12, protan: 8, deutan: 8 }, withUnknown: true },
  // "Not read yet" lots carry no scan reading to sit over; skip it on ink.
  { name: "both, over black ink", ground: ink, alpha: BOTH_FILL_OPACITY, min: { normal: 12, protan: 8, deutan: 8 }, withUnknown: false },
];

function worstPair(c: Case, vision: Vision) {
  const cats = CATEGORIES.filter((k) => c.withUnknown || k !== "unknown");
  let worst = { d: Infinity, pair: "" };
  for (let i = 0; i < cats.length; i++)
    for (let j = i + 1; j < cats.length; j++) {
      const a = over(hexToRgb(CATEGORY_COLORS[cats[i]]), c.alpha, c.ground);
      const b = over(hexToRgb(CATEGORY_COLORS[cats[j]]), c.alpha, c.ground);
      const d = deltaE(a, b, vision);
      if (d < worst.d) worst = { d, pair: `${cats[i]} vs ${cats[j]}` };
    }
  return worst;
}

describe("palette colour-blind safety", () => {
  for (const c of CASES)
    for (const v of VISIONS)
      it(`${c.name}: every pair ≥ ${c.min[v]} ΔE (${v})`, () => {
        const w = worstPair(c, v);
        expect(w.d, w.pair).toBeGreaterThanOrEqual(c.min[v]);
      });

  it("commercial over black ink does not read as apartments over paper", () => {
    // The S4 review: 1948 Commercial (solid black) tinted pink read plum, next to the brick.
    const cm = over(hexToRgb(CATEGORY_COLORS.commercial), BOTH_FILL_OPACITY, ink);
    const mf = over(hexToRgb(CATEGORY_COLORS.multi_family), BOTH_FILL_OPACITY, block);
    for (const v of VISIONS) expect(deltaE(cm, mf, v), v).toBeGreaterThanOrEqual(15);
  });

  it("the homes ramp gets darker with every step", () => {
    const ramp: Category[] = ["single_family", "two_unit", "three_unit", "multi_family"];
    const L = ramp.map((k) => deltaE(hexToRgb(CATEGORY_COLORS[k]), [0, 0, 0]));
    for (let i = 1; i < L.length; i++) expect(L[i]).toBeLessThan(L[i - 1]);
  });
});

describe("colorcheck", () => {
  it("measures identical colours as 0 and black/white as 100", () => {
    expect(deltaE([0.5, 0.2, 0.1], [0.5, 0.2, 0.1])).toBe(0);
    expect(deltaE([0, 0, 0], [1, 1, 1])).toBeCloseTo(100, 0);
  });
  it("collapses red and green under deuteranopia", () => {
    const red = hexToRgb("#D03030");
    const green = hexToRgb("#6F8F00");
    expect(deltaE(red, green, "deutan")).toBeLessThan(deltaE(red, green, "normal") / 2);
  });
});
