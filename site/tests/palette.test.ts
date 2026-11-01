import { describe, expect, it } from "vitest";
import { deltaE, hexToRgb, oklab, over, type RGB, type Vision } from "@/lib/colorcheck";
import {
  ALLOWED_FILL_OPACITY,
  BOTH_FILL_OPACITY,
  BOTH_SCAN_WEIGHT,
  CATEGORIES,
  CATEGORY_COLORS,
  HOMES_RAMP,
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
  { name: "allowed, over paper", ground: block, alpha: ALLOWED_FILL_OPACITY, min: { normal: 9.6, protan: 6.1, deutan: 7.5 }, withUnknown: true },
  { name: "both, over paper", ground: block, alpha: BOTH_FILL_OPACITY, min: { normal: 8.8, protan: 6.2, deutan: 6.1 }, withUnknown: true },
  // "Not read yet" lots carry no scan reading to sit over; skip it on ink.
  { name: "both, over black ink", ground: ink, alpha: BOTH_FILL_OPACITY, min: { normal: 8.4, protan: 5.8, deutan: 6.5 }, withUnknown: false },
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

  it("homes_top and commercial_dense are plainly apart wherever they can meet", () => {
    // The owner could not tell a dark green from a dark blue. Covers both views, and a dense
    // commercial block on the scan's black ink beside a homes_top block on paper.
    const seen = (hex: string) =>
      [
        over(hexToRgb(hex), ALLOWED_FILL_OPACITY, block),
        over(hexToRgb(hex), BOTH_FILL_OPACITY, block),
        over(hexToRgb(hex), BOTH_FILL_OPACITY, ink),
      ] as RGB[];
    for (const a of seen(CATEGORY_COLORS.homes_top))
      for (const b of seen(CATEGORY_COLORS.commercial_dense))
        for (const v of VISIONS) expect(deltaE(a, b, v), v).toBeGreaterThanOrEqual(21);
  });

  describe("the green steps: more homes, steadily deeper", () => {
    const GREENS = ["homes_low", "homes_mid", "homes_high", "homes_top"] as const;
    const VIEWS: [string, number, RGB][] = [
      ["allowed", ALLOWED_FILL_OPACITY, block],
      ["both, over paper", BOTH_FILL_OPACITY, block],
      ["both, over ink", BOTH_FILL_OPACITY, ink],
    ];
    const lab = (k: (typeof GREENS)[number], alpha: number, ground: RGB) =>
      oklab(over(hexToRgb(CATEGORY_COLORS[k]), alpha, ground));

    for (const [name, alpha, ground] of VIEWS) {
      it(`lightness strictly falls, by a visible step (${name})`, () => {
        const ls = GREENS.map((k) => lab(k, alpha, ground)[0]);
        for (let i = 1; i < ls.length; i++) expect(ls[i - 1] - ls[i], ls.join(" ")).toBeGreaterThanOrEqual(0.04);
      });
      it(`every green has a green hue, not teal or blue-leaning (${name})`, () => {
        for (const k of GREENS) {
          const [, a, b] = lab(k, alpha, ground);
          const hue = (Math.atan2(b, a) * 180) / Math.PI;
          expect(hue, k).toBeGreaterThanOrEqual(135);
          expect(hue, k).toBeLessThanOrEqual(158); // the rejected teal was 183 on paper
        }
      });
      it(`the top step has the most chroma of the greens (${name})`, () => {
        const chroma = (k: (typeof GREENS)[number]) => Math.hypot(...lab(k, alpha, ground).slice(1));
        for (const k of GREENS.slice(0, 3)) expect(chroma("homes_top"), k).toBeGreaterThan(chroma(k) + 0.02);
      });
    }

    it("the top step is friendly: neither near-black nor as light as the emerald was", () => {
      const [l] = lab("homes_top", BOTH_FILL_OPACITY, block);
      expect(l).toBeGreaterThan(0.6);
      expect(l).toBeLessThanOrEqual(0.7); // the screenshot's R-3 green is 0.69; the emerald was 0.77
    });
  });

  it("the scale runs from a red to a green", () => {
    const first = hexToRgb(CATEGORY_COLORS[HOMES_RAMP[0]]);
    const last = hexToRgb(CATEGORY_COLORS[HOMES_RAMP[HOMES_RAMP.length - 1]]);
    expect(first[0]).toBeGreaterThan(Math.max(first[1], first[2]));
    expect(last[1]).toBeGreaterThan(Math.max(last[0], last[2]));
  });

  it("dense commercial is a clear blue, not navy, even at the allowed opacity", () => {
    const [l, a, b] = oklab(over(hexToRgb(CATEGORY_COLORS.commercial_dense), ALLOWED_FILL_OPACITY, block));
    expect(l).toBeGreaterThanOrEqual(0.49); // the old navy-leaning #043E9F was 0.46
    expect(b).toBeLessThan(a);
  });

  it("steps stay apart, and none is far from the mean spacing", () => {
    const rgb = HOMES_RAMP.map((k) => hexToRgb(CATEGORY_COLORS[k]));
    const gaps = rgb.slice(1).map((c, i) => deltaE(rgb[i], c));
    const mean = gaps.reduce((s, g) => s + g, 0) / gaps.length;
    const shown = gaps.map((x) => x.toFixed(1)).join(" ");
    for (const g of gaps) {
      expect(g, shown).toBeGreaterThanOrEqual(9.5);
      expect(Math.abs(g - mean) / mean, shown).toBeLessThanOrEqual(0.5);
    }
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
