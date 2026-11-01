// "What it allowed" palette. One palette for every era, so change between eras pops.
//
// Homes allowed per lot read as one scale of seven steps: orange-red (one home), amber, pale
// yellow, then four greens that get steadily deeper as homes rise (pale, light, mid, then the
// richest green at about 20 homes or no limit). Every green is plainly green, with no teal: the
// owner read a teal 12-homes step as almost blue and darker than the steps either side of it.
// The top step has the most chroma, because the most homes should read as the best news on the
// map, and it stays well short of near-black. Mixed-use has no colour of its own: it sits on the
// scale by how many homes it allows. The other uses step out of that scale: light blue and a
// clear mid blue for light and dense commercial, brown for industrial, purple for unrestricted.
// Public land is a quiet grey, not green, so parks never read as part of the scale or as water;
// industrial is brown, not grey, so it never reads as the scans' own ink.
//
// tests/palette.test.ts checks every pair (OKLab ΔE ×100) under normal vision and simulated
// protanopia and deuteranopia (Machado 2009), in the "allowed" view (fills over the paper
// basemap) and in the "both" view, over paper and over the scans' solid black ink. Current
// minimums (normal / protan / deutan): allowed 9.6 / 6.2 / 7.6; both over paper 8.8 / 6.2 / 6.1;
// both over ink 8.4 / 5.9 / 6.6. These are lower than the previous palette's 13.9 / 8.4 / 8.4,
// 11.3 / 7.0 / 6.9 and 11.7 / 7.5 / 7.6, on purpose: four greens of one hue can only differ by
// lightness and chroma, and a green sits close to red and orange for colour-blind readers once
// their lightness matches. The owner's look outranks the bars here. Closest pairs: neighbouring
// greens (normal), two_unit with homes_high and homes_top (protan), single_family with homes_top
// (deutan). homes_top and commercial_dense stay at least 21 apart. The scale's adjacent steps
// are 9.8 to 20.7 apart (mean 13.9, none more than 49% off it): tightest between the top two
// greens, widest from amber to yellow.
// RH-1's orange-red, the amber, both commercial blues, the brown and the purple are unchanged
// from the colours the owner approved. The yellow softened from #FFFF6C so it doesn't read as
// neon; the grey took a slight cool cast (#939AA1 to #97A1B1) to stand off the mid green.
// ALLOWED_FILL_OPACITY stays 0.85: 0.7 there would match the look exactly but lower every
// "allowed" margin to the "both" numbers, and the fills have no scan to compete with.
// The colours came from a constrained search, then a choice by eye on rendered maps.
// "Unknown" is a pale neutral near the paper: it means "not read yet", not a zoning class.
export const CATEGORIES = [
  "single_family",
  "two_unit",
  "three_unit",
  "homes_low",
  "homes_mid",
  "homes_high",
  "homes_top",
  "commercial_light",
  "commercial_dense",
  "industrial",
  "public",
  "unrestricted",
  "unknown",
] as const;

export type Category = (typeof CATEGORIES)[number];

export const CATEGORY_COLORS: Record<Category, string> = {
  single_family: "#FE4400",
  two_unit: "#F89600",
  three_unit: "#FFEE66",
  homes_low: "#A0F5BE",
  homes_mid: "#5CDA80",
  homes_high: "#2CB25A",
  homes_top: "#009600",
  commercial_light: "#89D7FF",
  commercial_dense: "#1149AF",
  industrial: "#85420B",
  public: "#97A1B1",
  unrestricted: "#8562AF",
  unknown: "#FFFFFF",
};

export const CATEGORY_LABELS: Record<Category, string> = {
  single_family: "One home",
  two_unit: "Two homes",
  three_unit: "Three homes",
  homes_low: "3-4 homes",
  homes_mid: "About 6 homes",
  homes_high: "About 12 homes",
  homes_top: "20+ or no limit",
  commercial_light: "Light commercial",
  commercial_dense: "Dense commercial",
  industrial: "Industrial",
  public: "Public",
  unrestricted: "Unrestricted",
  unknown: "Not read yet",
};

/** The scale of seven steps, by how many homes a lot may hold. */
export const HOMES_RAMP: Category[] = [
  "single_family",
  "two_unit",
  "three_unit",
  "homes_low",
  "homes_mid",
  "homes_high",
  "homes_top",
];
export const OTHER_USES: Category[] = [
  "commercial_light",
  "commercial_dense",
  "industrial",
  "public",
  "unrestricted",
  "unknown",
];

export function isCategory(v: unknown): v is Category {
  return typeof v === "string" && (CATEGORIES as readonly string[]).includes(v);
}

export const ALLOWED_FILL_OPACITY = 0.85;
export const BOTH_FILL_OPACITY = 0.7;
/**
 * Scan ink weight under the colours in "both". Half-strength ink keeps the scan's labels and
 * hatching legible, while solid-black districts (1921/1948 Commercial) read as a darker tint
 * of their own colour instead of a new one (pink over full black read as plum, next to brick).
 */
export const BOTH_SCAN_WEIGHT = 0.5;

/** Fill opacity for the district layer in each view. "both" lets the scan's ink show through. */
export function districtFillOpacity(view: "scan" | "allowed" | "both"): number {
  if (view === "allowed") return ALLOWED_FILL_OPACITY;
  if (view === "both") return BOTH_FILL_OPACITY;
  return 0;
}
