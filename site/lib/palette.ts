// "What it allowed" palette. One palette for every era, so change between eras pops.
//
// Homes allowed per lot read as one scale of seven steps: orange-red (one home), amber, pale
// yellow, pale mint, green, teal, then a bright emerald green (about 20 homes or no limit). The
// top step is the most vivid colour on the scale, because the most homes should read as the
// best news on the map; it is deliberately not the darkest. Mixed-use has no colour of its own:
// it sits on the scale by how many homes it allows. The other uses step out of that scale: light
// blue and a clear mid blue for light and dense commercial, brown for industrial, purple for
// unrestricted. Public land is a quiet grey, not green, so parks never read as part of the scale
// or as water; industrial is brown, not grey, so it never reads as the scans' own ink.
//
// tests/palette.test.ts checks every pair (OKLab ΔE ×100) under normal vision and simulated
// protanopia and deuteranopia (Machado 2009), in the "allowed" view (fills over the paper
// basemap) and in the "both" view, over paper and over the scans' solid black ink. Current
// minimums (normal / protan / deutan): allowed 13.9 / 8.4 / 8.4; both over paper 11.3 / 7.0 /
// 6.9; both over ink 11.7 / 7.5 / 7.6. Bars: allowed 13.5 / 8 / 8; both over paper 11 / 6.9 /
// 6.5; both over ink 11.5 / 7.4 / 7.5. These were 15 / 9 / 9 and 12 / 8 / 8 while the top step
// was near-black. A bright top step sits at the same lightness as the steps it must differ from,
// so the bars came down rather than move the colours the owner liked. Closest pairs: homes_mid
// and homes_top (normal), two_unit and homes_top (protan, deutan), homes_top and public (deutan,
// both over paper). The scale's adjacent steps are 15.5 to 32.4 apart (mean 22.4, none more
// than 45% off it): tighter between orange-red and amber, yellow and mint, and green and teal,
// wider either side of the mint and up to the top step.
// Against the on-screen colours the owner praised (1960 Noe, "both", #279B48 #006553 #043E9F
// #7ADAFF #814201 #FE4B00 #F29E01 at 0.7 over paper), every one moved by 2.9 ΔE or less at 0.7.
// ALLOWED_FILL_OPACITY stays 0.85: 0.7 there would match the look exactly but lower every
// "allowed" margin to the "both" numbers, and the fills have no scan to compete with.
// The colours came from a constrained search over this file's starting colours.
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
  three_unit: "#FFFF6C",
  homes_low: "#77F7B0",
  homes_mid: "#3B9246",
  homes_high: "#00645B",
  homes_top: "#00CD5F",
  commercial_light: "#89D7FF",
  commercial_dense: "#1149AF",
  industrial: "#85420B",
  public: "#939AA1",
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
