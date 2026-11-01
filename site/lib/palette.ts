// "What it allowed" palette. One palette for every era, so change between eras pops.
//
// Homes allowed per lot read as one scale of seven steps running from red (one home) through
// orange and pale yellow to deep green (about 20 homes or no limit), evenly spaced in OKLab.
// Mixed-use has no colour of its own: it sits on the scale by how many homes it allows. The
// other uses step out of that scale: light blue and dark blue for light and dense commercial,
// brown for industrial, purple for unrestricted. Public land is a quiet grey, not green, so
// parks never read as part of the scale or as water; industrial is brown, not grey, so it
// never reads as the scans' own ink.
//
// tests/palette.test.ts checks every pair (OKLab ΔE ×100) under normal vision and simulated
// protanopia and deuteranopia (Machado 2009), in the "allowed" view (fills over the paper
// basemap) and in the "both" view, over paper and over the scans' solid black ink. Current
// minimums (normal / protan / deutan): allowed 15.1 / 9.1 / 9.1; both over paper 12.1 / 8.1 /
// 8.1; both over ink 12.8 / 8.0 / 8.0. The scale's adjacent steps are 16.4 to 19.9 apart (mean
// 18.3, none more than 10% off it); its closest pair is 16.4 normal, 9.0 protan, 8.7 deutan.
// The first mockup (#DB2D27 ... #00381C) had a closest pair of 1.2 for protanopia: red and
// dark green collapse to the same dark olive. To fix it, the red turned orange-red, the deep
// greens leaned toward teal (protanopia sees blue-green apart from red), and the other uses
// moved only as far as the bars needed. The colours came from a constrained search with the
// mockup as the starting point and a penalty for every step away from it.
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
  single_family: "#FE4B00",
  two_unit: "#F29E01",
  three_unit: "#F2F26F",
  homes_low: "#6CD980",
  homes_mid: "#279B48",
  homes_high: "#006553",
  homes_top: "#002E0C",
  commercial_light: "#7ADAFF",
  commercial_dense: "#043E9F",
  industrial: "#814201",
  public: "#9F9E9C",
  unrestricted: "#8056A4",
  unknown: "#FFF5FF",
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
