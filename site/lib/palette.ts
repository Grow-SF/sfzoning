import { MAP_COLORS } from "./theme";

// "What it allowed" palette. One palette for every era, so change between eras pops.
//
// Homes allowed per lot read as one scale of seven steps: dusty rose (one home), amber, pale
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
// Every pair was checked (OKLab ΔE ×100) under normal vision and simulated
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
// One home is a dusty rose (#E07A80), the owner's choice over the earlier orange-reds (#FE4400,
// then #FF5A28), which read too intense. A light red can't stay apart from the mid greens for
// red-green colour-blind readers (it matches homes_high under deuteranopia), so this palette no
// longer tries to serve them: ACCESSIBLE_COLORS below does, behind a switch in the key. The
// margins quoted above predate the rose and hold for normal vision only. The amber, both commercial blues, the brown and the purple are unchanged
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
  single_family: "#E07A80",
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

/**
 * Colour-blind palette, behind "Colour-blind friendly" in the key. The homes scale is viridis
 * (matplotlib's perceptually uniform map, built to read under every common colour-vision
 * deficiency), reversed so darker means more homes, sampled 0 to 0.8 so the lightest step stands
 * off the cream blocks. The other uses come from published colour-blind-safe sets: Paul Tol's
 * light cyan and grey, olive from his muted set, vermilion from his vibrant set, and black
 * (Okabe-Ito). Picked by scoring every pair, and every step against the blocks, under normal
 * vision and simulated protanopia, deuteranopia and tritanopia (Machado 2009) at both fill
 * strengths (0.85 and 0.7): weakest pair 5.4 (OKLab ΔE ×100) in every case. Under the same test
 * the standard palette falls to about 2 for deuteranopia.
 */
export const ACCESSIBLE_COLORS: Record<Category, string> = {
  single_family: "#7AD151",
  two_unit: "#35B779",
  three_unit: "#1F988B",
  homes_low: "#2A788E",
  homes_mid: "#39568C",
  homes_high: "#472F7D",
  homes_top: "#440154",
  commercial_light: "#99DDFF",
  commercial_dense: "#000000",
  industrial: "#999933",
  public: "#BBBBBB",
  unrestricted: "#CC3311",
  unknown: "#FFFFFF",
};

export type PaletteName = "standard" | "accessible";
export const PALETTES: Record<PaletteName, Record<Category, string>> = {
  standard: CATEGORY_COLORS,
  accessible: ACCESSIBLE_COLORS,
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

/**
 * A category's colour as the map shows it: drawn at `opacity` over the basemap's blocks. The key
 * uses this so its swatches match the lots rather than the full-strength colour.
 */
export function shownColor(c: Category, opacity: number, colors: Record<Category, string> = CATEGORY_COLORS): string {
  const rgb = (hex: string) => [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16));
  const top = rgb(colors[c]);
  const ground = rgb(MAP_COLORS.block);
  return `#${top.map((v, i) => Math.round(v * opacity + ground[i] * (1 - opacity)).toString(16).padStart(2, "0")).join("")}`;
}
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
