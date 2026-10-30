// "What it allowed" palette. One palette for every era, so change between eras pops.
//
// Homes allowed per lot read as one warm ramp, light to dark (one house is pale yellow,
// apartments deep brick), so density is legible from lightness alone, without hue.
// The other uses step out of that ramp: purple, pink, warm grey, park sage, navy. Public is
// sage, not blue, so parks never read as water on the basemap.
//
// Checked with the dataviz skill's validate_palette.js (--pairs all, light mode):
// every pair is >= 15 OKLab dE for normal vision; under simulated deutan/protan/tritan
// vision every pair is >= 7.6 except industrial vs three-unit (7.7, protan), which the
// legend labels and the parcel popup disambiguate. "Unknown" is deliberately close to
// the paper ground: it means "not read yet", not a zoning class.
export const CATEGORIES = [
  "single_family",
  "two_unit",
  "three_unit",
  "multi_family",
  "mixed",
  "commercial",
  "industrial",
  "public",
  "unrestricted",
  "unknown",
] as const;

export type Category = (typeof CATEGORIES)[number];

export const CATEGORY_COLORS: Record<Category, string> = {
  single_family: "#F4D03A",
  two_unit: "#EE8F2E",
  three_unit: "#C63A1C",
  multi_family: "#6E1F12",
  mixed: "#9D6BD0",
  commercial: "#EC5E8C",
  industrial: "#5E5A55",
  public: "#9CC9A8",
  unrestricted: "#1F3F8F",
  unknown: "#C4C2BC",
};

export const CATEGORY_LABELS: Record<Category, string> = {
  single_family: "One home",
  two_unit: "Two homes",
  three_unit: "Three homes",
  multi_family: "Apartments",
  mixed: "Homes and shops",
  commercial: "Commercial",
  industrial: "Industrial",
  public: "Public",
  unrestricted: "Unrestricted",
  unknown: "Not read yet",
};

/** Residential categories, ordered by how many homes a lot may hold. */
export const HOMES_RAMP: Category[] = ["single_family", "two_unit", "three_unit", "multi_family"];
export const OTHER_USES: Category[] = ["mixed", "commercial", "industrial", "public", "unrestricted", "unknown"];

export function isCategory(v: unknown): v is Category {
  return typeof v === "string" && (CATEGORIES as readonly string[]).includes(v);
}

/** MapLibre `match` expression colouring a feature by its `category` property. */
export function fillColorExpression(): unknown[] {
  const pairs = CATEGORIES.flatMap((c) => [c, CATEGORY_COLORS[c]]);
  return ["match", ["get", "category"], ...pairs, CATEGORY_COLORS.unknown];
}

/** Fill opacity for the district layer in each view. "both" lets the scan's ink show through. */
export function districtFillOpacity(view: "scan" | "allowed" | "both"): number {
  if (view === "allowed") return 0.85;
  if (view === "both") return 0.6;
  return 0;
}
