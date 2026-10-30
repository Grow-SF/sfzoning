// "What it allowed" palette. One palette for every era, so change between eras pops.
//
// Homes allowed per lot read as one warm ramp, light to dark (one house is yellow,
// apartments deep brick), so density is legible from lightness alone, without hue.
// The other uses step out of that ramp: violet, orchid pink, deep teal, green, navy. Public
// is green, not blue, so parks never read as water on the basemap; industrial is teal, not
// grey, so it never reads as the scans' own ink.
//
// tests/palette.test.ts checks every pair (OKLab ΔE ×100) under normal vision and simulated
// protanopia and deuteranopia (Machado 2009), in the "allowed" view (fills over the paper
// basemap) and in the "both" view, over paper and over the scans' solid black ink. Current
// minimums: allowed ≥ 15.7 normal / 9.9 colour-blind; both ≥ 13.5 normal / 8.4 colour-blind.
// The colours came from a constrained search over OKLCH boxes (one hue family per use), with
// chroma capped so the map stays a printed map, not a neon one.
// "Unknown" is a pale neutral near the paper: it means "not read yet", not a zoning class.
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
  single_family: "#E4CD37",
  two_unit: "#DC7E26",
  three_unit: "#B8342C",
  multi_family: "#5A1805",
  mixed: "#886CC9",
  commercial: "#F68FD5",
  industrial: "#00687B",
  public: "#5AAA82",
  unrestricted: "#001961",
  unknown: "#E8E6E1",
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
