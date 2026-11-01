import type { Category } from "./palette";

/** out/<series>/districts/stats.json from the pipeline. */
export type DistrictStats = Record<Category, { parcels: number; area_sqft: number }> & {
  unknown_share?: number;
  mapped_area_sqft?: number;
  unmapped_parcels?: number;
};

const THREE_PLUS: Category[] = ["three_unit", "homes_low", "homes_mid", "homes_high", "homes_top"];
const RESIDENTIAL: Category[] = ["single_family", "two_unit", ...THREE_PLUS];
const ALL: Category[] = [
  ...RESIDENTIAL,
  "commercial_light",
  "commercial_dense",
  "industrial",
  "public",
  "unrestricted",
  "unknown",
];

// Whole maps miss about 2% of lots at most (scattered, and lots newer than the map); 1990-96 misses 8%.
const MIN_LOT_COVERAGE = 0.95;

/** True if the map leaves out part of the city, so its shares can't be compared with whole maps. */
export function isPartialMap(stats: DistrictStats): boolean {
  if (stats.unmapped_parcels === undefined) return false;
  const mapped = ALL.reduce((s, c) => s + (stats[c]?.parcels ?? 0), 0);
  return mapped / (mapped + stats.unmapped_parcels) < MIN_LOT_COVERAGE;
}

/**
 * Of the parcels on the homes scale (single-family through the densest steps, mixed-use included),
 * the share whose district allowed three or more homes per lot. null if none are residential.
 */
export function threePlusShare(stats: DistrictStats): number | null {
  const n = (cs: Category[]) => cs.reduce((s, c) => s + (stats[c]?.parcels ?? 0), 0);
  const total = n(RESIDENTIAL);
  return total > 0 ? n(THREE_PLUS) / total : null;
}
