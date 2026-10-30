import type { Category } from "./palette";

/** out/<series>/districts/stats.json from the pipeline. */
export type DistrictStats = Record<Category, { parcels: number; area_sqft: number }> & {
  unknown_share?: number;
  mapped_area_sqft?: number;
};

const RESIDENTIAL: Category[] = ["single_family", "two_unit", "three_unit", "multi_family", "mixed"];
const THREE_PLUS: Category[] = ["three_unit", "multi_family", "mixed"];

/**
 * Of the parcels zoned for homes (single-family through mixed residential-commercial),
 * the share whose district allowed three or more homes per lot. null if none are residential.
 */
export function threePlusShare(stats: DistrictStats): number | null {
  const n = (cs: Category[]) => cs.reduce((s, c) => s + (stats[c]?.parcels ?? 0), 0);
  const total = n(RESIDENTIAL);
  return total > 0 ? n(THREE_PLUS) / total : null;
}
