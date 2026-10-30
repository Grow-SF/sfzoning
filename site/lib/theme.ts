// Map colours mirror the CSS tokens in app/globals.css (MapLibre cannot read CSS variables).
export const MAP_COLORS = {
  water: "#d3dde0",
  street: "#f7f4ee",
  block: "#dfd9cd",
  blockEdge: "#d2cbbd",
  accent: "#2337c6",
  /** District boundaries over the colour fills: ink, not black, so the scan's own linework stays distinct. */
  districtEdge: "#2b2722",
} as const;

/**
 * Scan opacity multiplier for a camera zoom. At city-wide zooms the scans' overview tiles
 * merge dense black linework into a heavy grey veil; easing the ink back keeps the overview
 * legible on the paper ground. Full strength from z14.5, where the linework resolves.
 */
export function scanInkWeight(zoom: number): number {
  const lo = 12, hi = 14.5, min = 0.55;
  if (zoom <= lo) return min;
  if (zoom >= hi) return 1;
  return min + ((zoom - lo) / (hi - lo)) * (1 - min);
}
