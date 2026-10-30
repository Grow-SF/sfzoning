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
 * City-zoom "grey veil": at z11–13 the scans' dense linework averages to a flat grey wash
 * (overview tiles), so scan ink eases back as the camera pulls out, full strength from z14.5
 * where the linework resolves. In "both" the colours carry the city view instead (see
 * StoryMap): the scan fades much further and the fills rise to their "allowed" strength.
 */
export const INK_ZOOM = { lo: 12, hi: 14.5 } as const;
export const INK_CITY = 0.45;

/** Scan opacity multiplier for a camera zoom (the number form of `inkByZoom`). */
export function scanInkWeight(zoom: number, min: number = INK_CITY): number {
  const { lo, hi } = INK_ZOOM;
  if (zoom <= lo) return min;
  if (zoom >= hi) return 1;
  return min + ((zoom - lo) / (hi - lo)) * (1 - min);
}

/** MapLibre zoom expression: `scale × scanInkWeight(zoom, min)`. Evaluated live as the camera moves. */
export function inkByZoom(scale: number, min: number = INK_CITY): unknown[] {
  return ["interpolate", ["linear"], ["zoom"], INK_ZOOM.lo, scale * min, INK_ZOOM.hi, scale];
}
