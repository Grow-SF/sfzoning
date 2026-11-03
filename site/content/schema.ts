import type { Feature, FeatureCollection, Geometry } from "geojson";
import type { Era } from "@/lib/eras";

export const VIEWS = ["scan", "allowed", "both"] as const;
export type View = (typeof VIEWS)[number];

export type Camera = { center: [number, number]; zoom: number; bearing?: number };

export type GeoJSON = Feature<Geometry> | FeatureCollection<Geometry>;

/**
 * `cvd`: the same paragraph for the colour-blind palette. Every paragraph that names a map colour
 * has one, naming that palette's colours instead (the story test checks this).
 */
export type Block = { kind: "p" | "quote" | "stat"; text: string; cvd?: string; href?: string };

/** Colour words of the standard palette; a paragraph using one needs a `cvd` version. */
export const STANDARD_COLOUR_WORDS = /\b(rose|amber|yellow|greens?|blue|brown|grey)\b/i;

export type Step = {
  id: string;
  era: string;
  view: View;
  camera: Camera;
  title: string;
  body: Block[];
  /** Ids into content/sources.ts (which mirrors docs/research/sources.md) */
  sources: string[];
  highlight?: GeoJSON;
};

// Generous box around San Francisco; a camera outside it is a typo.
const BOUNDS = { minLon: -122.7, maxLon: -122.2, minLat: 37.6, maxLat: 37.9 };

/** Returns human-readable problems with the story; empty means valid. */
export function validateStory(steps: Step[], eras: Pick<Era, "id">[]): string[] {
  const errors: string[] = [];
  const eraIds = new Set(eras.map((e) => e.id));
  const seen = new Set<string>();
  steps.forEach((s, i) => {
    const at = `step ${i} (${s.id || "no id"})`;
    if (!s.id) errors.push(`${at}: missing id`);
    else if (seen.has(s.id)) errors.push(`${at}: duplicate id "${s.id}"`);
    seen.add(s.id);
    if (!eraIds.has(s.era)) errors.push(`${at}: unknown era "${s.era}"`);
    if (!(VIEWS as readonly string[]).includes(s.view)) errors.push(`${at}: unknown view "${s.view}"`);
    if (!s.title?.trim()) errors.push(`${at}: empty title`);
    if (!s.body?.length) errors.push(`${at}: empty body`);
    if (!s.sources?.length) errors.push(`${at}: no sources`);
    s.body?.forEach((b, j) => {
      const word = s.view !== "scan" && !b.cvd && b.text.match(STANDARD_COLOUR_WORDS)?.[0];
      if (word) errors.push(`${at}: paragraph ${j} names "${word}" but has no colour-blind (cvd) version`);
    });
    const [lon, lat] = s.camera?.center ?? [NaN, NaN];
    const z = s.camera?.zoom;
    if (!(lon >= BOUNDS.minLon && lon <= BOUNDS.maxLon && lat >= BOUNDS.minLat && lat <= BOUNDS.maxLat))
      errors.push(`${at}: camera center [${lon}, ${lat}] is outside San Francisco`);
    if (!(z >= 10 && z <= 18)) errors.push(`${at}: camera zoom ${z} out of range 10–18`);
  });
  return errors;
}
