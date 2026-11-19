import { ERAS } from "./eras";
import type { Fmt, Kind } from "./releaseFiles";

/** What a visitor narrowed the release listing to; a full set means "not narrowed". */
export type Filter = { format: Set<Fmt>; layer: Set<Kind>; edition: Set<string> };

const FORMATS: Fmt[] = ["geojson", "gpkg", "shp", "cog"];
const LAYERS: Kind[] = ["lots", "districts", "scans", "sheets"];
const EDITIONS: string[] = ERAS.map((e) => e.id);

export const ALL: Filter = { format: new Set(FORMATS), layer: new Set(LAYERS), edition: new Set(EDITIONS) };

const GROUPS = [
  ["format", FORMATS],
  ["layer", LAYERS],
  ["edition", EDITIONS],
] as const;

/** A missing param means all; a comma list narrows. An empty, unknown or repeated param is an error naming it. */
export function parseFilter(params: URLSearchParams): { filter: Filter } | { error: string } {
  const filter: Filter = { format: new Set(ALL.format), layer: new Set(ALL.layer), edition: new Set(ALL.edition) };
  for (const [key, allowed] of GROUPS) {
    const given = params.getAll(key);
    if (given.length === 0) continue;
    if (given.length > 1) return { error: `${key} is given more than once` };
    const values = given[0].split(",");
    for (const v of values) {
      if (!v) return { error: `${key} has an empty value` };
      if (!(allowed as readonly string[]).includes(v)) return { error: `${key} has an unknown value: ${v}` };
    }
    (filter[key] as Set<string>) = new Set(values);
  }
  return { filter };
}

/** `""` when nothing is narrowed; else `?` and the narrowed groups (format, layer, edition) with values in canonical order. */
export function filterQuery(f: Filter): string {
  const parts: string[] = [];
  for (const [key, allowed] of GROUPS) {
    const set = f[key] as Set<string>;
    if (set.size === allowed.length) continue;
    parts.push(`${key}=${(allowed as readonly string[]).filter((v) => set.has(v)).join(",")}`);
  }
  return parts.length ? `?${parts.join("&")}` : "";
}

export function matches(file: { fmt: Fmt; kind: Kind; era: string }, f: Filter): boolean {
  return f.format.has(file.fmt) && f.layer.has(file.kind) && f.edition.has(file.era);
}

/** The format zip a filter selects: one vector format, lots and districts both in, every edition. Otherwise null. */
export function formatZip(f: Filter): "geojson" | "gpkg" | "shp" | null {
  if (f.format.size !== 1 || f.edition.size !== EDITIONS.length) return null;
  if (!f.layer.has("lots") || !f.layer.has("districts")) return null;
  const [fmt] = [...f.format];
  return fmt === "cog" ? null : fmt;
}

/**
 * The filter a page's chips select. A release shows chips only for its own editions; the editions
 * it lacks count as selected, so "every chip on" is the full filter whatever ERAS holds.
 */
export function chipFilter(checked: { format: string[]; layer: string[]; edition: string[] }, present: string[]): Filter {
  return {
    format: new Set(checked.format as Fmt[]),
    layer: new Set(checked.layer as Kind[]),
    edition: new Set([...checked.edition, ...EDITIONS.filter((e) => !present.includes(e))]),
  };
}
