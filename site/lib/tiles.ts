import published from "./published.json";

export const BASEMAP = "ref/blocks.pmtiles";
export const DEFAULT_TILES_BASE = "http://127.0.0.1:8787";

export function tilesBase(): string {
  return (process.env.NEXT_PUBLIC_TILES_BASE_URL || DEFAULT_TILES_BASE).replace(/\/+$/, "");
}

/**
 * Where a file is in the tile store. Published files are never overwritten: a changed one goes up
 * under a new name (scripts/publish-tiles.mjs), and this site version reads the names it shipped
 * with. A pipeline's local output keeps the usual names.
 */
export function publishedPath(path: string, map: Readonly<Record<string, string>> = published): string {
  const local = /^https?:\/\/(127\.0\.0\.1|localhost)[:/]/.test(`${tilesBase()}/`);
  return local ? path : (map[path] ?? path);
}

export function tileUrl(path: string): string {
  return `pmtiles://${tilesBase()}/${publishedPath(path)}`;
}

/** The series' scans as they look on paper (pipeline `sfzmaps paper`): opaque, colours untouched. */
export function scanPath(series: string): string {
  return `${series}/${series}-paper.pmtiles`;
}

export function scanSource(series: string): string {
  return tileUrl(scanPath(series));
}

/** Path of a series' classified-district vector tiles (layers `parcels`, `districts`). */
export function districtPath(series: string): string {
  return `${series}/districts/${series}-districts.pmtiles`;
}

/** Classified-district vector tiles (produced by step two; may not exist yet). */
export function districtSource(series: string): string {
  return tileUrl(districtPath(series));
}

/** Plain HTTP URL of a tiles-server file (for probing and JSON). */
export function tilesHttpUrl(path: string): string {
  return `${tilesBase()}/${publishedPath(path)}`;
}

/** Per-category parcel counts and areas for a series' districts. */
export function districtStatsUrl(series: string): string {
  return tilesHttpUrl(`${series}/districts/stats.json`);
}
