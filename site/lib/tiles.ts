export const BASEMAP = "ref/blocks.pmtiles";
export const DEFAULT_TILES_BASE = "http://127.0.0.1:8787";

export function tilesBase(): string {
  return (process.env.NEXT_PUBLIC_TILES_BASE_URL || DEFAULT_TILES_BASE).replace(/\/+$/, "");
}

export function tileUrl(path: string): string {
  return `pmtiles://${tilesBase()}/${path}`;
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
  return `${tilesBase()}/${path}`;
}

/** Per-category parcel counts and areas for a series' districts. */
export function districtStatsUrl(series: string): string {
  return tilesHttpUrl(`${series}/districts/stats.json`);
}
