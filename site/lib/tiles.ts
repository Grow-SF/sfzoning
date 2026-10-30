export const BASEMAP = "ref/blocks.pmtiles";
export const DEFAULT_TILES_BASE = "http://127.0.0.1:8787";

export function tilesBase(): string {
  return (process.env.NEXT_PUBLIC_TILES_BASE_URL || DEFAULT_TILES_BASE).replace(/\/+$/, "");
}

export function tileUrl(path: string): string {
  return `pmtiles://${tilesBase()}/${path}`;
}

export function scanSource(series: string): string {
  return tileUrl(`${series}/${series}.pmtiles`);
}

/** Classified-district vector tiles (produced by step two; may not exist yet). */
export function districtSource(series: string): string {
  return tileUrl(`${series}/districts.pmtiles`);
}
