import type { Protocol } from "pmtiles";
import { registerArchive } from "./archiveCache";

export type TileId = { z: number; x: number; y: number };

/** MapLibre's world is 512 px wide at zoom 0, whatever a source's tile size. */
const WORLD_PX = 512;

/**
 * The tiles of a `tileSize` source that fill a `width` x `height` map at `camera`: what MapLibre
 * would ask for there (bearing and padding ignored). Zoom above the archive's `maxZoom` is
 * clamped to it, which covers the same ground with fewer, larger tiles.
 */
export function tilesCovering(
  camera: { center: [number, number]; zoom: number },
  width: number,
  height: number,
  tileSize: 256 | 512,
  maxZoom: number,
): TileId[] {
  // MapLibre rounds the zoom for raster sources and floors it for vector ones.
  const wanted = tileSize === 256 ? Math.round(camera.zoom + 1) : Math.floor(camera.zoom);
  const z = Math.max(0, Math.min(maxZoom, wanted));
  const n = 2 ** z;
  const [lon, lat] = camera.center;
  const sin = Math.sin((lat * Math.PI) / 180);
  const cx = (lon + 180) / 360;
  const cy = 0.5 - Math.log((1 + sin) / (1 - sin)) / (4 * Math.PI);
  const world = WORLD_PX * 2 ** camera.zoom;
  const clamp = (v: number) => Math.min(n - 1, Math.max(0, v));
  const x0 = clamp(Math.floor((cx - width / 2 / world) * n));
  const x1 = clamp(Math.floor((cx + width / 2 / world) * n));
  const y0 = clamp(Math.floor((cy - height / 2 / world) * n));
  const y1 = clamp(Math.floor((cy + height / 2 / world) * n));
  const tiles: TileId[] = [];
  for (let x = x0; x <= x1; x++) for (let y = y0; y <= y1; y++) tiles.push({ z, x, y });
  return tiles;
}

/**
 * Fetch tiles of the archives at `urls` ahead of need. Each archive is in `protocol` once (see
 * registerArchive), so MapLibre, asking for the same URL, reuses its header, directories and the
 * tile bytes it keeps in memory. `tiles` says which, given the archive's max zoom. Never rejects: a
 * tile that can't be had now is simply fetched when the map needs it.
 */
export async function preload(protocol: Protocol, urls: string[], tiles: (maxZoom: number) => TileId[]): Promise<void> {
  try {
    const { PMTiles, FetchSource } = await import("pmtiles");
    await Promise.all(
      urls.map(async (url) => {
        try {
          const archive = registerArchive(protocol, url, { PMTiles, FetchSource });
          const { maxZoom } = await archive.getHeader();
          await Promise.all(tiles(maxZoom).map((t) => archive.getZxy(t.z, t.x, t.y).catch(() => {})));
        } catch {
          // Nothing to do: the map fetches what it needs when it needs it.
        }
      }),
    );
  } catch {
    // As above.
  }
}
