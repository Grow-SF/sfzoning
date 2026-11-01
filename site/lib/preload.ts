import type { Protocol } from "pmtiles";
import { registerArchive, replaceArchive, type Pm } from "./archiveCache";

export type TileId = { z: number; x: number; y: number };

/** MapLibre's world is 512 px wide at zoom 0, whatever a source's tile size. */
const WORLD_PX = 512;

/**
 * How many destinations to trust as still in memory. Each is up to two scan viewports and its shared
 * tiles, measured at about 4 MB at worst on desktop, so six fit CACHE_BYTES with room for the map's own reads.
 */
const RECENT_DESTINATIONS = 6;

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
 * Fetch tiles of the archive at `url` ahead of need. It sits in `protocol` once (see registerArchive),
 * so MapLibre, asking for the same URL, reuses its header, directories and the tile bytes it keeps in
 * memory. `tiles` says which, given the archive's max zoom. Resolves true once every tile has been
 * read or is known not to be in the archive, false otherwise; never rejects, since the map fetches
 * for itself what it needs. An archive remembers a failed header or directory read for good, so a
 * failure replaces it with a fresh one at once: a map source looks its archive up by URL at each
 * read, and finds the fresh one too.
 */
export async function preload(protocol: Protocol, pm: Pm, url: string, tiles: (maxZoom: number) => TileId[]): Promise<boolean> {
  try {
    const archive = registerArchive(protocol, url, pm);
    const failed = () => {
      replaceArchive(protocol, url, archive, pm);
      return false;
    };
    try {
      const { maxZoom } = await archive.getHeader();
      const read = await Promise.all(tiles(maxZoom).map((t) => archive.getZxy(t.z, t.x, t.y).then(() => true, failed)));
      return read.every(Boolean);
    } catch {
      return failed();
    }
  } catch {
    return false;
  }
}

type Camera = { center: [number, number]; zoom: number };

/**
 * The destinations preloaded lately, and how far each archive got there. Only the most recent are
 * remembered, since the byte cache evicts the tiles of older ones: a destination forgotten here
 * is fetched again and counts as not ready, rather than being trusted after its bytes have gone.
 */
export class RecentDestinations {
  /** Destination, least recently used first; then archive, and whether its tiles are all in memory. */
  private seen = new Map<string, Map<string, boolean>>();
  constructor(readonly limit = RECENT_DESTINATIONS) {}

  private static key = (cam: Camera) => JSON.stringify([cam.center, cam.zoom]);

  private touch(cam: Camera) {
    const key = RecentDestinations.key(cam);
    const archives = this.seen.get(key) ?? new Map<string, boolean>();
    this.seen.delete(key);
    this.seen.set(key, archives);
    for (const oldest of this.seen.keys()) {
      if (this.seen.size <= this.limit) break;
      this.seen.delete(oldest);
    }
    return archives;
  }

  /** True if this archive should now be fetched for `cam`: it isn't loaded or loading there already. */
  start(archive: string, cam: Camera): boolean {
    const archives = this.touch(cam);
    if (archives.has(archive)) return false;
    archives.set(archive, false);
    return true;
  }

  /** A fetch has ended. One that failed is forgotten, so a later step tries again. */
  finish(archive: string, cam: Camera, ok: boolean) {
    const archives = this.seen.get(RecentDestinations.key(cam));
    if (!archives?.has(archive)) return;
    if (ok) archives.set(archive, true);
    else archives.delete(archive);
  }

  /** The map itself has drawn this archive at `cam`. */
  arrived(archive: string, cam: Camera) {
    this.touch(cam).set(archive, true);
  }

  ready(archive: string, cam: Camera): boolean {
    return this.seen.get(RecentDestinations.key(cam))?.get(archive) === true;
  }
}
