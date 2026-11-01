import { afterEach, describe, expect, it, vi } from "vitest";
import { Protocol } from "pmtiles";
import { preload, RecentDestinations, tilesCovering } from "@/lib/preload";
import { ByteCache, cachedSource, type Pm } from "@/lib/archiveCache";
import { ERAS } from "@/lib/eras";
import { BASEMAP, scanSource, tileUrl } from "@/lib/tiles";
import { COLOURS, HOLD_LIMIT_MS, heldScans, nextHeld } from "@/lib/scanHold";
import { INK_CITY, inkByZoom, scanInkWeight } from "@/lib/theme";

const orig = process.env.NEXT_PUBLIC_TILES_BASE_URL;
afterEach(() => {
  if (orig === undefined) delete process.env.NEXT_PUBLIC_TILES_BASE_URL;
  else process.env.NEXT_PUBLIC_TILES_BASE_URL = orig;
});

describe("tileUrl", () => {
  it("defaults to the local tile server", () => {
    delete process.env.NEXT_PUBLIC_TILES_BASE_URL;
    expect(tileUrl("1948/1948.pmtiles")).toBe("pmtiles://http://127.0.0.1:8787/1948/1948.pmtiles");
  });
  it("builds pmtiles URLs from env, trimming trailing slash", () => {
    process.env.NEXT_PUBLIC_TILES_BASE_URL = "https://tiles.example.org/";
    expect(tileUrl("ref/blocks.pmtiles")).toBe("pmtiles://https://tiles.example.org/ref/blocks.pmtiles");
    expect(BASEMAP).toBe("ref/blocks.pmtiles");
  });
  it("derives scan sources from the series id", () => {
    delete process.env.NEXT_PUBLIC_TILES_BASE_URL;
    expect(scanSource("1948")).toMatch(/1948\/1948-paper\.pmtiles$/);
  });
});

describe("scanInkWeight", () => {
  it("eases scan ink at city zooms and is full strength up close", () => {
    expect(scanInkWeight(11)).toBeCloseTo(INK_CITY);
    expect(scanInkWeight(13.25)).toBeGreaterThan(INK_CITY);
    expect(scanInkWeight(13.25)).toBeLessThan(1);
    expect(scanInkWeight(15)).toBe(1);
  });
  it("has a MapLibre zoom expression with the same end stops", () => {
    expect(inkByZoom(0.5)).toEqual(["interpolate", ["linear"], ["zoom"], 12, 0.5 * INK_CITY, 14.5, 0.5]);
  });
});

describe("ERAS", () => {
  it("has unique ids", () => {
    const ids = ERAS.map((e) => e.id);
    expect(new Set(ids).size).toBe(ids.length);
  });
  it("is sorted by year", () => {
    const years = ERAS.map((e) => e.year);
    expect(years).toEqual([...years].sort((a, b) => a - b));
  });
  it("covers the 15 story eras", () => {
    expect(ERAS).toHaveLength(15);
  });
});

describe("heldScans", () => {
  const hold = { held: ["1948"], since: 1000 };
  const never = () => false;
  const always = () => true;
  it("keeps the held scans while the wanted one is not loaded", () => {
    expect(heldScans(hold, ["1950"], never, true, 2000)).toEqual(hold);
  });
  it("keeps them while loaded but the camera is still moving", () => {
    expect(heldScans(hold, ["1950"], always, false, 2000)).toEqual(hold);
  });
  it("releases them once loaded and the camera has stopped", () => {
    expect(heldScans(hold, ["1950"], always, true, 2000)).toBeNull();
  });
  it("releases them at the limit even if the scan never arrived", () => {
    expect(heldScans(hold, ["1950"], never, false, 1000 + HOLD_LIMIT_MS - 1)).toEqual(hold);
    expect(heldScans(hold, ["1950"], never, false, 1000 + HOLD_LIMIT_MS)).toBeNull();
  });
  it("drops a held scan that is wanted again", () => {
    const two = { held: ["1948", "1950"], since: 1000 };
    expect(heldScans(two, ["1950"], never, false, 2000)).toEqual({ held: ["1948"], since: 1000 });
    expect(heldScans({ held: ["1950"], since: 1000 }, ["1950"], never, false, 2000)).toBeNull();
  });
  it("has nothing to release without a hold", () => {
    expect(heldScans(null, ["1950"], never, false, 2000)).toBeNull();
  });
});

describe("tilesCovering", () => {
  const camera = { center: [-122.4194, 37.7749] as [number, number], zoom: 14.2 };
  const worldTile = (lon: number, lat: number, z: number) => {
    const n = 2 ** z;
    const rad = (lat * Math.PI) / 180;
    return { x: Math.floor(((lon + 180) / 360) * n), y: Math.floor(((1 - Math.log(Math.tan(rad) + 1 / Math.cos(rad)) / Math.PI) / 2) * n) };
  };
  const has = (tiles: { z: number; x: number; y: number }[], z: number, t: { x: number; y: number }) =>
    tiles.some((k) => k.z === z && k.x === t.x && k.y === t.y);

  it("covers the centre tile and all four corners of the viewport", () => {
    const tiles = tilesCovering(camera, 1440, 900, 512, 18);
    expect(new Set(tiles.map((t) => t.z))).toEqual(new Set([14]));
    expect(has(tiles, 14, worldTile(camera.center[0], camera.center[1], 14))).toBe(true);
    // Half the viewport in degrees at z14.2 (512px world), a little inside the edge.
    const dLon = (720 / (512 * 2 ** camera.zoom)) * 360 * 0.99;
    const dLat = (450 / (512 * 2 ** camera.zoom)) * 360 * Math.cos((camera.center[1] * Math.PI) / 180) * 0.99;
    for (const sx of [-1, 1])
      for (const sy of [-1, 1])
        expect(has(tiles, 14, worldTile(camera.center[0] + sx * dLon, camera.center[1] + sy * dLat, 14))).toBe(true);
  });
  it("uses one zoom higher for 256px tiles", () => {
    expect(new Set(tilesCovering(camera, 1440, 900, 256, 18).map((t) => t.z))).toEqual(new Set([15]));
  });
  it("clamps to the archive's max zoom, covering the same ground", () => {
    const tiles = tilesCovering({ ...camera, zoom: 17.5 }, 1440, 900, 512, 14);
    expect(new Set(tiles.map((t) => t.z))).toEqual(new Set([14]));
    expect(has(tiles, 14, worldTile(camera.center[0], camera.center[1], 14))).toBe(true);
    // Clamped to z14 the tiles are larger than the z17.5 viewport; its four corners are still inside them.
    const dLon = (720 / (512 * 2 ** 17.5)) * 360 * 0.99;
    const dLat = (450 / (512 * 2 ** 17.5)) * 360 * Math.cos((camera.center[1] * Math.PI) / 180) * 0.99;
    for (const sx of [-1, 1])
      for (const sy of [-1, 1])
        expect(has(tiles, 14, worldTile(camera.center[0] + sx * dLon, camera.center[1] + sy * dLat, 14))).toBe(true);
  });
  it("rounds the zoom for 256px tiles as MapLibre does, and floors it for 512px", () => {
    expect(new Set(tilesCovering({ ...camera, zoom: 14.5 }, 1440, 900, 256, 18).map((t) => t.z))).toEqual(new Set([16]));
    expect(new Set(tilesCovering({ ...camera, zoom: 14.4 }, 1440, 900, 256, 18).map((t) => t.z))).toEqual(new Set([15]));
    expect(new Set(tilesCovering({ ...camera, zoom: 14.8 }, 1440, 900, 512, 18).map((t) => t.z))).toEqual(new Set([14]));
  });
});

describe("cachedSource", () => {
  const bytes = (n: number) => new Uint8Array(n).buffer;
  const fake = (read: (offset: number, length: number) => Promise<{ data: ArrayBuffer }>) => {
    const calls: string[] = [];
    const source = {
      getKey: () => "a",
      getBytes: (offset: number, length: number) => (calls.push(`${offset}+${length}`), read(offset, length)),
    };
    return { source, calls };
  };

  it("makes one upstream request for two reads of a range", async () => {
    const { source, calls } = fake(async (_o, l) => ({ data: bytes(l) }));
    const s = cachedSource(source, new ByteCache(1000));
    await s.getBytes(0, 10);
    await s.getBytes(0, 10);
    expect(calls).toEqual(["0+10"]);
  });
  it("shares one request between concurrent reads", async () => {
    const { source, calls } = fake(async (_o, l) => ({ data: bytes(l) }));
    const s = cachedSource(source, new ByteCache(1000));
    const [a, b] = await Promise.all([s.getBytes(5, 10), s.getBytes(5, 10)]);
    expect(calls).toEqual(["5+10"]);
    expect(a.data).not.toBe(b.data); // each caller owns its copy
  });
  it("evicts the least recently used range once past the bound", async () => {
    const { source, calls } = fake(async (_o, l) => ({ data: bytes(l) }));
    const s = cachedSource(source, new ByteCache(25));
    await s.getBytes(0, 10);
    await s.getBytes(10, 10);
    await s.getBytes(0, 10); // 0 is now the more recent
    await s.getBytes(20, 10); // 30 bytes: evicts 10+10
    calls.length = 0;
    await s.getBytes(0, 10);
    await s.getBytes(20, 10);
    expect(calls).toEqual([]);
    await s.getBytes(10, 10);
    expect(calls).toEqual(["10+10"]);
  });
  it("does not keep a failed read", async () => {
    let fail = true;
    const { source, calls } = fake(async (_o, l) => {
      if (fail) throw new Error("offline");
      return { data: bytes(l) };
    });
    const s = cachedSource(source, new ByteCache(1000));
    await expect(s.getBytes(0, 10)).rejects.toThrow("offline");
    fail = false;
    await expect(s.getBytes(0, 10)).resolves.toBeDefined();
    expect(calls).toEqual(["0+10", "0+10"]);
  });
});

describe("preload", () => {
  const url = "https://tiles.example.org/x.pmtiles";
  const tile = () => [{ z: 0, x: 0, y: 0 }];
  /**
   * Stand-ins for the pmtiles classes. Like the real archive, each one keeps its header read,
   * rejected or not, for good. `reads` answers each upstream read in turn; `tiles` each tile read.
   */
  const fakes = (reads: boolean[], tiles: (() => Promise<unknown>)[] = []) => {
    class FetchSource {
      constructor(readonly url: string) {}
      getKey = () => this.url;
      getBytes = async () => {
        if (reads.shift() === false) throw new Error("offline");
        return { data: new ArrayBuffer(1) };
      };
    }
    class PMTiles {
      private header?: Promise<{ maxZoom: number }>;
      constructor(readonly source: FetchSource) {}
      getHeader = () => (this.header ??= this.source.getBytes().then(() => ({ maxZoom: 14 })));
      getZxy = () => (tiles.shift() ?? (() => Promise.resolve({ data: new ArrayBuffer(1) })))();
    }
    return { PMTiles, FetchSource } as unknown as Pm;
  };

  it("reports success, with a tile missing from the archive counting as success", async () => {
    const pm = fakes([true], [() => Promise.resolve(undefined)]);
    expect(await preload(new Protocol(), pm, url, tile)).toBe(true);
  });
  it("reports failure, without rejecting, when a tile can't be read", async () => {
    const pm = fakes([true], [() => Promise.reject(new Error("offline"))]);
    expect(await preload(new Protocol(), pm, url, tile)).toBe(false);
  });
  it("leaves the archive usable after a failed read: the next attempt succeeds", async () => {
    const protocol = new Protocol();
    const pm = fakes([false, true]);
    const once = "https://tiles.example.org/fails-once.pmtiles";
    expect(await preload(protocol, pm, once, tile)).toBe(false);
    expect(await preload(protocol, pm, once, tile)).toBe(true);
  });
  it("a map source reading by URL after a failure gets a working archive, not none", async () => {
    const protocol = new Protocol();
    const pm = fakes([false, true]);
    const live = "https://tiles.example.org/live.pmtiles";
    await preload(protocol, pm, live, tile);
    await expect(protocol.get(live)!.getHeader()).resolves.toMatchObject({ maxZoom: 14 });
  });
});

describe("RecentDestinations", () => {
  const at = (n: number) => ({ center: [n, 0] as [number, number], zoom: 12 });

  it("counts a destination as ready only once its fetch has succeeded", () => {
    const d = new RecentDestinations();
    expect(d.start("1948", at(1))).toBe(true);
    expect(d.ready("1948", at(1))).toBe(false);
    expect(d.start("1948", at(1))).toBe(false); // already on its way
    d.finish("1948", at(1), true);
    expect(d.ready("1948", at(1))).toBe(true);
  });
  it("forgets a failed fetch: not ready, and a later step tries again", () => {
    const d = new RecentDestinations();
    d.start("1948", at(1));
    d.finish("1948", at(1), false);
    expect(d.ready("1948", at(1))).toBe(false);
    expect(d.start("1948", at(1))).toBe(true);
  });
  it("forgets the oldest destination past the limit: not ready, and fetched again", () => {
    const d = new RecentDestinations(2);
    for (const n of [1, 2, 3]) {
      d.start("1948", at(n));
      d.finish("1948", at(n), true);
    }
    expect(d.ready("1948", at(1))).toBe(false);
    expect(d.ready("1948", at(3))).toBe(true);
    expect(d.start("1948", at(1))).toBe(true);
  });
  it("keeps a destination a step asked for again", () => {
    const d = new RecentDestinations(2);
    d.arrived("1948", at(1));
    d.arrived("1948", at(2));
    expect(d.start("1948", at(1))).toBe(false);
    d.arrived("1948", at(3));
    expect(d.ready("1948", at(1))).toBe(true);
    expect(d.ready("1948", at(2))).toBe(false);
  });
  it("a fetch finishing after its destination was forgotten doesn't bring it back", () => {
    const d = new RecentDestinations(1);
    d.start("1948", at(1));
    d.arrived("1948", at(2));
    d.finish("1948", at(1), true);
    expect(d.ready("1948", at(1))).toBe(false);
  });
});

describe("nextHeld", () => {
  it("holds the scans a step leaves", () => {
    expect(nextHeld([], ["1948"], ["1960"], false, false)).toEqual(["1948"]);
  });
  it("past two scans, drops the oldest and keeps the one on screen now", () => {
    expect(nextHeld(["1921", "1948"], ["1960"], ["1972"], false, false)).toEqual(["1948", "1960"]);
  });
  it("a scan held before and on screen again since counts as the most recent", () => {
    expect(nextHeld(["1948", "1921"], ["1948"], ["1972"], false, false)).toEqual(["1921", "1948"]);
  });
  it("doesn't hold a scan the step wants", () => {
    expect(nextHeld(["1948"], ["1960"], ["1948"], false, false)).toEqual(["1960"]);
  });
  it("keeps colours on a step with a scan alone, and they don't use up the scans' places", () => {
    expect(nextHeld(["1921", "1948"], ["1960"], ["1972"], true, true)).toEqual(["1948", "1960", COLOURS]);
    expect(nextHeld([COLOURS], ["1960"], ["1972"], true, false)).toEqual(["1960", COLOURS]);
  });
  it("releases held colours on a step that draws colours of its own", () => {
    expect(nextHeld([COLOURS], ["1960"], ["1972"], false, true)).toEqual(["1960"]);
  });
});
