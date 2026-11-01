import { afterEach, describe, expect, it, vi } from "vitest";
import { Protocol } from "pmtiles";
import { preload, tilesCovering } from "@/lib/preload";
import { ByteCache, cachedSource } from "@/lib/archiveCache";
import { ERAS } from "@/lib/eras";
import { BASEMAP, scanSource, tileUrl } from "@/lib/tiles";
import { HOLD_LIMIT_MS, heldScans } from "@/lib/scanHold";
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
  it("resolves when a fetch fails", async () => {
    vi.stubGlobal("fetch", () => Promise.reject(new Error("offline")));
    try {
      await expect(preload(new Protocol(), ["https://tiles.example.org/x.pmtiles"], () => [{ z: 0, x: 0, y: 0 }])).resolves.toBeUndefined();
    } finally {
      vi.unstubAllGlobals();
    }
  });
});
