import { afterEach, describe, expect, it } from "vitest";
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
  it("keeps them while loaded but not yet faded in", () => {
    expect(heldScans(hold, ["1950"], always, false, 2000)).toEqual(hold);
  });
  it("releases them once loaded and faded", () => {
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
