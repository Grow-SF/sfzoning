import { afterEach, describe, expect, it } from "vitest";
import { ERAS } from "@/lib/eras";
import { BASEMAP, districtSource, scanSource, tileUrl } from "@/lib/tiles";

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
  it("derives scan and district sources from the series id", () => {
    delete process.env.NEXT_PUBLIC_TILES_BASE_URL;
    expect(scanSource("1948")).toMatch(/1948\/1948\.pmtiles$/);
    expect(districtSource("1980-1990-zoning")).toMatch(/1980-1990-zoning\/districts\.pmtiles$/);
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
