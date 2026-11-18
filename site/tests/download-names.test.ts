import { describe, expect, it } from "vitest";
import { EXTS, assignSheetNumbers, blobPath, latestName, parseName, versionedName } from "../lib/downloadNames.mjs";

const HASH = "3f9a2c1b" + "0".repeat(56);

describe("names", () => {
  it("builds versioned and latest names", () => {
    expect(versionedName({ series: "1948", layer: "lots", date: "2026-11-04", sha256: HASH, ext: "gpkg.zip" }))
      .toBe("sfzoning-1948-lots-2026-11-04-3f9a2c1b.gpkg.zip");
    expect(latestName({ series: "1980-1990-zoning", layer: "sheet-p22", ext: "tif" }))
      .toBe("sfzoning-1980-1990-zoning-sheet-p22-latest.tif");
    expect(latestName({ series: "1948", layer: "sheets", ext: "zip" })).toBe("sfzoning-1948-sheets-latest.zip");
  });

  it("exposes the extension table and blob paths", () => {
    expect(EXTS).toEqual({ geojson: "geojson.zip", gpkg: "gpkg.zip", shp: "shp.zip", cog: "tif" });
    expect(blobPath("1948", "sfzoning-1948-lots-latest.gpkg.zip")).toBe("downloads/1948/sfzoning-1948-lots-latest.gpkg.zip");
  });

  it("parses versioned names, sidecars and checksums", () => {
    expect(parseName("1921", "sfzoning-1921-citywide-2026-11-04-ce47cd18.README.txt"))
      .toMatchObject({ layer: "citywide", ext: "README.txt", date: "2026-11-04", hash8: "ce47cd18", latest: false, checksum: false });
    expect(parseName("1921", "sfzoning-1921-lots-2026-11-04-ce47cd18.gpkg.zip.sha256")?.checksum).toBe(true);
  });

  it("parses latest names and the sheet zip", () => {
    expect(parseName("1980-1990-zoning", "sfzoning-1980-1990-zoning-sheet-p22-latest.tif"))
      .toMatchObject({ layer: "sheet-p22", ext: "tif", latest: true, date: null, hash8: null });
    expect(parseName("1948", "sfzoning-1948-sheets-latest.zip"))
      .toMatchObject({ layer: "sheets", ext: "zip", latest: true });
  });

  it("rejects anything that is not a name for that series", () => {
    const bad = [
      "../x",
      "sfzoning-1948-lots-latest.gpkg.zip/../../x",
      "sfzoning-1948-lots/2026-11-04-ce47cd18.gpkg.zip",
      "sfzoning-1948-lots-latest.gpkg.zip.sha256",
      "sfzoning-1921-lots-latest.gpkg.zip".replace("1921", "1948"),
      "sfzoning-1921-lots-2026-13-04-ce47cd18.gpkg.zip",
      "sfzoning-1921-lots-2026-02-30-ce47cd18.gpkg.zip",
      "sfzoning-1921-lots-2026-11-04-CE47CD18.gpkg.zip",
      "sfzoning-1921-lots-latest.exe",
      "sfzoning-1921-sheets-2026-11-04-ce47cd18.zip",
      "sfzoning-1921-sheets-latest.gpkg.zip",
      "sfzoning-1921-sheet-x-latest.tif",
    ];
    for (const name of bad) expect(parseName("1921", name), name).toBeNull();
  });
});

describe("assignSheetNumbers", () => {
  it("numbers new ids in natural order when nothing is known", () => {
    expect(assignSheetNumbers(["10", "2", "5a", "1"], {})).toEqual({ "1": "01", "2": "02", "5a": "03", "10": "04" });
  });

  it("uses the -pNN suffix when an id has one", () => {
    expect(assignSheetNumbers(["x-p05", "x-p22"], {})).toEqual({ "x-p05": "p05", "x-p22": "p22" });
  });

  it("throws when two new ids take the same pNN", () => {
    expect(() => assignSheetNumbers(["a-p05", "b-p05"], {})).toThrow(/a-p05.*b-p05.*p05/);
  });

  it("throws when a new id takes a pNN a known entry already has", () => {
    expect(() => assignSheetNumbers(["b-p05"], { "a-p05": "p05" })).toThrow(/a-p05.*b-p05.*p05/);
  });

  it("keeps known numbers and never reuses a removed one", () => {
    expect(assignSheetNumbers(["1", "5a", "10", "11"], { "1": "01", "2": "02", "5a": "03", "10": "04" }))
      .toEqual({ "1": "01", "2": "02", "5a": "03", "10": "04", "11": "05" });
  });
});
