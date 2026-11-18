import { describe, expect, it } from "vitest";
import { versionedName } from "../lib/downloadNames.mjs";
import type { Manifest, Version } from "../lib/downloadRows";
import { layerLabel, seriesLayers, versionPairs, versionRows } from "../lib/versions";

/** A release file: the versioned name the publish script would make, and a full SHA-256 whose first 8 digits are the hash. */
const ver = (series: string, layer: string, fmt: string, date: string, hash8: string, bytes = 1000): Version => {
  const ext = ({ geojson: "geojson.zip", gpkg: "gpkg.zip", shp: "shp.zip", cog: "tif" } as Record<string, string>)[fmt];
  return { series, layer, fmt, date, bytes, sha256: hash8.repeat(8), name: versionedName({ series, layer, date, sha256: hash8.repeat(8), ext }) };
};

const GPKG_NOW = ver("1948", "lots", "gpkg", "2026-12-02", "bbbbbbbb");
const SHEET_NOW = ver("1948", "sheet-01", "cog", "2026-12-02", "cccccccc");
const sheet = (number: string, name: string) => ({ sheet: `S${number}`, number, name, bytes: 5, md5: "m", sha256: "s", updated: "2026-12-02" });

const manifest: Manifest = {
  files: {
    "1948/lots.gpkg": { series: "1948", layer: "lots", fmt: "gpkg", name: GPKG_NOW.name, bytes: 38, sha256: GPKG_NOW.sha256, fingerprint: "f", updated: "2026-12-02" },
  },
  sheets: {
    "1948": { updated: "2026-12-02", readme: "", items: [sheet("01", SHEET_NOW.name)] },
  },
};

const history = (...versions: Version[]) => ({ versions });

describe("versionRows", () => {
  it("gives two rows on different dates, newest first, with latest only on the newer", () => {
    const old = ver("1948", "lots", "gpkg", "2026-11-20", "aaaaaaaa");
    const rows = versionRows(history(old, GPKG_NOW), manifest, "1948", "lots");
    expect(rows.map((r) => [r.date, r.latest])).toEqual([
      ["2026-12-02", true],
      ["2026-11-20", false],
    ]);
    expect(rows[1].files[0].name).toBe(old.name);
  });

  it("gives two rows for two releases on the same day, ordered by history, latest on the later one", () => {
    const first = ver("1948", "lots", "gpkg", "2026-12-02", "aaaaaaaa");
    const rows = versionRows(history(first, GPKG_NOW), manifest, "1948", "lots");
    expect(rows).toHaveLength(2);
    expect(rows[0].files[0].name).toBe(GPKG_NOW.name);
    expect(rows[0].latest).toBe(true);
    expect(rows[1].files[0].name).toBe(first.name);
    expect(rows[1].latest).toBe(false);
  });

  it("groups the formats released together into one row, with labels, sizes and checksum links", () => {
    const geo = ver("1948", "lots", "geojson", "2026-12-02", "dddddddd", 41);
    const shp = ver("1948", "lots", "shp", "2026-12-02", "eeeeeeee", 35);
    const rows = versionRows(history(geo, GPKG_NOW, shp), manifest, "1948", "lots");
    expect(rows).toHaveLength(1);
    expect(rows[0].files.map((f) => f.label)).toEqual(["GeoJSON", "GeoPackage", "Shapefile"]);
    const f = rows[0].files[0];
    expect(f.bytes).toBe(41);
    expect(f.url).toBe(`/data/files/1948/${geo.name}`);
    expect(f.checksumUrl).toBe(`/data/files/1948/${geo.name}.sha256`);
    expect(f.sha256).toHaveLength(64);
    expect(f.sha256).toBe("dddddddd".repeat(8));
    expect(f.checksumUrl.endsWith(`-dddddddd.geojson.zip.sha256`)).toBe(true);
  });

  it("keeps the formats of one release in one row when they differ", () => {
    const geo = ver("1948", "lots", "geojson", "2026-12-02", "dddddddd", 41);
    const rows = versionRows(history(geo, GPKG_NOW), manifest, "1948", "lots");
    expect(rows).toHaveLength(1);
    expect(rows[0].files.map((f) => f.label)).toEqual(["GeoJSON", "GeoPackage"]);
    expect(rows[0].latest).toBe(true);
  });

  it("labels a sheet's GeoTIFF and marks it latest against the manifest's sheet name", () => {
    const old = ver("1948", "sheet-01", "cog", "2026-11-20", "aaaaaaaa");
    const rows = versionRows(history(old, SHEET_NOW), manifest, "1948", "sheet-01");
    expect(rows.map((r) => r.latest)).toEqual([true, false]);
    expect(rows[0].files[0].label).toBe("GeoTIFF");
    expect(rows[0].files[0].url).toBe(`/data/files/1948/${SHEET_NOW.name}`);
  });

  it("returns no rows, and no layers, with zero versions", () => {
    expect(versionRows(history(), manifest, "1948", "lots")).toEqual([]);
    expect(seriesLayers(history(), "1948")).toEqual([]);
    expect(versionPairs(history())).toEqual([]);
  });
});

describe("seriesLayers and versionPairs", () => {
  it("lists lots, districts, citywide, then sheets in natural order", () => {
    const h = history(
      ver("1948", "sheet-100", "cog", "2026-12-02", "aaaaaaaa"),
      ver("1948", "citywide", "cog", "2026-12-02", "bbbbbbbb"),
      ver("1948", "sheet-99", "cog", "2026-12-02", "cccccccc"),
      ver("1948", "districts", "gpkg", "2026-12-02", "dddddddd"),
      ver("1948", "lots", "gpkg", "2026-12-02", "eeeeeeee"),
      ver("1921", "lots", "gpkg", "2026-12-02", "ffffffff"),
    );
    expect(seriesLayers(h, "1948")).toEqual(["lots", "districts", "citywide", "sheet-99", "sheet-100"]);
    expect(seriesLayers(h, "1921")).toEqual(["lots"]);
  });

  it("gives each series and layer pair once", () => {
    const h = history(
      ver("1948", "lots", "gpkg", "2026-11-20", "aaaaaaaa"),
      ver("1948", "lots", "geojson", "2026-11-20", "bbbbbbbb"),
      ver("1921", "citywide", "cog", "2026-11-20", "cccccccc"),
    );
    expect(versionPairs(h)).toEqual([
      { series: "1948", layer: "lots" },
      { series: "1921", layer: "citywide" },
    ]);
  });

  it("labels layers for people", () => {
    expect(layerLabel("lots")).toBe("Lots");
    expect(layerLabel("districts")).toBe("Districts");
    expect(layerLabel("citywide")).toBe("Citywide scan");
    expect(layerLabel("sheet-01")).toBe("Sheet 01");
  });
});
