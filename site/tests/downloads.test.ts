import { describe, expect, it } from "vitest";
import { formatBytes, lastUpdated, latestRedirects, rows, sheetsCsv, type Manifest } from "../lib/downloadRows";
import { ERAS } from "../lib/eras";
import { menuItems } from "../lib/menuItems";

const B = "https://blob.example/downloads";
const file = (series: string, layer: string, fmt: string, name: string, bytes: number, updated: string) => ({
  series, layer, fmt, url: `${B}/abc/${name}`, bytes, sha256: "x", fingerprint: "f", updated,
  extra: { [`${series}-${layer}.README.txt`]: `${B}/abc/${series}-${layer}.README.txt` },
});
const manifest: Manifest = {
  files: {
    "1948/lots.geojson": file("1948", "lots", "geojson", "1948-lots.geojson.zip", 3_182_670, "2026-11-01"),
    "1948/lots.gpkg": file("1948", "lots", "gpkg", "1948-lots.gpkg.zip", 5_000_000, "2026-11-01"),
    "1948/districts.shp": file("1948", "districts", "shp", "1948-districts.shp.zip", 1_000_000, "2026-11-02"),
    "1948/citywide.cog": file("1948", "citywide", "cog", "1948-citywide.tif", 891_233_486, "2026-11-03"),
    "1980-1990-zoning/citywide.cog": file("1980-1990-zoning", "citywide", "cog", "1980-1990-zoning-citywide.tif", 1_000, "2026-11-01"),
    "1980-1990-height/citywide.cog": file("1980-1990-height", "citywide", "cog", "1980-1990-height-citywide.tif", 1_000, "2026-11-01"),
  },
  sheets: {
    "1948": {
      updated: "2026-11-04", readme: "",
      items: [{ sheet: "p01", name: "1948-p01.tif", url: `${B}/sheets/p01.tif`, bytes: 100, md5: "m", updated: "2026-11-04" }],
    },
  },
  superseded: [],
};

describe("rows", () => {
  const r = rows(manifest);
  it("gives one row per era in ERAS order", () => {
    expect(r.map((x) => x.era.id)).toEqual(ERAS.map((e) => e.id));
    expect(r).toHaveLength(15);
  });
  it("fills cells from the manifest and leaves the rest empty", () => {
    const y = r[3];
    expect(y.lots.map((c) => c.label)).toEqual(["GeoJSON", "GeoPackage"]);
    expect(y.districts.map((c) => c.label)).toEqual(["Shapefile"]);
    expect(y.scans[0].citywide?.bytes).toBe(891_233_486);
    expect(y.scans[0].sheets).toEqual({ series: "1948", count: 1, bytes: 100, updated: "2026-11-04" });
    expect(r[0].lots).toEqual([]);
    expect(r[0].scans[0].citywide).toBeUndefined();
  });
  it("gives 1980-90 zoning, height and special-use scans, and today none", () => {
    const e = r.find((x) => x.era.id === "1980-1990")!;
    expect(e.scans.map((s) => [s.series, s.label])).toEqual([
      ["1980-1990-zoning", "Zoning"], ["1980-1990-height", "Height"], ["1980-1990-su", "Special use"],
    ]);
    expect(e.scans[1].citywide?.url).toContain("height-citywide");
    expect(r.at(-1)!.scans).toEqual([]);
  });
});

describe("helpers", () => {
  it("formats sizes", () => {
    expect(formatBytes(3_182_670)).toBe("3.2 MB");
    expect(formatBytes(700_000_000)).toBe("0.7 GB");
  });
  it("finds the latest update", () => expect(lastUpdated(manifest)).toBe("2026-11-04"));
  it("writes the sheet csv", () => {
    expect(sheetsCsv(manifest, "1948").split("\n")).toEqual([
      "sheet,name,url,bytes,md5,updated",
      `p01,1948-p01.tif,${B}/sheets/p01.tif,100,m,2026-11-04`,
    ]);
  });
  it("maps stable links to current urls", () => {
    const r = latestRedirects(manifest);
    expect(r).toContainEqual({
      source: "/data/latest/1948/1948-lots.geojson.zip", destination: `${B}/abc/1948-lots.geojson.zip`, permanent: false,
    });
    expect(r.map((x) => x.source)).toContain("/data/latest/1948/1948-lots.README.txt");
    expect(r.map((x) => x.source)).toContain("/data/latest/1948/1948-p01.tif");
  });
});

describe("menuItems", () => {
  it("lists Download the Data only when there are downloads, in a fixed order", () => {
    expect(menuItems(false).map((i) => i.href)).toEqual(["/", "/maps", "/about"]);
    expect(menuItems(true).map((i) => i.href)).toEqual(["/", "/maps", "/data", "/about"]);
  });
});
