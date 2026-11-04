import { describe, expect, it } from "vitest";
import { formatBytes, lastUpdated, latestRedirects, latestUrl, rows, sheetCredit, sheetLicense, sheetsCsv, type Manifest } from "../lib/downloadRows";
import { ERAS } from "../lib/eras";
import { menuItems } from "../lib/menuItems";
import { makeZip } from "client-zip";
import { zipEntries, zipLength } from "../lib/sheetZip";

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
  it("writes the sheet csv, each row carrying its license and credit (Rumsey for 1948)", () => {
    expect(sheetsCsv(manifest, "1948").split("\n")).toEqual([
      "sheet,name,url,bytes,md5,updated,license,credit",
      `p01,1948-p01.tif,${B}/sheets/p01.tif,100,m,2026-11-04,` +
        "CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/); 1948 imagery CC BY-NC-SA 3.0 (https://creativecommons.org/licenses/by-nc-sa/3.0/)," +
        `"Based on sfzoning.com by Steven Bacio for GrowSF (https://growsf.org); imagery: David Rumsey Map Collection, David Rumsey Map Center, Stanford Libraries"`,
    ]);
  });
  it("credits only sfzoning.com outside 1948", () => {
    expect(sheetLicense("1960")).toBe("CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/)");
    expect(sheetCredit("1960")).toBe("Based on sfzoning.com by Steven Bacio for GrowSF (https://growsf.org)");
  });
  it("quotes csv fields with commas, quotes and carriage returns", () => {
    const m: Manifest = { files: {}, superseded: [], sheets: { s: { updated: "d", readme: "", items: [
      { sheet: "a", name: 'x,"y"', url: "u", bytes: 1, md5: "m\r", updated: "d" },
    ] } } };
    expect(sheetsCsv(m, "s").split("\n")[1]).toMatch(/^a,"x,""y""",u,1,"m\r",d,/);
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

describe("latestUrl", () => {
  it("finds files, sidecars and sheets, and nothing else", () => {
    expect(latestUrl(manifest, "1948", "1948-lots.geojson.zip")).toBe(`${B}/abc/1948-lots.geojson.zip`);
    expect(latestUrl(manifest, "1948", "1948-citywide.README.txt")).toBe(`${B}/abc/1948-citywide.README.txt`);
    expect(latestUrl(manifest, "1948", "1948-p01.tif")).toBe(`${B}/sheets/p01.tif`);
    expect(latestUrl(manifest, "1948", "nope.zip")).toBeNull();
    expect(latestUrl(manifest, "1921", "1948-lots.geojson.zip")).toBeNull();
  });
});

describe("menuItems", () => {
  it("lists Download the Data only when there are downloads, in a fixed order", () => {
    expect(menuItems(false).map((i) => i.href)).toEqual(["/", "/maps", "/about"]);
    expect(menuItems(true).map((i) => i.href)).toEqual(["/", "/maps", "/data", "/about"]);
  });
});

describe("zipEntries", () => {
  const src = {
    license: "LICENSE TEXT",
    sheets: { "1948": { updated: "2026-11-04", readme: "README TEXT", items: [
      { sheet: "p01", name: "1948-p01.tif", url: `${B}/sheets/p01.tif`, bytes: 1, md5: "m", updated: "2026-11-04" },
      { sheet: "p02", name: "1948-p02.tif", url: `${B}/sheets/p02.tif`, bytes: 1, md5: "m", updated: "2026-11-04" },
    ] } },
  };
  it("yields the README and LICENSE, then each sheet fetched in order", async () => {
    const urls: string[] = [];
    const fetchFn = async (url: string | URL | Request) => { urls.push(String(url)); return new Response("x"); };
    const names: string[] = [];
    for await (const e of zipEntries(src, "1948", fetchFn)) {
      names.push(e.name);
      if (e.name === "README.txt") expect(e.input).toBe("README TEXT");
      if (e.name === "LICENSE.txt") expect(e.input).toBe("LICENSE TEXT");
    }
    expect(names).toEqual(["README.txt", "LICENSE.txt", "1948-p01.tif", "1948-p02.tif"]);
    expect(urls).toEqual([`${B}/sheets/p01.tif`, `${B}/sheets/p02.tif`]);
  });
  it("names the sheet that failed", async () => {
    const fetchFn = async (url: string | URL | Request) =>
      new Response("", { status: String(url).endsWith("p02.tif") ? 404 : 200 });
    const run = async () => { for await (const e of zipEntries(src, "1948", fetchFn)) void e; };
    await expect(run()).rejects.toThrow(/1948-p02\.tif.*404/);
  });
  it("fails the zip when a sheet's body is not the manifest's size", async () => {
    const body = () => new ReadableStream({ start(c) { c.enqueue(new Uint8Array(3)); c.close(); } });
    const fetchFn = async () => new Response(body());
    await expect(new Response(makeZip(zipEntries(src, "1948", fetchFn))).arrayBuffer()).rejects.toThrow(
      "Could not download sheet 1948-p01.tif: size at least 3, expected 1",
    );
  });
  it("fails the zip when a sheet's body is short", async () => {
    const fetchFn = async () => new Response(new ReadableStream({ start(c) { c.close(); } }));
    await expect(new Response(makeZip(zipEntries(src, "1948", fetchFn))).arrayBuffer()).rejects.toThrow(
      "Could not download sheet 1948-p01.tif: size 0, expected 1",
    );
  });
  it("judges a sheet by its decoded bytes, not a compressed Content-Length", async () => {
    const fetchFn = async () => new Response("x", { headers: { "content-length": "40" } });
    await expect(new Response(makeZip(zipEntries(src, "1948", fetchFn))).arrayBuffer()).resolves.toBeInstanceOf(ArrayBuffer);
  });
  it("predicts the zip's exact length", async () => {
    const s = { license: "Licence ©", sheets: { a: { updated: "2026-11-04", readme: "Zoning, café — “1948”\n", items: [
      { sheet: "1", name: "a-1.tif", url: "u1", bytes: 1234, md5: "m", updated: "2026-11-04" },
      { sheet: "2", name: "a-2é.tif", url: "u2", bytes: 7, md5: "m", updated: "2026-11-03" },
    ] } } };
    const fetchFn = async (url: string) => new Response(new Uint8Array(url === "u1" ? 1234 : 7));
    const zip = await new Response(makeZip(zipEntries(s, "a", fetchFn))).arrayBuffer();
    expect(zipLength(s, "a")).toBe(zip.byteLength);
  });
  it("needs the license text", async () => {
    const run = async () => { for await (const e of zipEntries({ sheets: src.sheets }, "1948", async () => new Response("x"))) void e; };
    await expect(run()).rejects.toThrow(/license/i);
  });
});
