import { describe, expect, it } from "vitest";
import { downloadsJson, formatBytes, lastUpdated, rows, sheetCredit, sheetLicense, sheetsCsv, type Manifest } from "../lib/downloadRows";
import { ERAS } from "../lib/eras";
import { menuItems } from "../lib/menuItems";
import { makeZip } from "client-zip";
import { filesResponse } from "../lib/filesRoute";
import { sheetsZipResponse, zipEntries, zipLength } from "../lib/sheetZip";

const B = "https://blob.example";
/** A versioned name; `layer` and `ext` as downloadNames makes them. */
const V = (series: string, layer: string, ext: string, date = "2026-11-01") => `sfzoning-${series}-${layer}-${date}-3f9a2c1b.${ext}`;
const sh = (sheet: string, name: string, bytes: number, updated = "2026-11-04") => ({ sheet, number: "01", name, bytes, md5: "m", sha256: "x", updated });
const file = (series: string, layer: string, fmt: string, name: string, bytes: number, updated: string) => ({
  series, layer, fmt, name, bytes, sha256: "x", fingerprint: "f", updated,
  extra: { "README.txt": name.replace(/\.[^-]*$/, ".README.txt"), "LICENSE.txt": name.replace(/\.[^-]*$/, ".LICENSE.txt") },
});
const LOTS = V("1948", "lots", "geojson.zip");
const manifest: Manifest = {
  base: B,
  files: {
    "1948/lots.geojson": file("1948", "lots", "geojson", LOTS, 3_182_670, "2026-11-01"),
    "1948/lots.gpkg": file("1948", "lots", "gpkg", V("1948", "lots", "gpkg.zip"), 5_000_000, "2026-11-01"),
    "1948/districts.shp": file("1948", "districts", "shp", V("1948", "districts", "shp.zip"), 1_000_000, "2026-11-02"),
    "1948/citywide.cog": file("1948", "citywide", "cog", V("1948", "citywide", "tif"), 891_233_486, "2026-11-03"),
    "1980-1990-zoning/citywide.cog": file("1980-1990-zoning", "citywide", "cog", V("1980-1990-zoning", "citywide", "tif"), 1_000, "2026-11-01"),
    "1980-1990-height/citywide.cog": file("1980-1990-height", "citywide", "cog", V("1980-1990-height", "citywide", "tif"), 1_000, "2026-11-01"),
  },
  sheets: {
    "1948": {
      updated: "2026-11-04", readme: "",
      items: [sh("p01", V("1948", "sheet-01", "tif", "2026-11-04"), 100)],
    },
  },
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
  });
  it("links each cell to its latest, dated and versions URLs", () => {
    const y = r[3];
    expect(y.lots[0]).toMatchObject({
      latestUrl: "/data/files/1948/sfzoning-1948-lots-latest.geojson.zip",
      versionedUrl: `/data/files/1948/${LOTS}`,
      versionsUrl: "/data/versions/1948/lots",
    });
    expect(y.lots[1].latestUrl).toBe("/data/files/1948/sfzoning-1948-lots-latest.gpkg.zip");
    expect(y.districts[0]).toMatchObject({
      latestUrl: "/data/files/1948/sfzoning-1948-districts-latest.shp.zip",
      versionsUrl: "/data/versions/1948/districts",
    });
    expect(y.scans[0].citywide).toMatchObject({
      latestUrl: "/data/files/1948/sfzoning-1948-citywide-latest.tif",
      versionedUrl: `/data/files/1948/${V("1948", "citywide", "tif")}`,
      versionsUrl: "/data/versions/1948/citywide",
    });
    expect(r[0].scans[0].citywide).toBeUndefined();
  });
  it("gives 1980-90 zoning, height and special-use scans, and today none", () => {
    const e = r.find((x) => x.era.id === "1980-1990")!;
    expect(e.scans.map((s) => [s.series, s.label])).toEqual([
      ["1980-1990-zoning", "Zoning"], ["1980-1990-height", "Height"], ["1980-1990-su", "Special use"],
    ]);
    expect(e.scans[1].citywide?.versionedUrl).toContain("height-citywide");
    expect(e.scans[1].citywide?.latestUrl).toBe("/data/files/1980-1990-height/sfzoning-1980-1990-height-citywide-latest.tif");
    expect(e.scans[1].citywide?.versionsUrl).toBe("/data/versions/1980-1990-height/citywide");
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
      "sheet,number,name,url,latest_url,bytes,md5,sha256,updated,license,credit",
      `p01,01,${V("1948", "sheet-01", "tif", "2026-11-04")},https://sfzoning.com/data/files/1948/${V("1948", "sheet-01", "tif", "2026-11-04")},` +
        "https://sfzoning.com/data/files/1948/sfzoning-1948-sheet-01-latest.tif,100,m,x,2026-11-04," +
        "CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/); 1948 imagery CC BY-NC-SA 3.0 (https://creativecommons.org/licenses/by-nc-sa/3.0/)," +
        `"Based on sfzoning.com by Steven Bacio for GrowSF (https://growsf.org); imagery: David Rumsey Map Collection, David Rumsey Map Center, Stanford Libraries"`,
    ]);
  });
  it("credits only sfzoning.com outside 1948", () => {
    expect(sheetLicense("1960")).toBe("CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/)");
    expect(sheetCredit("1960")).toBe("Based on sfzoning.com by Steven Bacio for GrowSF (https://growsf.org)");
  });
  it("quotes csv fields with commas, quotes and carriage returns", () => {
    const m: Manifest = { files: {}, sheets: { s: { updated: "d", readme: "", items: [
      { ...sh("a", 'x,"y"', 1, "d"), md5: "m\r" },
    ] } } };
    expect(sheetsCsv(m, "s").split("\n")[1]).toMatch(
      /^a,01,"x,""y""","https:\/\/sfzoning\.com\/data\/files\/s\/x,""y""",https:\/\/sfzoning\.com\/data\/files\/s\/sfzoning-s-sheet-01-latest\.tif,1,"m\r",x,d,/,
    );
  });
  it("lists url and latest as absolute sfzoning.com links in downloads.json", () => {
    const j = downloadsJson(manifest);
    const at = "https://sfzoning.com/data/files/1948/";
    expect(j.files["1948/lots.geojson"]).toMatchObject({ url: at + LOTS, latest: at + "sfzoning-1948-lots-latest.geojson.zip" });
    expect(j.files["1948/citywide.cog"].latest).toBe(at + "sfzoning-1948-citywide-latest.tif");
    expect(j.sheets["1948"].latest).toBe(at + "sfzoning-1948-sheets-latest.zip");
    expect(j.sheets["1948"].items[0]).toMatchObject({
      url: at + V("1948", "sheet-01", "tif", "2026-11-04"), latest: at + "sfzoning-1948-sheet-01-latest.tif",
    });
  });
});

describe("filesResponse", () => {
  const never = async () => { throw new Error("fetched"); };
  const missing = async (series: string, name: string, m: Manifest = manifest) => {
    const res = await filesResponse(m, series, name, never);
    expect(res.status).toBe(404);
  };
  it("redirects a versioned name of a known series to Blob, permanently, and its .sha256 likewise", async () => {
    for (const n of [LOTS, `${LOTS}.sha256`]) {
      const res = await filesResponse(manifest, "1948", n, never);
      expect(res.status).toBe(308);
      expect(res.headers.get("location")).toBe(`${B}/downloads/1948/${n}`);
    }
  });
  it("redirects an older version that is no longer in the manifest", async () => {
    const old = V("1948", "lots", "geojson.zip", "2026-10-01");
    const res = await filesResponse(manifest, "1948", old, never);
    expect(res.status).toBe(308);
    expect(res.headers.get("location")).toBe(`${B}/downloads/1948/${old}`);
  });
  it("sends a latest name to the current versioned name, never cached", async () => {
    const res = await filesResponse(manifest, "1948", "sfzoning-1948-lots-latest.geojson.zip", never);
    expect(res.status).toBe(307);
    expect(res.headers.get("location")).toBe(`/data/files/1948/${LOTS}`);
    expect(res.headers.get("cache-control")).toBe("no-store");
    const sheet = await filesResponse(manifest, "1948", "sfzoning-1948-sheet-01-latest.tif", never);
    expect(sheet.headers.get("location")).toBe(`/data/files/1948/${V("1948", "sheet-01", "tif", "2026-11-04")}`);
  });
  it("404s a latest name nothing current matches, and an ambiguous sidecar", async () => {
    await missing("1948", "sfzoning-1948-sheet-02-latest.tif");
    await missing("1948", "sfzoning-1948-lots-latest.README.txt");
    await missing("1948", "sfzoning-1948-lots-latest.shp.zip");
  });
  it("streams the sheets zip under its latest name", async () => {
    const m: Manifest = {
      ...manifest, license: "LICENSE TEXT",
      sheets: { "1948": { updated: "2026-11-04", readme: "GETTING THE NEWEST VERSION\nuse the latest link", items: [sh("p01", V("1948", "sheet-01", "tif", "2026-11-04"), 3)] } },
    };
    const res = await filesResponse(m, "1948", "sfzoning-1948-sheets-latest.zip", async () => new Response(new Uint8Array(3)));
    expect(res.status).toBe(200);
    expect(res.headers.get("content-disposition")).toBe('attachment; filename="sfzoning-1948-sheets-latest.zip"');
    const body = new Uint8Array(await res.arrayBuffer());
    expect(res.headers.get("content-length")).toBe(String(body.length));
    expect(new TextDecoder().decode(body)).toContain("GETTING THE NEWEST VERSION");
  });
  it("404s, fetching nothing, an unknown series, malformed names and path tricks", async () => {
    await missing("1900", LOTS.replace("1948", "1900"));
    await missing("1948", "1948-lots.geojson.zip");
    await missing("1948", "..");
    await missing("1948", `../1921/${LOTS}`);
    await missing("1948", `1921/${LOTS}`);
    await missing("1948", `${LOTS}%2F..`);
    await missing("1948", LOTS.replace("3f9a2c1b", "3F9A2C1B"));
    await missing("1948", LOTS.replace("2026-11-01", "2026-02-30"));
    await missing("..", LOTS);
    await missing("1948/../1921", LOTS);
  });
  it("404s inherited object keys as a series, without fetching", async () => {
    const src = { ...manifest, license: "L" };
    for (const s of ["constructor", "__proto__", "toString"]) {
      await missing(s, `sfzoning-${s}-lots-2026-11-01-3f9a2c1b.gpkg.zip`);
      await missing(s, `sfzoning-${s}-sheets-latest.zip`);
      expect(sheetsZipResponse(src, s, never).status).toBe(404);
    }
  });
  it("404s the latest name of a sheet no longer in the manifest, and a sheets zip for a series without sheets", async () => {
    const noSheets: Manifest = { ...manifest, sheets: {} };
    await missing("1948", "sfzoning-1948-sheet-01-latest.tif", noSheets);
    await missing("1948", "sfzoning-1948-sheets-latest.zip", noSheets);
  });
  describe("a local manifest", () => {
    const local: Manifest = {
      files: { "1921/citywide.cog": { ...file("1921", "citywide", "cog", V("1921", "citywide", "tif"), 5, "2026-11-01"), local: "1921/1921-citywide.tif", extraLocal: { "README.txt": "1921/1921-citywide.README.txt" } } },
      sheets: { "1921": { updated: "2026-11-04", readme: "", items: [{ ...sh("p01", V("1921", "sheet-01", "tif", "2026-11-04"), 1), local: "work/1921/p01/warped_paper.tif" }] } },
      local: true, base: "http://127.0.0.1:8787",
    };
    it("redirects current names, latest or versioned, to the local paths", async () => {
      const at = async (n: string) => (await filesResponse(local, "1921", n, never)).headers.get("location");
      expect(await at(V("1921", "citywide", "tif"))).toBe("http://127.0.0.1:8787/downloads/1921/1921-citywide.tif");
      expect(await at("sfzoning-1921-citywide-latest.tif")).toBe("http://127.0.0.1:8787/downloads/1921/1921-citywide.tif");
      expect(await at(V("1921", "citywide", "README.txt"))).toBe("http://127.0.0.1:8787/downloads/1921/1921-citywide.README.txt");
      expect(await at(V("1921", "sheet-01", "tif", "2026-11-04"))).toBe("http://127.0.0.1:8787/work/1921/p01/warped_paper.tif");
    });
    it("404s anything else", async () => {
      await missing("1921", V("1921", "citywide", "tif", "2026-10-01"), local);
      await missing("1921", `${V("1921", "citywide", "tif")}.sha256`, local);
      await missing("1921", V("1921", "citywide", "LICENSE.txt"), local);
    });
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
    license: "LICENSE TEXT", base: B,
    sheets: { "1948": { updated: "2026-11-04", readme: "README TEXT", items: [
      sh("p01", "1948-p01.tif", 1),
      sh("p02", "1948-p02.tif", 1),
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
    expect(urls).toEqual([`${B}/downloads/1948/1948-p01.tif`, `${B}/downloads/1948/1948-p02.tif`]);
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
    const s = { license: "Licence ©", base: B, sheets: { a: { updated: "2026-11-04", readme: "Zoning, café — “1948”\n", items: [
      sh("1", "a-1.tif", 1234),
      sh("2", "a-2é.tif", 7, "2026-11-03"),
    ] } } };
    const fetchFn = async (url: string) => new Response(new Uint8Array(url.endsWith("a-1.tif") ? 1234 : 7));
    const zip = await new Response(makeZip(zipEntries(s, "a", fetchFn))).arrayBuffer();
    expect(zipLength(s, "a")).toBe(zip.byteLength);
  });
  it("needs the license text", async () => {
    const run = async () => { for await (const e of zipEntries({ sheets: src.sheets }, "1948", async () => new Response("x"))) void e; };
    await expect(run()).rejects.toThrow(/license/i);
  });
});

describe("server downloads", () => {
  const src = {
    license: "LICENSE TEXT", base: B,
    sheets: { "1948": { updated: "2026-11-04", readme: "README TEXT", items: [
      sh("p01", "1948-p01.tif", 3),
      sh("p02", "1948-p02é.tif", 2),
    ] } },
  };
  const fetchFn = async (url: string) => new Response(new Uint8Array(url.endsWith("p01.tif") ? 3 : 2));
  it("answers the sheets zip as an attachment of its predicted length", async () => {
    const res = sheetsZipResponse(src, "1948", fetchFn);
    expect(res.status).toBe(200);
    expect(res.headers.get("content-type")).toBe("application/zip");
    expect(res.headers.get("content-disposition")).toBe('attachment; filename="sfzoning-1948-sheets-latest.zip"');
    const body = await res.arrayBuffer();
    expect(res.headers.get("content-length")).toBe(String(body.byteLength));
    expect(body.byteLength).toBe(zipLength(src, "1948"));
  });
  it("404s an unknown series without fetching", async () => {
    const never = async () => { throw new Error("fetched"); };
    expect(sheetsZipResponse(src, "1900", never).status).toBe(404);
  });
});
