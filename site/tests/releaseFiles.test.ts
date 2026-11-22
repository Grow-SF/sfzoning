import { describe, expect, it } from "vitest";
import { versionedName } from "../lib/downloadNames.mjs";
import type { Version } from "../lib/downloadRows";
import { eraOf, extraFiles, releaseChanges, releaseFiles, releaseManifest, urlsTxt } from "../lib/releaseFiles";
import { ALL, filterQuery, formatZip, matches, parseFilter, type Filter } from "../lib/releaseFilter";

const EXT: Record<string, string> = { geojson: "geojson.zip", gpkg: "gpkg.zip", shp: "shp.zip", cog: "tif" };
const ver = (series: string, layer: string, fmt: string, date: string, h: string, bytes = 1000): Version => ({
  series, layer, fmt, date, bytes, sha256: h.repeat(8),
  name: versionedName({ series, layer, date, sha256: h.repeat(8), ext: EXT[fmt] }),
});

const D = "2026-12-02";
const v = [
  ver("1921", "lots", "geojson", D, "a1a1a1a1"),
  ver("1921", "lots", "gpkg", D, "a2a2a2a2"),
  ver("1921", "lots", "shp", D, "a3a3a3a3"),
  ver("1921", "sheet-02", "cog", D, "b2b2b2b2"),
  ver("1921", "sheet-01", "cog", D, "b1b1b1b1"),
  ver("1921", "citywide", "cog", D, "b0b0b0b0"),
  ver("1980-1990-zoning", "lots", "gpkg", D, "c1c1c1c1"),
  ver("1980-1990-height", "citywide", "cog", D, "c2c2c2c2"),
  ver("1980-1990-zoning", "citywide", "cog", D, "c3c3c3c3"),
];
const history = { versions: v };
const zip = { name: "z", bytes: 1, sha256: "x" };
const zips = { all: zip, geojson: zip, gpkg: zip, shp: zip };
const release = (names: string[], extras: { series: string; name: string; sha256: string }[] = []) => ({ id: D, date: D, names, extras, zips });
const all = release(v.map((x) => x.name), [{ series: "1921", name: "sfzoning-1921-citywide-README.txt", sha256: "e" }]);

const ok = (q: string): Filter => (parseFilter(new URLSearchParams(q)) as { filter: Filter }).filter;

describe("eraOf", () => {
  it("finds the era of district and scan series", () => {
    expect(eraOf("1980-1990-height")).toBe("1980-1990");
    expect(eraOf("1980-1990-su")).toBe("1980-1990");
    expect(eraOf("1980")).toBe("1980");
    expect(eraOf("today")).toBe("today");
    expect(eraOf("nope")).toBeUndefined();
  });
});

describe("releaseFiles", () => {
  it("lists in era order: lots by format, then citywide, then sheets by number", () => {
    const files = releaseFiles(all, history);
    expect(files.map((f) => [f.era, f.series, f.layer, f.fmt, f.kind])).toEqual([
      ["1921", "1921", "lots", "geojson", "lots"],
      ["1921", "1921", "lots", "gpkg", "lots"],
      ["1921", "1921", "lots", "shp", "lots"],
      ["1921", "1921", "citywide", "cog", "scans"],
      ["1921", "1921", "sheet-01", "cog", "sheets"],
      ["1921", "1921", "sheet-02", "cog", "sheets"],
      ["1980-1990", "1980-1990-zoning", "lots", "gpkg", "lots"],
      ["1980-1990", "1980-1990-zoning", "citywide", "cog", "scans"],
      ["1980-1990", "1980-1990-height", "citywide", "cog", "scans"],
    ]);
  });

  it("throws naming a name missing from history", () => {
    expect(() => releaseFiles(release(["sfzoning-1921-lots-2026-01-01-deadbeef.gpkg.zip"]), history)).toThrow(/deadbeef/);
  });
});

describe("extraFiles", () => {
  it("adds the era of each sidecar", () => {
    expect(extraFiles(all)[0].era).toBe("1921");
  });
});

describe("releaseChanges", () => {
  it("reports a new version as changed and a dropped series as removed", () => {
    const prev = releaseFiles(all, history);
    const newer = ver("1921", "lots", "gpkg", "2026-12-09", "f2f2f2f2");
    const names = all.names.filter((n) => !n.includes("1980-1990-height")).map((n) => (n === v[1].name ? newer.name : n));
    const cur = releaseFiles(release(names), { versions: [...v, newer] });
    const c = releaseChanges(prev, cur);
    expect(c.added).toEqual([]);
    expect(c.changed.map((x) => [x.from.name, x.to.name])).toEqual([[v[1].name, newer.name]]);
    expect(c.removed.map((f) => f.series)).toEqual(["1980-1990-height"]);
  });

  it("calls everything added when there is no previous release", () => {
    const cur = releaseFiles(all, history);
    expect(releaseChanges(null, cur).added).toHaveLength(cur.length);
  });
});

describe("urlsTxt and manifest", () => {
  it("lists absolute urls, files first then extras", () => {
    const files = releaseFiles(all, history);
    const lines = urlsTxt(files, extraFiles(all)).trimEnd().split("\n");
    expect(lines[0]).toBe(`https://sfzoning.com/data/files/1921/${files[0].name}`);
    expect(lines).toHaveLength(files.length + 1);
    expect(lines.at(-1)).toBe("https://sfzoning.com/data/files/1921/sfzoning-1921-citywide-README.txt");
  });

  it("gives each file its url and latest url in the manifest", () => {
    const files = releaseFiles(all, history);
    const m = releaseManifest(all, files, extraFiles(all), "L") as any;
    expect(m.files[0]).toMatchObject({
      series: "1921", layer: "lots", fmt: "geojson",
      url: `https://sfzoning.com/data/files/1921/${files[0].name}`,
      latest: "https://sfzoning.com/data/files/1921/sfzoning-1921-lots-latest.geojson.zip",
    });
    expect(m).toMatchObject({ id: D, date: D, license: "L", zips });
    expect(m.extras).toHaveLength(1);
  });
});

describe("parseFilter", () => {
  it("reads a narrowed group and leaves the rest open", () => {
    const f = ok("format=shp");
    expect([...f.format]).toEqual(["shp"]);
    expect(f.layer).toEqual(ALL.layer);
    expect(f.edition).toEqual(ALL.edition);
  });

  it("rejects empty, unknown, or repeated params, naming them", () => {
    for (const [q, p] of [["format=", "format"], ["format=SHP", "format"], ["edition=1999", "edition"], ["format=shp&format=gpkg", "format"], ["layer=x", "layer"]]) {
      expect((parseFilter(new URLSearchParams(q)) as { error: string }).error).toContain(p);
    }
  });

  it("round-trips through filterQuery in canonical order", () => {
    expect(filterQuery(ok("format=shp,gpkg"))).toBe("?format=gpkg,shp");
    expect(filterQuery(ok("edition=1921&format=shp"))).toBe("?format=shp&edition=1921");
    expect(filterQuery(ALL)).toBe("");
  });
});

describe("matches and formatZip", () => {
  it("matches on all three groups", () => {
    const f = ok("format=shp&edition=1921");
    expect(matches({ fmt: "shp", kind: "lots", era: "1921" }, f)).toBe(true);
    expect(matches({ fmt: "gpkg", kind: "lots", era: "1921" }, f)).toBe(false);
    expect(matches({ fmt: "shp", kind: "lots", era: "1935" }, f)).toBe(false);
  });

  it("gives a format zip only for one vector format, lots and districts, every edition", () => {
    expect(formatZip(ok("format=shp"))).toBe("shp");
    expect(formatZip(ok("format=shp&edition=1921"))).toBeNull();
    expect(formatZip(ok("format=cog"))).toBeNull();
    expect(formatZip(ok("format=shp&layer=lots"))).toBeNull();
    expect(formatZip(ok("format=shp&layer=lots,districts"))).toBe("shp");
    expect(formatZip(ALL)).toBeNull();
  });
});
