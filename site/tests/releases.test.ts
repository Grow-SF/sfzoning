import { describe, expect, it } from "vitest";
import { versionedName } from "../lib/downloadNames.mjs";
import type { Manifest, Version } from "../lib/downloadRows";
import { RELEASE_ID, currentNames, nextReleaseId, releaseChanged, sha256Sums, zipMembers, zipNames } from "../lib/releases.mjs";

const DATE = "2026-11-18";
const ver = (series: string, layer: string, fmt: string, hash8: string): Version => {
  const ext = ({ geojson: "geojson.zip", gpkg: "gpkg.zip", shp: "shp.zip", cog: "tif" } as Record<string, string>)[fmt];
  return { series, layer, fmt, date: DATE, bytes: 10, sha256: hash8.repeat(8), name: versionedName({ series, layer, date: DATE, sha256: hash8.repeat(8), ext }) };
};

const LOTS = ver("1948", "lots", "gpkg", "aaaaaaaa");
const CITY = ver("1948", "citywide", "cog", "bbbbbbbb");
const SHEET = ver("1948", "sheet-01", "cog", "cccccccc");

const manifest = {
  files: {
    "1948/lots.gpkg": { series: "1948", layer: "lots", fmt: "gpkg", name: LOTS.name, bytes: 10, sha256: LOTS.sha256, fingerprint: "f", updated: DATE },
    "1948/citywide.tif": {
      series: "1948", layer: "citywide", fmt: "cog", name: CITY.name, bytes: 10, sha256: CITY.sha256, fingerprint: "f", updated: DATE,
      extra: { "README.txt": "sfzoning-1948-citywide-README.txt", "LICENSE.txt": "sfzoning-1948-citywide-LICENSE.txt" },
      extraSha256: { "README.txt": "r".repeat(64), "LICENSE.txt": "l".repeat(64) },
    },
  },
  sheets: { "1948": { updated: DATE, readme: "", items: [{ sheet: "S01", number: "01", name: SHEET.name, bytes: 5, md5: "m", sha256: "s", updated: DATE }] } },
} as unknown as Manifest;

const history = { versions: [LOTS, CITY, SHEET] };
const { names, extras } = currentNames(manifest);
const release = (id = DATE) => ({ id, date: DATE, names, extras, zips: {} as never });

describe("currentNames", () => {
  it("lists every name sorted, and the sidecars with their hashes", () => {
    expect(names).toEqual([LOTS.name, CITY.name, SHEET.name].sort());
    expect(extras).toEqual([
      { series: "1948", name: "sfzoning-1948-citywide-LICENSE.txt", sha256: "l".repeat(64) },
      { series: "1948", name: "sfzoning-1948-citywide-README.txt", sha256: "r".repeat(64) },
    ]);
  });
});

describe("releaseChanged", () => {
  it("is true with no releases", () => expect(releaseChanged({ releases: [] }, names, extras)).toBe(true));
  it("is false for the same names and extras", () => expect(releaseChanged({ releases: [release()] }, names, extras)).toBe(false));
  it("is true when one extra hash differs", () => {
    const changed = extras.map((e: { sha256: string }, i: number) => (i === 0 ? { ...e, sha256: "x".repeat(64) } : e));
    expect(releaseChanged({ releases: [release()] }, names, changed)).toBe(true);
  });
  it("is true when a name differs", () => {
    expect(releaseChanged({ releases: [release()] }, names.slice(1), extras)).toBe(true);
  });
});

describe("nextReleaseId", () => {
  it("adds -2, -3 for later releases on one day", () => {
    const rec = { releases: [] as ReturnType<typeof release>[] };
    const ids: string[] = [];
    for (let i = 0; i < 3; i++) {
      const id = nextReleaseId(rec, DATE);
      ids.push(id);
      rec.releases.push(release(id));
    }
    expect(ids).toEqual(["2026-11-18", "2026-11-18-2", "2026-11-18-3"]);
    for (const id of ids) expect(RELEASE_ID.test(id)).toBe(true);
    expect(RELEASE_ID.test("2026-11-18-1")).toBe(false);
    expect(RELEASE_ID.test("2026-11-18-02")).toBe(false);
  });
});

describe("zips", () => {
  it("names the zips by release id", () => {
    expect(zipNames("2026-11-18-2").shp).toBe("sfzoning-2026-11-18-2-shp.zip");
    expect(zipNames("2026-11-18").all).toBe("sfzoning-2026-11-18.zip");
  });

  it("puts every name and sidecar in all, and only matching formats in the others", () => {
    const m = zipMembers(release(), history);
    expect(m.all).toHaveLength(5);
    expect(m.gpkg).toEqual([LOTS.name]);
    expect(m.shp).toEqual([]);
  });
});

describe("sha256Sums", () => {
  it("writes one sha256sum line per entry", () => {
    expect(sha256Sums([{ name: "a.tif", sha256: "ab" }])).toBe("ab  a.tif\n");
  });
});
