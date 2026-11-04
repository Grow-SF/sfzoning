import { describe, expect, it } from "vitest";
import * as script from "../scripts/publish-downloads.mjs";

type Plan = { upload: { to: string }[]; files: Record<string, { url: string; updated: string }>; superseded: unknown[] };
const planDownloads = script.planDownloads as (...a: unknown[]) => Plan;
const planSheets = script.planSheets as (...a: unknown[]) => { upload: { to: string }[]; sheets: unknown };
const expired = script.expired as (...a: unknown[]) => string[];
type Series = { updated: string; readme: string; items: { url: string; updated: string }[] };
const sh = (r: { sheets: unknown }) => (r.sheets as Record<string, Series>)["1948"];
const BASE = "https://b.example";
const entry = (fp: string) => ({
  series: "1948", layer: "lots", fmt: "geojson", path: "1948/1948-lots.geojson.zip",
  fingerprint: fp, bytes: 10, sha256: "s" + fp, extra: [] as string[],
});
const index = (fp: string) => ({ files: { "1948/lots.geojson": entry(fp) }, sheets: {}, readme: {}, license: "L" });

describe("planDownloads", () => {
  it("keeps url and date when fingerprint unchanged", () => {
    const manifest = { files: { "1948/lots.geojson": { fingerprint: "a".repeat(40), url: "u", updated: "2026-01-01", extra: {} } }, superseded: [] };
    const r = planDownloads(index("a".repeat(40)), manifest, "2026-11-04", BASE);
    expect(r.upload).toEqual([]);
    expect(r.files["1948/lots.geojson"].url).toBe("u");
    expect(r.files["1948/lots.geojson"].updated).toBe("2026-01-01");
    expect(r.superseded).toEqual([]);
  });

  it("new fingerprint gets new hashed path and today's date and supersedes the old url", () => {
    const manifest = { files: { "1948/lots.geojson": { fingerprint: "old", url: "u-old", updated: "2026-01-01", extra: {} } }, superseded: [] };
    const fp = "abcdef0123456789".repeat(2);
    const r = planDownloads(index(fp), manifest, "2026-11-04", BASE);
    const to = `downloads/${fp.slice(0, 12)}/1948-lots.geojson.zip`;
    expect(r.upload.map((u: { to: string }) => u.to)).toEqual([to]);
    expect(r.files["1948/lots.geojson"]).toMatchObject({ url: `${BASE}/${to}`, updated: "2026-11-04" });
    expect(r.superseded).toEqual([{ url: "u-old", since: "2026-11-04" }]);
  });
});

describe("planSheets", () => {
  const sheet = (md5: string) => ({ sheet: "10", name: "1948-sheet-10.tif", md5, bytes: 5, sha256: "x" });
  const idx = (md5: string) => ({ sheets: { "1948": [sheet(md5)] }, readme: { "1948": "R" } });
  const md5 = "ab" + "c".repeat(30);
  const dvcName = `dvc/files/md5/ab/${"c".repeat(30)}`;

  it("packed sheet (not in listing) is uploaded under downloads/sheets", () => {
    const r = planSheets(idx(md5), { sheets: {} }, new Set(), "2026-11-04", BASE);
    expect(r.upload[0].to).toBe(`downloads/sheets/${md5}/1948-sheet-10.tif`);
    expect(sh(r).items[0].url).toBe(`${BASE}/downloads/sheets/${md5}/1948-sheet-10.tif`);
    expect(sh(r).readme).toBe("R");
  });

  it("sheet in the listing links to the DVC object", () => {
    const r = planSheets(idx(md5), { sheets: {} }, new Set([dvcName]), "2026-11-04", BASE);
    expect(r.upload).toEqual([]);
    expect(sh(r).items[0].url).toBe(`${BASE}/${dvcName}`);
  });

  it("sheet date unchanged when md5 unchanged", () => {
    const manifest = { sheets: { "1948": { updated: "2026-01-02", items: [{ sheet: "10", md5, updated: "2026-01-02" }] } } };
    const r = planSheets(idx(md5), manifest, new Set([dvcName]), "2026-11-04", BASE);
    expect(sh(r).items[0].updated).toBe("2026-01-02");
    expect(sh(r).updated).toBe("2026-01-02");
  });

  it("changed md5 gets today's date, and the series date is the latest", () => {
    const manifest = { sheets: { "1948": { updated: "2026-01-02", items: [{ sheet: "10", md5: "old", updated: "2026-01-02" }] } } };
    const r = planSheets(idx(md5), manifest, new Set(), "2026-11-04", BASE);
    expect(sh(r).items[0].updated).toBe("2026-11-04");
    expect(sh(r).updated).toBe("2026-11-04");
  });
});

describe("expired", () => {
  it("returns only entries 30+ days old", () => {
    const s = [
      { url: "old", since: "2026-10-05" },
      { url: "new", since: "2026-10-06" },
    ];
    expect(expired(s, "2026-11-04")).toEqual(["old"]);
  });
});
