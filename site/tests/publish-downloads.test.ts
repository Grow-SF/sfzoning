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

describe("local manifests", () => {
  const L = "http://127.0.0.1:8787";
  it("local urls point at <base>/downloads/<index path> and <base>/work/<series>/<sheet>", () => {
    const i = { files: { "1948/lots.geojson": { ...entry("a".repeat(40)), extra: ["1948/x.README.txt"] } } };
    const r = planDownloads(i, {}, "2026-11-04", L, true);
    expect(r.files["1948/lots.geojson"].url).toBe(`${L}/downloads/1948/1948-lots.geojson.zip`);
    const f = r.files["1948/lots.geojson"] as unknown as { extra: Record<string, string> };
    expect(f.extra["x.README.txt"]).toBe(`${L}/downloads/1948/x.README.txt`);
    const s = planSheets({ sheets: { "1948": [{ sheet: "10", name: "n.tif", md5: "ab" + "c".repeat(30), bytes: 1 }] }, readme: {} }, {}, new Set(), "2026-11-04", L, true);
    expect(sh(s).items[0].url).toBe(`${L}/work/1948/10/warped_paper.tif`);
  });

  it("a real publish refuses a local manifest or any url outside the real base", () => {
    const mp = script.manifestProblem as (...a: unknown[]) => string | null;
    expect(mp({ local: true, base: BASE, files: {}, sheets: {} }, BASE)).toMatch(/git checkout/);
    expect(mp({ base: "http://127.0.0.1:8787", files: {}, sheets: {} }, BASE)).toMatch(/git checkout/);
    expect(mp({ base: BASE, files: { k: { url: "http://127.0.0.1:8787/x", extra: {} } }, sheets: {} }, BASE)).toMatch(/outside/);
    expect(mp({ files: {}, sheets: {}, superseded: [] }, BASE)).toBeNull();
    expect(mp({ base: BASE, files: { k: { url: `${BASE}/downloads/a/b`, extra: {} } }, sheets: {}, superseded: [] }, BASE)).toBeNull();
  });
});

describe("superseded", () => {
  const prune = script.pruneSuperseded as (...a: unknown[]) => { url: string }[];
  it("a url that becomes current again (A to B to A) is not left to be deleted", () => {
    const m0 = { files: {}, superseded: [] };
    const a = planDownloads(index("a".repeat(40)), m0, "2026-01-01", BASE);
    const b = planDownloads(index("b".repeat(40)), { files: a.files, superseded: a.superseded }, "2026-02-01", BASE);
    expect(b.superseded).toHaveLength(1);
    const back = planDownloads(index("a".repeat(40)), { files: b.files, superseded: b.superseded }, "2026-03-01", BASE);
    expect(back.superseded).toHaveLength(2);
    expect(prune(back.superseded, back.files, {}).map((x) => x.url)).not.toContain(back.files["1948/lots.geojson"].url);
  });

  it("a vanished index entry supersedes its old urls", () => {
    const manifest = { files: { gone: { fingerprint: "x", url: "u1", updated: "d", extra: { "R.txt": "u2" } } }, superseded: [] };
    const r = planDownloads({ files: {} }, manifest, "2026-11-04", BASE);
    expect(r.superseded.map((x) => (x as { url: string }).url)).toEqual(["u1", "u2"]);
  });

  it("a packed sheet that moves to a dvc object supersedes its downloads url, and dvc urls never do", () => {
    const md5 = "ab" + "c".repeat(30);
    const dvc = `dvc/files/md5/ab/${"c".repeat(30)}`;
    const packed = `${BASE}/downloads/sheets/${md5}/1948-sheet-10.tif`;
    const idx = { sheets: { "1948": [{ sheet: "10", name: "1948-sheet-10.tif", md5, bytes: 1 }] }, readme: {} };
    type R = { superseded: { url: string }[] };
    const toDvc = planSheets(idx, { sheets: { "1948": { items: [{ sheet: "10", md5, url: packed, updated: "d" }] } } }, new Set([dvc]), "2026-11-04", BASE) as unknown as R;
    expect(toDvc.superseded.map((x) => x.url)).toEqual([packed]);
    const fromDvc = planSheets(idx, { sheets: { "1948": { items: [{ sheet: "10", md5: "old", url: `${BASE}/${dvc}`, updated: "d" }] } } }, new Set(), "2026-11-04", BASE) as unknown as R;
    expect(fromDvc.superseded).toEqual([]);
  });

  it("only urls under downloads/ are ever deleted", () => {
    const del = script.deletable as (...a: unknown[]) => string[];
    const old = [{ url: `${BASE}/downloads/a/b`, since: "2026-01-01" }, { url: `${BASE}/dvc/files/md5/ab/cd`, since: "2026-01-01" }];
    expect(del(old, "2026-11-04", BASE)).toEqual([`${BASE}/downloads/a/b`]);
  });
});

describe("today", () => {
  it("is a calendar date", () => {
    const t = script.today as () => string;
    expect(t()).toMatch(/^\d{4}-\d{2}-\d{2}$/);
  });
});

describe("manifest unchanged", () => {
  const same = script.sameManifest as (a: unknown, b: unknown) => boolean;
  const m = {
    generated: "2026-11-01", base: BASE, license: "L", superseded: [{ url: "u", since: "d" }],
    files: { a: { url: "x", bytes: 1, extra: { r: "y" } } }, sheets: { s: { updated: "d", readme: "", items: [] } },
  };
  it("ignores the generated date and key order", () => {
    const later = { files: { a: { extra: { r: "y" }, bytes: 1, url: "x" } }, sheets: m.sheets, superseded: m.superseded, license: "L", base: BASE, generated: "2026-11-04" };
    expect(same(m, later)).toBe(true);
  });
  it("sees a change in files, sheets or superseded", () => {
    expect(same(m, { ...m, files: { a: { url: "x2", bytes: 1, extra: { r: "y" } } } })).toBe(false);
    expect(same(m, { ...m, sheets: {} })).toBe(false);
    expect(same(m, { ...m, superseded: [] })).toBe(false);
  });
});

describe("names already on the store", () => {
  const resolve = script.resolveListed as (...a: unknown[]) => { todo: { key: string; to: string }[]; files: Record<string, { url: string; extra: Record<string, string> }> };
  const fp = "f".repeat(40);
  const sha = "1234567890ab" + "0".repeat(52);
  const plan = () => {
    const i = { files: { "1948/lots.geojson": { ...entry(fp), sha256: sha, extra: ["1948/x.README.txt"] } } };
    return planDownloads(i, {}, "2026-11-04", BASE) as unknown as { upload: { key: string; to: string; root: string; local: string }[]; files: Record<string, { url: string; extra: Record<string, string> }> };
  };
  const folder = `downloads/${fp.slice(0, 12)}`;

  it("skips a name already listed at the same size", () => {
    const p = plan();
    const sizes = new Map([[`${folder}/1948-lots.geojson.zip`, 10]]);
    const r = resolve(p.upload, p.files, sizes, () => 10, BASE);
    expect(r.todo.map((u) => u.to)).toEqual([`${folder}/x.README.txt`]);
    expect(r.files["1948/lots.geojson"].url).toBe(`${BASE}/${folder}/1948-lots.geojson.zip`);
  });

  it("moves a file whose listed size differs (an aborted run) to a fresh folder, sidecars too", () => {
    const p = plan();
    const sizes = new Map([[`${folder}/1948-lots.geojson.zip`, 3]]);
    const r = resolve(p.upload, p.files, sizes, () => 10, BASE);
    const fresh = `downloads/${fp.slice(0, 12)}-${sha.slice(0, 12)}`;
    expect(r.todo.map((u) => u.to)).toEqual([`${fresh}/1948-lots.geojson.zip`, `${fresh}/x.README.txt`]);
    expect(r.files["1948/lots.geojson"].url).toBe(`${BASE}/${fresh}/1948-lots.geojson.zip`);
    expect(r.files["1948/lots.geojson"].extra["x.README.txt"]).toBe(`${BASE}/${fresh}/x.README.txt`);
  });

  it("fails clearly when the fresh folder is wrong too, or a sheet's listed size differs", () => {
    const p = plan();
    const fresh = `downloads/${fp.slice(0, 12)}-${sha.slice(0, 12)}`;
    const sizes = new Map([[`${folder}/1948-lots.geojson.zip`, 3], [`${fresh}/1948-lots.geojson.zip`, 4]]);
    expect(() => resolve(p.upload, p.files, sizes, () => 10, BASE)).toThrow(/listed at 4 bytes.*10/);
    const sheet = [{ key: "1948/sheet-1", local: "1948/1/warped_paper.tif", to: "downloads/sheets/m/n.tif", root: "work" }];
    expect(() => resolve(sheet, {}, new Map([["downloads/sheets/m/n.tif", 1]]), () => 10, BASE)).toThrow(/downloads\/sheets\/m\/n\.tif.*listed at 1 bytes/);
  });
});
