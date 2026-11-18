import { describe, expect, it } from "vitest";
import * as script from "../scripts/publish-downloads.mjs";
import { blobPath, versionedName } from "../lib/downloadNames.mjs";

type Upload = { to: string; local?: string; root?: string; text?: string; bytes: number };
type Version = { series: string; layer: string; fmt: string; name: string; date: string; bytes: number; sha256: string };
type Result = {
  files: Record<string, Record<string, unknown> & { name: string; updated: string; extra?: Record<string, string> }>;
  sheets: Record<string, { updated: string; readme: string; items: (Record<string, unknown> & { name: string; updated: string; number: string })[] }>;
  numbers: Record<string, Record<string, string>>;
  newVersions: Version[];
  uploads: Upload[];
  copies: { from: string; to: string }[];
};
const plan = script.plan as (...a: unknown[]) => Result;

const BASE = "https://b.example";
const SHA = "3f9a2c1b" + "0".repeat(56);
const SHA2 = "ce47cd18" + "1".repeat(56);
const MD5 = "ab" + "c".repeat(30);
const DVC = `dvc/files/md5/ab/${"c".repeat(30)}`;

const lots = (sha = SHA, made = "2026-11-04") => ({
  series: "1948", layer: "lots", fmt: "gpkg", path: "1948/1948-lots.gpkg.zip", fingerprint: "f".repeat(64),
  bytes: 10, sha256: sha, made, extra: [] as string[],
});
const city = {
  series: "1921", layer: "citywide", fmt: "cog", path: "1921/1921-citywide.tif", fingerprint: "e".repeat(64),
  bytes: 50, sha256: SHA2, made: "2026-11-03",
  extra: ["1921/1921-citywide.README.txt", "1921/1921-citywide.LICENSE.txt"],
};
const sidecars = {
  "1921/1921-citywide.README.txt": { sha256: "a".repeat(64), bytes: 7 },
  "1921/1921-citywide.LICENSE.txt": { sha256: "b".repeat(64), bytes: 8 },
};
const idx = (files: Record<string, unknown> = {}, sheets: Record<string, unknown[]> = {}) => ({
  files, sheets, readme: { "1948": "R" }, license: "L", sidecars,
});
const sheet = (id = "10", md5 = MD5, sha = SHA2) => ({ sheet: id, name: `1948-sheet-${id}.tif`, md5, bytes: 5, sha256: sha, mtime_ns: 1 });
const empty = { files: {}, sheets: {} };
const hist = { versions: [] as Version[] };
const run = (index: unknown, manifest: unknown = empty, history = hist, numbers = {}, listing = new Map<string, number>(), today = "2026-11-18") =>
  plan(index, manifest, history, numbers, listing, today);

const lotsName = versionedName({ series: "1948", layer: "lots", date: "2026-11-04", sha256: SHA, ext: "gpkg.zip" });

describe("plan: vector and citywide files", () => {
  it("names a new file, uploads it and its .sha256, and records one version", () => {
    const r = run(idx({ "1948/lots.gpkg": lots() }));
    expect(r.files["1948/lots.gpkg"]).toMatchObject({
      series: "1948", layer: "lots", fmt: "gpkg", name: lotsName, bytes: 10, sha256: SHA, updated: "2026-11-04",
    });
    const text = `${SHA}  ${lotsName}\n`;
    expect(r.uploads).toEqual([
      { to: blobPath("1948", lotsName), local: "1948/1948-lots.gpkg.zip", root: "downloads", bytes: 10 },
      { to: blobPath("1948", `${lotsName}.sha256`), text, bytes: Buffer.byteLength(text) },
    ]);
    expect(r.newVersions).toEqual([{ series: "1948", layer: "lots", fmt: "gpkg", name: lotsName, date: "2026-11-04", bytes: 10, sha256: SHA }]);
    expect(r.copies).toEqual([]);
  });

  it("citywide and its README/LICENSE share the image's date and hash, but each has its own checksum", () => {
    const r = run(idx({ "1921/citywide.cog": city }));
    const stem = versionedName({ series: "1921", layer: "citywide", date: "2026-11-03", sha256: SHA2, ext: "tif" });
    const readme = versionedName({ series: "1921", layer: "citywide", date: "2026-11-03", sha256: SHA2, ext: "README.txt" });
    const license = versionedName({ series: "1921", layer: "citywide", date: "2026-11-03", sha256: SHA2, ext: "LICENSE.txt" });
    expect(r.files["1921/citywide.cog"].name).toBe(stem);
    expect(r.files["1921/citywide.cog"].extra).toEqual({ "README.txt": readme, "LICENSE.txt": license });
    const to = r.uploads.map((u) => u.to);
    for (const n of [stem, readme, license]) {
      expect(to).toContain(blobPath("1921", n));
      expect(to).toContain(blobPath("1921", `${n}.sha256`));
    }
    expect(r.uploads.find((u) => u.to === blobPath("1921", `${readme}.sha256`))!.text).toBe(`${"a".repeat(64)}  ${readme}\n`);
    expect(r.uploads.find((u) => u.to === blobPath("1921", readme))).toMatchObject({ local: "1921/1921-citywide.README.txt", bytes: 7 });
    expect(r.newVersions.map((v) => v.name)).toEqual([stem]); // sidecars and checksums are not history
  });

  it("an unchanged file (same sha256) makes no upload and no version, and everything deep-equals its input", () => {
    const first = run(idx({ "1948/lots.gpkg": lots() }, { "1948": [sheet()] }), empty, hist, {}, new Map([[DVC, 5]]));
    const manifest = { files: first.files, sheets: first.sheets };
    const history = { versions: first.newVersions };
    const again = run(idx({ "1948/lots.gpkg": lots() }, { "1948": [sheet()] }), manifest, history, first.numbers, new Map([[DVC, 5]]), "2027-01-01");
    expect(again.uploads).toEqual([]);
    expect(again.copies).toEqual([]);
    expect(again.newVersions).toEqual([]);
    expect(again.files).toEqual(manifest.files);
    expect(again.sheets).toEqual(manifest.sheets);
    expect(again.numbers).toEqual(first.numbers);
  });

  it("a changed file gets a new versioned name and a version", () => {
    const first = run(idx({ "1948/lots.gpkg": lots() }));
    const next = run(idx({ "1948/lots.gpkg": lots(SHA2, "2026-12-01") }), { files: first.files, sheets: {} }, { versions: first.newVersions });
    expect(next.files["1948/lots.gpkg"].name).not.toBe(lotsName);
    expect(next.files["1948/lots.gpkg"].updated).toBe("2026-12-01");
    expect(next.newVersions).toHaveLength(1);
  });

  it("an old manifest entry without a name counts as absent", () => {
    const old = { "1948/lots.gpkg": { series: "1948", url: "u", sha256: SHA, fingerprint: "f", updated: "2026-11-04" } };
    const r = run(idx({ "1948/lots.gpkg": lots() }), { files: old, sheets: {} });
    expect(r.files["1948/lots.gpkg"].name).toBe(lotsName);
    expect(r.uploads.length).toBeGreaterThan(0);
  });

  it("a name already listed at the right size is not uploaded again; a wrong size throws naming the file", () => {
    const listing = new Map([[blobPath("1948", lotsName), 10]]);
    const r = run(idx({ "1948/lots.gpkg": lots() }), empty, hist, {}, listing);
    expect(r.uploads.map((u) => u.to)).toEqual([blobPath("1948", `${lotsName}.sha256`)]);
    expect(() => run(idx({ "1948/lots.gpkg": lots() }), empty, hist, {}, new Map([[blobPath("1948", lotsName), 3]]))).toThrow(
      new RegExp(`${lotsName}.*3 bytes.*10`),
    );
  });

  it("a version already in history is not added twice", () => {
    const history = { versions: [{ series: "1948", layer: "lots", fmt: "gpkg", name: lotsName, date: "2026-11-04", bytes: 10, sha256: SHA }] };
    const r = run(idx({ "1948/lots.gpkg": lots() }), empty, history);
    expect(r.newVersions).toEqual([]);
  });

  it("an index entry without made throws, naming the key and the fix", () => {
    const { made: _m, ...bad } = lots();
    void _m;
    expect(() => run(idx({ "1948/lots.gpkg": bad }))).toThrow(/1948\/lots\.gpkg.*uv run sfzmaps downloads/);
  });
});

describe("plan: sheets", () => {
  const sheetName = (date: string, sha = SHA2, number = "01") =>
    versionedName({ series: "1948", layer: `sheet-${number}`, date, sha256: sha, ext: "tif" });

  it("a sheet that is a DVC object on the store is copied, not uploaded; both get a .sha256", () => {
    const r = run(idx({}, { "1948": [sheet()] }), empty, hist, {}, new Map([[DVC, 5]]));
    const name = sheetName("2026-11-18");
    expect(r.copies).toEqual([{ from: DVC, to: blobPath("1948", name), bytes: 5 }]);
    expect(r.uploads.map((u) => u.to)).toEqual([blobPath("1948", `${name}.sha256`)]);
    expect(r.sheets["1948"].items[0]).toMatchObject({ sheet: "10", number: "01", name, bytes: 5, md5: MD5, sha256: SHA2, updated: "2026-11-18" });
    expect(r.sheets["1948"].readme).toBe("R");
    expect(r.numbers).toEqual({ "1948": { "10": "01" } });
    expect(r.newVersions).toEqual([{ series: "1948", layer: "sheet-01", fmt: "cog", name, date: "2026-11-18", bytes: 5, sha256: SHA2 }]);
  });

  it("a sheet that is not on the store is uploaded from work/", () => {
    const r = run(idx({}, { "1948": [sheet()] }));
    const name = sheetName("2026-11-18");
    expect(r.copies).toEqual([]);
    expect(r.uploads[0]).toEqual({ to: blobPath("1948", name), local: "1948/10/warped_paper.tif", root: "work", bytes: 5 });
    expect(r.uploads[1].to).toBe(blobPath("1948", `${name}.sha256`));
  });

  it("keeps its date while the md5 is unchanged (even for an old nameless item); a changed md5 gets today and a version", () => {
    const old = { sheet: "10", md5: MD5, url: "u", bytes: 5, name: "1948-sheet-10.tif", updated: "2026-11-04" };
    const manifest = { files: {}, sheets: { "1948": { updated: "2026-11-04", readme: "R", items: [old] } } };
    const same = run(idx({}, { "1948": [sheet()] }), manifest, hist, {}, new Map([[DVC, 5]]));
    const name = sheetName("2026-11-04");
    expect(same.sheets["1948"].items[0]).toMatchObject({ name, updated: "2026-11-04" });
    expect(same.sheets["1948"].updated).toBe("2026-11-04");
    const md5b = "ab" + "d".repeat(30);
    const changed = run(idx({}, { "1948": [sheet("10", md5b, "9".repeat(64))] }), manifest, hist, { "1948": { "10": "01" } });
    expect(changed.sheets["1948"].items[0].updated).toBe("2026-11-18");
    expect(changed.sheets["1948"].items[0].name).toBe(sheetName("2026-11-18", "9".repeat(64)));
    expect(changed.sheets["1948"].updated).toBe("2026-11-18");
    expect(changed.newVersions).toHaveLength(1);
  });

  it("dropping a sheet from the index removes it from sheets but keeps its number", () => {
    const r = run(idx({}, { "1948": [sheet("10"), sheet("11")] }));
    const next = run(idx({}, { "1948": [sheet("11")] }), { files: {}, sheets: r.sheets }, hist, r.numbers);
    expect(next.sheets["1948"].items.map((i) => i.sheet)).toEqual(["11"]);
    expect(next.numbers["1948"]).toEqual(r.numbers["1948"]);
    expect(Object.keys(next.numbers["1948"])).toContain("10");
  });

  it("a sheet name listed at another size throws", () => {
    const name = sheetName("2026-11-18");
    expect(() => run(idx({}, { "1948": [sheet()] }), empty, hist, {}, new Map([[blobPath("1948", name), 9]]))).toThrow(new RegExp(name));
  });
});

describe("plan: --local", () => {
  it("keeps versioned names, adds local paths, and uploads nothing", () => {
    const i = idx({ "1921/citywide.cog": city, "1948/lots.gpkg": lots() }, { "1948": [sheet()] });
    const r = plan(i, empty, hist, {}, new Map(), "2026-11-18", { local: true });
    expect(r.uploads).toEqual([]);
    expect(r.copies).toEqual([]);
    expect(r.files["1948/lots.gpkg"]).toMatchObject({ name: lotsName, local: "1948/1948-lots.gpkg.zip" });
    expect(r.files["1921/citywide.cog"]).toMatchObject({
      local: "1921/1921-citywide.tif",
      extraLocal: { "README.txt": "1921/1921-citywide.README.txt", "LICENSE.txt": "1921/1921-citywide.LICENSE.txt" },
    });
    expect(r.sheets["1948"].items[0]).toMatchObject({ local: "work/1948/10/warped_paper.tif" });
  });
});

describe("manifestProblem", () => {
  const mp = script.manifestProblem as (...a: unknown[]) => string | null;
  it("a real publish refuses a local manifest or another base", () => {
    expect(mp({ local: true, base: BASE, files: {}, sheets: {} }, BASE)).toMatch(/git checkout/);
    expect(mp({ base: "http://127.0.0.1:8787", files: {}, sheets: {} }, BASE)).toMatch(/git checkout/);
    expect(mp({ files: {}, sheets: {} }, BASE)).toBeNull();
    expect(mp({ base: BASE, files: {}, sheets: {} }, BASE)).toBeNull();
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
    generated: "2026-11-01", base: BASE, license: "L",
    files: { a: { name: "x", bytes: 1, extra: { r: "y" } } }, sheets: { s: { updated: "d", readme: "", items: [] } },
  };
  it("ignores the generated date and key order", () => {
    const later = { files: { a: { extra: { r: "y" }, bytes: 1, name: "x" } }, sheets: m.sheets, license: "L", base: BASE, generated: "2026-11-04" };
    expect(same(m, later)).toBe(true);
  });
  it("sees a change in files or sheets", () => {
    expect(same(m, { ...m, files: { a: { name: "x2", bytes: 1, extra: { r: "y" } } } })).toBe(false);
    expect(same(m, { ...m, sheets: {} })).toBe(false);
  });
});
