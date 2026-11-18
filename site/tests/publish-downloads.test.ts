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
      { to: blobPath("1948", lotsName), local: "1948/1948-lots.gpkg.zip", root: "downloads", bytes: 10, sha256: SHA },
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

  it("a kept entry missing from history is re-recorded (a lost history repairs itself); with history intact nothing is added", () => {
    const first = run(idx({ "1948/lots.gpkg": lots() }, { "1948": [sheet()] }), empty, hist, {}, new Map([[DVC, 5]]));
    const manifest = { files: first.files, sheets: first.sheets };
    const again = (history: { versions: Version[] }) =>
      run(idx({ "1948/lots.gpkg": lots() }, { "1948": [sheet()] }), manifest, history, first.numbers, new Map([[DVC, 5]]));
    expect(again(hist).newVersions).toEqual(first.newVersions);
    expect(again({ versions: first.newVersions }).newVersions).toEqual([]);
  });

  it("an index entry without made throws, naming the key and the fix", () => {
    const { made: _m, ...bad } = lots();
    void _m;
    expect(() => run(idx({ "1948/lots.gpkg": bad }))).toThrow(/1948\/lots\.gpkg.*uv run sfzmaps downloads/);
  });
});

describe("plan: citywide sidecars", () => {
  const CITY_README = "1921/1921-citywide.README.txt";
  const CITY_LICENSE = "1921/1921-citywide.LICENSE.txt";
  const cityIndex = (made: string, readme = "a".repeat(64), license = "b".repeat(64)) => ({
    files: { "1921/citywide.cog": { ...city, made } },
    sheets: {},
    readme: { "1948": "R" },
    license: "L",
    sidecars: { [CITY_README]: { sha256: readme, bytes: 7 }, [CITY_LICENSE]: { sha256: license, bytes: 8 } },
  });
  const cityName = (date: string, ext: string) => versionedName({ series: "1921", layer: "citywide", date, sha256: SHA2, ext });

  it("keeps a citywide entry whose image and sidecars are unchanged: no upload, no new version", () => {
    const first = run(cityIndex("2026-11-03"));
    expect(first.files["1921/citywide.cog"].extraSha256).toEqual({ "README.txt": "a".repeat(64), "LICENSE.txt": "b".repeat(64) });
    const manifest = { files: first.files, sheets: {} };
    const again = run(cityIndex("2026-11-03"), manifest, { versions: first.newVersions }, {}, new Map(), "2026-11-18");
    expect(again.uploads).toEqual([]);
    expect(again.newVersions).toEqual([]);
    expect(again.files).toEqual(manifest.files);
  });

  it("a changed README with the same image publishes image, README and LICENSE anew under the new made date, and records one version", () => {
    const first = run(cityIndex("2026-11-03"));
    const next = run(cityIndex("2026-11-10", "c".repeat(64)), { files: first.files, sheets: {} }, { versions: first.newVersions });
    const stem = cityName("2026-11-10", "tif");
    const readme = cityName("2026-11-10", "README.txt");
    const license = cityName("2026-11-10", "LICENSE.txt");
    const entry = next.files["1921/citywide.cog"];
    expect(entry).toMatchObject({ name: stem, updated: "2026-11-10", extra: { "README.txt": readme, "LICENSE.txt": license } });
    expect(entry.extraSha256).toEqual({ "README.txt": "c".repeat(64), "LICENSE.txt": "b".repeat(64) });
    const to = next.uploads.map((u) => u.to);
    for (const n of [stem, readme, license]) {
      expect(to).toContain(blobPath("1921", n));
      expect(to).toContain(blobPath("1921", `${n}.sha256`));
    }
    expect(next.uploads).toHaveLength(6);
    expect(next.newVersions.map((v) => v.name)).toEqual([stem]);
  });

  it("an entry written before extraSha256 existed is versioned anew under a later made date, even when every hash matches", () => {
    const first = run(cityIndex("2026-11-03"));
    const { extraSha256: _old, ...oldEntry } = first.files["1921/citywide.cog"];
    void _old;
    const next = run(cityIndex("2026-11-10"), { files: { "1921/citywide.cog": oldEntry }, sheets: {} }, { versions: first.newVersions });
    const stem = cityName("2026-11-10", "tif");
    expect(next.files["1921/citywide.cog"]).toMatchObject({ name: stem, updated: "2026-11-10" });
    expect(next.uploads.map((u) => u.to)).toContain(blobPath("1921", stem));
    expect(next.files["1921/citywide.cog"].extraSha256).toEqual({ "README.txt": "a".repeat(64), "LICENSE.txt": "b".repeat(64) });
  });

  it("a legacy entry (no extraSha256) on the same made date proceeds without the collision error and records extraSha256", () => {
    const first = run(cityIndex("2026-11-03"));
    const { extraSha256: _old, ...oldEntry } = first.files["1921/citywide.cog"];
    void _old;
    const next = run(cityIndex("2026-11-03"), { files: { "1921/citywide.cog": oldEntry }, sheets: {} }, { versions: first.newVersions });
    expect(next.files["1921/citywide.cog"]).toMatchObject({ name: cityName("2026-11-03", "tif") });
    expect(next.files["1921/citywide.cog"].extraSha256).toEqual({ "README.txt": "a".repeat(64), "LICENSE.txt": "b".repeat(64) });
    expect(next.newVersions).toEqual([]);
  });

  it("a sidecar change on the same made date as the published image stops with a clear error instead of colliding on names", () => {
    const first = run(cityIndex("2026-11-03"));
    expect(() => run(cityIndex("2026-11-03", "c".repeat(64)), { files: first.files, sheets: {} }, { versions: first.newVersions })).toThrow(
      /1921\/citywide\.cog.*names would collide.*later day/,
    );
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
    expect(r.uploads[0]).toEqual({ to: blobPath("1948", name), local: "1948/10/warped_paper.tif", root: "work", bytes: 5, sha256: SHA2 });
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

describe("uploadMismatch", () => {
  const um = script.uploadMismatch as (u: unknown, f: string, o: unknown) => Promise<string | null>;
  const u = { local: "1948/x.zip", bytes: 10, sha256: SHA };
  it("passes a file with the indexed size and hash, and checks nothing for a text body", async () => {
    expect(await um(u, "/f", { size: () => 10, hash: async () => SHA })).toBeNull();
    expect(await um({ text: "t", bytes: 1 }, "/f", { size: () => 99, hash: async () => "no" })).toBeNull();
  });
  it("stops on a changed size or hash, saying to rerun sfzmaps downloads", async () => {
    expect(await um(u, "/f", { size: () => 11, hash: async () => SHA })).toMatch(/11 bytes.*uv run sfzmaps downloads/);
    expect(await um(u, "/f", { size: () => 10, hash: async () => "d".repeat(64) })).toMatch(/sha256 d+.*uv run sfzmaps downloads/);
  });
});

describe("sha256File and writeJsonAtomic", () => {
  it("hashes a file by streaming, and leaves no .part file behind", async () => {
    const { mkdtempSync, writeFileSync, existsSync, readFileSync } = await import("node:fs");
    const { tmpdir } = await import("node:os");
    const { createHash } = await import("node:crypto");
    const dir = mkdtempSync(`${tmpdir()}/pd-`);
    writeFileSync(`${dir}/a`, "hello");
    expect(await script.sha256File(`${dir}/a`)).toBe(createHash("sha256").update("hello").digest("hex"));
    script.writeJsonAtomic(`${dir}/j.json`, { a: 1 });
    expect(JSON.parse(readFileSync(`${dir}/j.json`, "utf8"))).toEqual({ a: 1 });
    expect(existsSync(`${dir}/j.json.part`)).toBe(false);
  });
});

describe("multipart parts", () => {
  const { Readable } = require("node:stream") as typeof import("node:stream");
  const split = script.splitParts as (s: AsyncIterable<Buffer>, n: number) => AsyncGenerator<Buffer>;
  const up = script.uploadInParts as (u: unknown, s: AsyncIterable<Buffer>, n: number) => Promise<unknown>;
  const stream = (total: number, chunk: number) =>
    Readable.from(Array.from({ length: Math.ceil(total / chunk) }, (_, i) => Buffer.alloc(Math.min(chunk, total - i * chunk), i)));
  const sizes = async (s: AsyncIterable<Buffer>, n: number) => {
    const out: number[] = [];
    for await (const p of split(s, n)) out.push(p.length);
    return out;
  };

  it("yields equal parts and a shorter last one, whatever the chunking", async () => {
    expect(await sizes(stream(25, 7), 10)).toEqual([10, 10, 5]);
    expect(await sizes(stream(25, 100), 10)).toEqual([10, 10, 5]);
  });
  it("yields no empty trailing part for an exact multiple", async () => {
    expect(await sizes(stream(30, 4), 10)).toEqual([10, 10, 10]);
  });
  it("keeps every byte in order", async () => {
    const parts: Buffer[] = [];
    for await (const p of split(stream(25, 7), 10)) parts.push(p);
    expect(Buffer.concat(parts).equals(Buffer.concat(await Readable.from(stream(25, 7)).toArray()))).toBe(true);
  });
  it("creates once, uploads parts 1..n in order one at a time, then completes", async () => {
    const calls: string[] = [];
    let busy = false;
    const uploader = {
      async uploadPart(n: number, body: Buffer) {
        expect(busy).toBe(false);
        busy = true;
        await new Promise((r) => setTimeout(r, 1));
        calls.push(`part ${n} ${body.length}`);
        busy = false;
        return { etag: `e${n}`, partNumber: n };
      },
      async complete(parts: unknown) {
        calls.push(`complete ${JSON.stringify(parts)}`);
        return { url: "u" };
      },
    };
    await up(uploader, stream(25, 7), 10);
    expect(calls).toEqual([
      "part 1 10", "part 2 10", "part 3 5",
      'complete [{"etag":"e1","partNumber":1},{"etag":"e2","partNumber":2},{"etag":"e3","partNumber":3}]',
    ]);
  });
});
