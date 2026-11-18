#!/usr/bin/env node
// Publishes the GIS downloads (`sfzmaps downloads` output in pipeline/data/downloads) to the Blob
// store under permanent versioned names, and writes the three files the site reads:
// lib/downloads.json (the current set), lib/download-versions.json (every version ever released,
// append-only) and lib/sheet-numbers.json (the number each sheet id was given, never reused).
// Nothing is ever overwritten or deleted.
//
//   node --env-file=.env.local scripts/publish-downloads.mjs                # needs BLOB_READ_WRITE_TOKEN
//   ... scripts/publish-downloads.mjs --dry-run                             # lists, prints the plan, uploads nothing
//   ... scripts/publish-downloads.mjs --local http://127.0.0.1:8787        # manifest for local checking; no token, no requests
//   ... scripts/publish-downloads.mjs --skip-dvc-check                      # emergencies only
//
// Checking the /data page locally: serve pipeline/data from the tile server, then write a local manifest:
//
//   (cd site && TILES_DIR=../pipeline/data npm run tiles)
//   (cd site && node scripts/publish-downloads.mjs --local http://127.0.0.1:8787)
//
// Every file and sheet entry of that manifest carries "local": its path under <base>/downloads/ (or
// <base>/work/<series>/<sheet>/warped_paper.tif for a sheet). It has "local": true at the top; a real
// publish refuses it, so restore it first: git checkout lib/downloads.json
//
// Layout: every file goes to downloads/<series>/<versioned name> (see lib/downloadNames.mjs), with a
// <name>.sha256 beside it. A sheet whose DVC object (dvc/files/md5/..) is on the store gets a
// server-side copy under its versioned name; any other sheet is uploaded from work/. What is on the
// store comes from one listing pass (1,000 names per request), never a request per file. Rerunning
// after a failure uploads only what the listing doesn't show; a name listed at the wrong size stops
// the run (nothing is overwritten or deleted, so fix it by hand). The files are written only after
// a confirming listing shows every upload.

import { createHash } from "node:crypto";
import { createReadStream, existsSync, readFileSync, renameSync, statSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { EXTS, assignSheetNumbers, blobPath, parseName, versionedName } from "../lib/downloadNames.mjs";
import { BASE, ONE_YEAR, checkPipelineData } from "./publish-tiles.mjs";

const SITE = fileURLToPath(new URL("..", import.meta.url));
const DOWNLOADS = path.resolve(process.env.DOWNLOADS_DIR || path.join(SITE, "../pipeline/data/downloads"));
const WORK = path.resolve(process.env.WORK_DIR || path.join(DOWNLOADS, "../work"));
const MANIFEST = path.join(SITE, "lib/downloads.json");
const HISTORY = path.join(SITE, "lib/download-versions.json");
const NUMBERS = path.join(SITE, "lib/sheet-numbers.json");
const MULTIPART_MIN = 50_000_000;
const PART_BYTES = 64 * 1024 * 1024; // fewer requests than the SDK's fixed 8 MB parts (minimum is 5 MB)
const SIDECARS = { "README.txt": ".README.txt", "LICENSE.txt": ".LICENSE.txt" };

/** Today's calendar date in San Francisco (the owner's day), not UTC's. */
export const today = () => new Date().toLocaleDateString("en-CA", { timeZone: "America/Los_Angeles" });

/** An entry already published under a versioned name (the 2026-11-04 shape had a url and a plain name). */
const isVersioned = (series, entry) => Boolean(entry?.name && parseName(series, entry.name) && !parseName(series, entry.name).latest);

const dvcPath = (md5) => `dvc/files/md5/${md5.slice(0, 2)}/${md5.slice(2)}`;

/**
 * What to publish. Pure: no I/O.
 *
 * index: `index.json` (plus `sidecars`: { <index path>: { sha256, bytes } } for each citywide README
 * and LICENSE, which main() hashes from the local files). manifest/history/numbers: what is committed.
 * listing: Map of blob path -> size for downloads/ and dvc/files/md5/. today: the date a new sheet
 * version gets.
 *
 * A file whose sha256 matches the manifest's (and whose `name` is a versioned one; the 2026-11-04
 * shape had a url and a plain name, so it counts as absent) is kept as it is. Anything else gets a versioned name, an upload
 * (or, for a sheet whose DVC object is on the store, a server-side copy), and a `.sha256` beside it,
 * unless the listing shows that name at the right size already. A name listed at another size throws.
 * With `local`, nothing is uploaded: entries carry the path under the base they can be read from.
 */
export function plan(index, manifest, history, numbers, listing, today, { local = false } = {}) {
  const uploads = [];
  const copies = [];
  const newVersions = [];
  const known = new Set((history?.versions ?? []).map((v) => v.name));

  const checkListed = (to, bytes) => {
    if (!listing.has(to)) return false;
    if (listing.get(to) !== bytes) {
      throw new Error(`${to} is on the store already but listed at ${listing.get(to)} bytes, not ${bytes} ` +
        "(an aborted upload?); the store never overwrites or deletes, so this needs a look by hand");
    }
    return true;
  };
  /** The `.sha256` beside a versioned name: sha256sum format. */
  const checksum = (series, name, sha256) => {
    const text = `${sha256}  ${name}\n`;
    const to = blobPath(series, `${name}.sha256`);
    if (!local && !checkListed(to, Buffer.byteLength(text))) uploads.push({ to, text, bytes: Buffer.byteLength(text) });
  };
  const version = (v) => {
    if (!known.has(v.name)) {
      known.add(v.name);
      newVersions.push(v);
    }
  };

  const files = {};
  for (const [key, e] of Object.entries(index.files ?? {})) {
    if (!e.made) throw new Error(`${key}: index.json has no "made" date; run \`uv run sfzmaps downloads\` in pipeline/ first`);
    const old = manifest.files?.[key];
    if (isVersioned(e.series, old) && old.sha256 === e.sha256) {
      files[key] = old;
      // A kept entry is recorded too (a no-op when history has it), so a lost history repairs itself.
      version({ series: e.series, layer: e.layer, fmt: e.fmt, name: old.name, date: old.updated, bytes: old.bytes, sha256: old.sha256 });
      continue;
    }
    const name = versionedName({ series: e.series, layer: e.layer, date: e.made, sha256: e.sha256, ext: EXTS[e.fmt] });
    const f = {
      series: e.series, layer: e.layer, fmt: e.fmt, name, bytes: e.bytes, sha256: e.sha256,
      fingerprint: e.fingerprint, updated: e.made,
    };
    const publish = (rel, n, bytes, sha) => {
      if (local) return;
      const to = blobPath(e.series, n);
      if (!checkListed(to, bytes)) uploads.push({ to, local: rel, root: "downloads", bytes, sha256: sha });
      checksum(e.series, n, sha);
    };
    publish(e.path, name, e.bytes, e.sha256);
    if (local) f.local = e.path;
    version({ series: e.series, layer: e.layer, fmt: e.fmt, name, date: e.made, bytes: e.bytes, sha256: e.sha256 });
    if (e.extra?.length) {
      f.extra = {};
      if (local) f.extraLocal = {};
      for (const rel of e.extra) {
        const kind = Object.keys(SIDECARS).find((k) => rel.endsWith(SIDECARS[k]));
        if (!kind) throw new Error(`${key}: ${rel} is not a README.txt or LICENSE.txt sidecar`);
        const n = versionedName({ series: e.series, layer: e.layer, date: e.made, sha256: e.sha256, ext: kind });
        f.extra[kind] = n;
        if (local) f.extraLocal[kind] = rel;
        else {
          const side = index.sidecars?.[rel];
          if (!side) throw new Error(`${key}: no sha256 for ${rel}`);
          publish(rel, n, side.bytes, side.sha256);
        }
      }
    }
    files[key] = f;
  }

  const sheets = {};
  const nextNumbers = { ...numbers };
  for (const [series, entries] of Object.entries(index.sheets ?? {})) {
    const before = new Map((manifest.sheets?.[series]?.items ?? []).map((i) => [i.sheet, i]));
    const nums = entries.length ? assignSheetNumbers(entries.map((s) => s.sheet), numbers[series] ?? {}) : (numbers[series] ?? {});
    if (entries.length || numbers[series]) nextNumbers[series] = nums;
    const items = entries.map((s) => {
      const old = before.get(s.sheet);
      if (isVersioned(series, old) && old.md5 === s.md5) {
        version({ series, layer: `sheet-${old.number}`, fmt: "cog", name: old.name, date: old.updated, bytes: old.bytes, sha256: old.sha256 });
        return old;
      }
      const updated = old && old.md5 === s.md5 ? old.updated : today;
      const layer = `sheet-${nums[s.sheet]}`;
      const name = versionedName({ series, layer, date: updated, sha256: s.sha256, ext: "tif" });
      const item = { sheet: s.sheet, number: nums[s.sheet], name, bytes: s.bytes, md5: s.md5, sha256: s.sha256, updated };
      if (local) item.local = `work/${series}/${s.sheet}/warped_paper.tif`;
      else {
        const to = blobPath(series, name);
        if (!checkListed(to, s.bytes)) {
          const dvc = dvcPath(s.md5);
          if (listing.has(dvc)) copies.push({ from: dvc, to, bytes: s.bytes });
          else uploads.push({ to, local: `${series}/${s.sheet}/warped_paper.tif`, root: "work", bytes: s.bytes, sha256: s.sha256 });
        }
        checksum(series, name, s.sha256);
      }
      version({ series, layer, fmt: "cog", name, date: updated, bytes: s.bytes, sha256: s.sha256 });
      return item;
    });
    const latest = items.map((i) => i.updated).sort().at(-1) ?? manifest.sheets?.[series]?.updated ?? today;
    sheets[series] = { updated: latest, readme: index.readme?.[series] ?? "", items };
  }
  return { files, sheets, numbers: nextNumbers, newVersions, uploads, copies };
}

/** Why a real publish must not build on this manifest (a local or foreign one), or null. */
export function manifestProblem(manifest, base = BASE) {
  const fix = "restore lib/downloads.json from git (git checkout lib/downloads.json)";
  if (manifest.local) return `lib/downloads.json is a --local manifest; ${fix}`;
  if (manifest.base !== undefined && manifest.base !== base) return `lib/downloads.json is for ${manifest.base}, not ${base}; ${fix}`;
  return null;
}

/** JSON with object keys sorted, so two manifests compare by content, not key order. */
const stable = (v) =>
  JSON.stringify(v, (_, x) => (x && typeof x === "object" && !Array.isArray(x) ? Object.fromEntries(Object.entries(x).sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0))) : x));

/** True when `next` publishes the same thing as `prev` (only `generated` may differ). */
export function sameManifest(prev, next) {
  const pick = (m) => stable({ base: m.base, local: m.local, license: m.license, files: m.files, sheets: m.sheets });
  return pick(prev) === pick(next);
}

/** Every blob under prefix as name -> size, following the cursor (1,000 per request). */
async function listSizes(list, prefix, sizes, counter) {
  let cursor;
  do {
    const page = await list({ prefix, cursor, limit: 1000 });
    counter.n++;
    for (const b of page.blobs) sizes.set(b.pathname, b.size);
    cursor = page.hasMore ? page.cursor : undefined;
  } while (cursor);
}

const localPath = (u) => path.join(u.root === "work" ? WORK : DOWNLOADS, u.local);
const requestsFor = (bytes) => (bytes > MULTIPART_MIN ? 2 + Math.ceil(bytes / PART_BYTES) : 1);

/** Cut a readable stream into Buffers of exactly `size` bytes; only the last may be shorter, and none is empty. */
export async function* splitParts(stream, size = PART_BYTES) {
  let held = [];
  let n = 0;
  for await (const chunk of stream) {
    let c = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk);
    while (n + c.length >= size) {
      const cut = size - n;
      yield Buffer.concat([...held, c.subarray(0, cut)]);
      held = [];
      n = 0;
      c = c.subarray(cut);
    }
    if (c.length) {
      held.push(c);
      n += c.length;
    }
  }
  if (n) yield Buffer.concat(held);
}

/** One create, each part in order (no concurrency), then complete. `uploader` is from createMultipartUploader. */
export async function uploadInParts(uploader, stream, size = PART_BYTES) {
  const done = [];
  let number = 0;
  for await (const part of splitParts(stream, size)) {
    number += 1;
    done.push(await uploader.uploadPart(number, part));
  }
  return uploader.complete(done);
}

/** sha256 of a file, streamed (files run to ~130 MB). */
export function sha256File(file) {
  return new Promise((resolve, reject) => {
    const h = createHash("sha256");
    createReadStream(file).on("data", (c) => h.update(c)).on("error", reject).on("end", () => resolve(h.digest("hex")));
  });
}

/**
 * Whether a local file is still the one index.json described: size and sha256 must match what the
 * plan used to name it. @returns {Promise<string | null>} a message, or null when it matches (or
 * when the upload is a text body, which has nothing local to check). `size` and `hash` are injectable.
 */
export async function uploadMismatch(u, file, { size = (f) => statSync(f).size, hash = sha256File } = {}) {
  if (u.local === undefined) return null;
  const rerun = "rerun `uv run sfzmaps downloads` in pipeline/ and publish again";
  const bytes = size(file);
  if (bytes !== u.bytes) return `${file} is ${bytes} bytes but index.json says ${u.bytes}; it changed after the index was written; ${rerun}`;
  const sha = await hash(file);
  if (sha !== u.sha256) return `${file} has sha256 ${sha} but index.json says ${u.sha256}; it changed after the index was written; ${rerun}`;
  return null;
}

/** Write via a .part file and a rename, so a crash never leaves half a file. */
export function writeJsonAtomic(file, v) {
  const part = `${file}.part`;
  writeFileSync(part, JSON.stringify(v, null, 1) + "\n");
  renameSync(part, file);
}

const readJson = (file, fallback) => (existsSync(file) ? JSON.parse(readFileSync(file, "utf8")) : fallback);
const writeJson = (file, v) => writeFileSync(file, JSON.stringify(v, null, 1) + "\n");

/** sha256 and size of each citywide sidecar, from the local files (the index doesn't carry them). */
function hashSidecars(index) {
  const out = {};
  for (const e of Object.values(index.files ?? {})) {
    for (const rel of e.extra ?? []) {
      const buf = readFileSync(path.join(DOWNLOADS, rel));
      out[rel] = { sha256: createHash("sha256").update(buf).digest("hex"), bytes: buf.length };
    }
  }
  return out;
}

function build(index, files, sheets, today, base, local = false) {
  return { generated: today, base, ...(local ? { local: true } : {}), license: index.license, files, sheets };
}

function argValue(flag) {
  const i = process.argv.indexOf(flag);
  return i >= 0 ? process.argv[i + 1] : undefined;
}

async function main() {
  const dryRun = process.argv.includes("--dry-run");
  const local = argValue("--local")?.replace(/\/+$/, "");
  const now = today();
  const index = JSON.parse(readFileSync(path.join(DOWNLOADS, "index.json"), "utf8"));

  if (local) {
    const p = plan(index, { files: {}, sheets: {} }, { versions: [] }, {}, new Map(), now, { local: true });
    writeJson(MANIFEST, build(index, p.files, p.sheets, now, local, true));
    console.log(`wrote ${MANIFEST} for ${local} (${Object.keys(p.files).length} files); nothing uploaded. Don't commit it.`);
    return;
  }

  if (process.argv.includes("--skip-dvc-check")) {
    console.warn("WARNING: --skip-dvc-check: publishing without checking the data matches dvc.lock");
  } else {
    const problem = checkPipelineData();
    if (problem && !dryRun) throw new Error(`refusing to publish: ${problem}`);
    if (problem) console.warn(`(a real publish would refuse: ${problem})`);
  }

  const manifest = readJson(MANIFEST, { files: {}, sheets: {} });
  const bad = manifestProblem(manifest);
  if (bad) throw new Error(`refusing to publish: ${bad}`);
  const history = readJson(HISTORY, { versions: [] });
  const numbers = readJson(NUMBERS, {});
  const { list, put, copy, createMultipartUploader } = await import("@vercel/blob");
  const sizes = new Map();
  const counter = { n: 0 };
  await listSizes(list, "downloads/", sizes, counter);
  await listSizes(list, "dvc/files/md5/", sizes, counter);

  index.sidecars = hashSidecars(index);
  const p = plan(index, manifest, history, numbers, sizes, now);
  const bytes = p.uploads.reduce((n, u) => n + u.bytes, 0);
  const requests = counter.n + p.uploads.reduce((n, u) => n + requestsFor(u.bytes), 0) + p.copies.length + 1;

  console.log(`listing: ${sizes.size} names under downloads/ and dvc/files/md5/ in ${counter.n} request(s)`);
  console.log(`files: ${Object.keys(p.files).length}, sheets: ${Object.values(p.sheets).reduce((n, x) => n + x.items.length, 0)}, new versions: ${p.newVersions.length}`);
  console.log(`to upload: ${p.uploads.length} file(s), ${bytes} bytes; to copy on the store: ${p.copies.length}`);
  console.log(`requests: ${requests} (listing, uploads, copies and one confirming listing)`);
  if (dryRun) {
    for (const u of p.uploads) console.log(`would upload ${u.local ?? "(checksum text)"} -> ${u.to}`);
    for (const c of p.copies) console.log(`would copy ${c.from} -> ${c.to}`);
    return;
  }

  for (const u of p.uploads) {
    // The index is what named this file (and its hash); stop before it goes up under the wrong name.
    if (u.local) {
      const problem = await uploadMismatch(u, localPath(u));
      if (problem) throw new Error(`stopping before uploading ${u.to}: ${problem}`);
    }
    console.log(`upload ${u.to} (${u.bytes} bytes)`);
    const options = {
      access: "public",
      addRandomSuffix: false,
      allowOverwrite: false, // the store itself refuses to replace a file
      cacheControlMaxAge: ONE_YEAR,
    };
    if (u.text === undefined && u.bytes > MULTIPART_MIN) {
      const uploader = await createMultipartUploader(u.to, options);
      await uploadInParts(uploader, createReadStream(localPath(u), { highWaterMark: 1024 * 1024 }));
    } else {
      await put(u.to, u.text ?? createReadStream(localPath(u)), {
        ...options,
        ...(u.text !== undefined ? { contentType: "text/plain; charset=utf-8" } : {}),
      });
    }
  }
  for (const c of p.copies) {
    console.log(`copy ${c.from} -> ${c.to}`);
    await copy(c.from, c.to, {
      access: "public", addRandomSuffix: false, allowOverwrite: false, cacheControlMaxAge: ONE_YEAR, contentType: "image/tiff",
    });
  }
  if (p.uploads.length || p.copies.length) {
    // One listing confirms every name by size (names carry the hash, so size is what can differ).
    const after = new Map();
    await listSizes(list, "downloads/", after, { n: 0 });
    for (const u of [...p.uploads, ...p.copies]) {
      if (after.get(u.to) !== u.bytes) throw new Error(`${u.to}: listed ${after.get(u.to)} bytes, expected ${u.bytes}`);
    }
  }

  const next = build(index, p.files, p.sheets, now, BASE);
  // Order matters: numbers, then history, then the manifest last. A crash leaves the manifest
  // behind, so a rerun still sees work to do (and plan() re-records any version history lacks).
  if (stable(numbers) !== stable(p.numbers)) writeJsonAtomic(NUMBERS, p.numbers);
  if (p.newVersions.length) writeJsonAtomic(HISTORY, { versions: [...history.versions, ...p.newVersions] });
  if (!sameManifest(manifest, next)) writeJsonAtomic(MANIFEST, next);
  console.log(`published ${p.newVersions.length} new version(s); commit lib/downloads.json, lib/download-versions.json and lib/sheet-numbers.json, then deploy.`);
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  main().catch((e) => {
    console.error(e.message);
    process.exit(1);
  });
}
