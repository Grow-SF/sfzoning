#!/usr/bin/env node
// Publishes the GIS downloads (`sfzmaps downloads` output in pipeline/data/downloads) to the Blob
// store and writes lib/downloads.json, which the /data page reads. Nothing is ever overwritten or
// (apart from superseded downloads/ files after 30 days) deleted.
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
// Files are then at <base>/downloads/<index path> and sheets at <base>/work/<series>/<sheet>/warped_paper.tif.
// That manifest has "local": true; a real publish refuses it, so restore it first: git checkout lib/downloads.json
//
// Layout: a file goes to downloads/<first 12 of its fingerprint>/<readable name>, its README and
// LICENSE beside it. A sheet already shipped as a DVC object (dvc/files/md5/..) is linked there;
// a sheet under 8 MB is packed by `sfzmaps ship`, so it goes up as downloads/sheets/<md5>/<name>.
// What is on the store comes from one listing pass (1,000 names per request), never a request per
// file. Rerunning after a failure uploads only what the listing doesn't show.

import { createReadStream, readFileSync, statSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { BASE, ONE_YEAR, checkPipelineData } from "./publish-tiles.mjs";

const SITE = fileURLToPath(new URL("..", import.meta.url));
const DOWNLOADS = path.resolve(process.env.DOWNLOADS_DIR || path.join(SITE, "../pipeline/data/downloads"));
const WORK = path.resolve(process.env.WORK_DIR || path.join(DOWNLOADS, "../work"));
const MANIFEST = path.join(SITE, "lib/downloads.json");
const MULTIPART_MIN = 50_000_000;
const PART_BYTES = 8 * 1024 * 1024;
const DAY = 86_400_000;

/** Today's calendar date in San Francisco (the owner's day), not UTC's. */
export const today = () => new Date().toLocaleDateString("en-CA", { timeZone: "America/Los_Angeles" });

const days = (a, b) => Math.round((Date.parse(b) - Date.parse(a)) / DAY);

/**
 * What to publish for the vector and citywide files. A file whose fingerprint matches the
 * manifest keeps its url and date; a changed one goes to a new hashed folder dated `today`, and
 * its old urls (file and sidecars) are superseded. Returns every superseded url so far.
 */
export function planDownloads(index, manifest, today, base = BASE, local = false) {
  const prev = manifest.files ?? {};
  const files = {};
  const upload = [];
  const superseded = [...(manifest.superseded ?? [])];
  for (const [key, e] of Object.entries(index.files)) {
    const old = prev[key];
    if (old && old.fingerprint === e.fingerprint) {
      files[key] = old;
      continue;
    }
    const folder = `downloads/${e.fingerprint.slice(0, 12)}`;
    const place = (rel) => {
      const to = `${folder}/${path.posix.basename(rel)}`;
      upload.push({ key, local: rel, to, root: "downloads" });
      return local ? `${base}/downloads/${rel}` : `${base}/${to}`;
    };
    const url = place(e.path);
    const extra = {};
    for (const rel of e.extra ?? []) extra[path.posix.basename(rel)] = place(rel);
    files[key] = {
      series: e.series, layer: e.layer, fmt: e.fmt, url, bytes: e.bytes, sha256: e.sha256,
      fingerprint: e.fingerprint, updated: today, extra,
    };
    if (old) {
      for (const u of [old.url, ...Object.values(old.extra ?? {})]) superseded.push({ url: u, since: today });
    }
  }
  for (const [key, old] of Object.entries(prev)) {
    if (index.files[key]) continue; // gone from the index: its urls are cleaned up
    for (const u of [old.url, ...Object.values(old.extra ?? {})]) superseded.push({ url: u, since: today });
  }
  return { upload, files, superseded };
}

/**
 * What to publish for the per-sheet GeoTIFFs. `listing` is the set of names on the store (it needs
 * only `has`). A sheet at `dvc/files/md5/<2>/<30>` links there; any other is uploaded under
 * downloads/sheets/<md5>/. A sheet's date moves only when its md5 does; a series' date is the
 * latest of its sheets'. An old downloads/sheets url whose md5 changed is superseded.
 */
export function planSheets(index, manifest, listing, today, base = BASE, local = false) {
  const sheets = {};
  const upload = [];
  const superseded = [];
  for (const [series, entries] of Object.entries(index.sheets ?? {})) {
    const old = manifest.sheets?.[series];
    const oldBy = new Map((old?.items ?? []).map((i) => [i.sheet, i]));
    const items = entries.map((s) => {
      const before = oldBy.get(s.sheet);
      const dvc = `dvc/files/md5/${s.md5.slice(0, 2)}/${s.md5.slice(2)}`;
      let name = dvc;
      if (local) {
        name = `work/${series}/${s.sheet}/warped_paper.tif`;
      } else if (!listing.has(dvc)) {
        name = `downloads/sheets/${s.md5}/${s.name}`;
        upload.push({ key: `${series}/sheet-${s.sheet}`, local: `${series}/${s.sheet}/warped_paper.tif`, to: name, root: "work", bytes: s.bytes });
      }
      const same = before && before.md5 === s.md5;
      const url = `${base}/${name}`;
      if (before && before.url !== url && isPacked(before.url)) superseded.push({ url: before.url, since: today });
      return { sheet: s.sheet, name: s.name, url, bytes: s.bytes, md5: s.md5, updated: same ? before.updated : today };
    });
    const latest = items.map((i) => i.updated).sort().at(-1) ?? old?.updated ?? today;
    const now = new Set(entries.map((e) => e.sheet));
    for (const b of old?.items ?? []) if (!now.has(b.sheet) && isPacked(b.url)) superseded.push({ url: b.url, since: today });
    sheets[series] = { updated: latest, readme: index.readme?.[series] ?? "", items };
  }
  for (const [series, old] of Object.entries(manifest.sheets ?? {})) {
    if (index.sheets?.[series]) continue;
    for (const b of old.items ?? []) if (isPacked(b.url)) superseded.push({ url: b.url, since: today });
  }
  return { upload, sheets, superseded };
}

/** A sheet url this script uploaded itself (never a shipped dvc object, which is not ours to delete). */
const isPacked = (url) => typeof url === "string" && url.includes("/downloads/sheets/");

/** Every url the manifest's current files, sidecars and sheets point at. */
function liveUrls(files, sheets) {
  const urls = new Set();
  for (const f of Object.values(files)) {
    urls.add(f.url);
    for (const u of Object.values(f.extra ?? {})) urls.add(u);
  }
  for (const s of Object.values(sheets)) for (const i of s.items) urls.add(i.url);
  return urls;
}

/** `superseded` without duplicates and without any url that is current again (a revert). */
export function pruneSuperseded(superseded, files, sheets) {
  const live = liveUrls(files, sheets);
  const seen = new Set();
  return superseded.filter((x) => {
    if (live.has(x.url) || seen.has(x.url)) return false;
    seen.add(x.url);
    return true;
  });
}

/** Why a real publish must not build on this manifest (a local or foreign one), or null. */
export function manifestProblem(manifest, base = BASE) {
  const fix = "restore lib/downloads.json from git (git checkout lib/downloads.json)";
  if (manifest.local) return `lib/downloads.json is a --local manifest; ${fix}`;
  if (manifest.base !== undefined && manifest.base !== base) return `lib/downloads.json is for ${manifest.base}, not ${base}; ${fix}`;
  const urls = [
    ...Object.values(manifest.files ?? {}).flatMap((f) => [f.url, ...Object.values(f.extra ?? {})]),
    ...Object.values(manifest.sheets ?? {}).flatMap((s) => s.items.map((i) => i.url)),
    ...(manifest.superseded ?? []).map((x) => x.url),
  ];
  const bad = urls.find((u) => !u.startsWith(`${base}/`));
  return bad ? `lib/downloads.json has a url outside ${base} (${bad}); ${fix}` : null;
}

/** Superseded urls to delete: expired, and only ever under downloads/. */
export function deletable(superseded, today, base = BASE) {
  return expired(superseded, today).filter((u) => u.startsWith(`${base}/downloads/`));
}

/** Urls of superseded files `days` or more old; the caller deletes them. */
export function expired(superseded, today, ageDays = 30) {
  return superseded.filter((s) => days(s.since, today) >= ageDays).map((s) => s.url);
}

/** Every blob under prefix as name → size, following the cursor (1,000 per request). */
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

function build(index, files, sheets, superseded, today, base, local = false) {
  return {
    generated: today, base, ...(local ? { local: true } : {}), license: index.license, files, sheets, superseded,
  };
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
    const d = planDownloads(index, {}, now, local, true);
    const s = planSheets(index, {}, new Set(), now, local, true);
    writeFileSync(MANIFEST, JSON.stringify(build(index, d.files, s.sheets, [], now, local, true), null, 1) + "\n");
    console.log(`wrote ${MANIFEST} for ${local} (${Object.keys(d.files).length} files); nothing uploaded. Don't commit it.`);
    return;
  }

  if (process.argv.includes("--skip-dvc-check")) {
    console.warn("WARNING: --skip-dvc-check: publishing without checking the data matches dvc.lock");
  } else {
    const problem = checkPipelineData();
    if (problem && !dryRun) throw new Error(`refusing to publish: ${problem}`);
    if (problem) console.warn(`(a real publish would refuse: ${problem})`);
  }

  const manifest = JSON.parse(readFileSync(MANIFEST, "utf8"));
  const bad = manifestProblem(manifest);
  if (bad) throw new Error(`refusing to publish: ${bad}`);
  const { list, put, del } = await import("@vercel/blob");
  const sizes = new Map();
  const counter = { n: 0 };
  await listSizes(list, "downloads/", sizes, counter);
  await listSizes(list, "dvc/files/md5/", sizes, counter);

  const d = planDownloads(index, manifest, now);
  const s = planSheets(index, manifest, sizes, now);
  const superseded = pruneSuperseded([...d.superseded, ...s.superseded], d.files, s.sheets);
  const todo = [...d.upload, ...s.upload].filter((u) => !sizes.has(u.to));
  const stale = deletable(superseded, now);
  let requests = counter.n;
  if (todo.length) requests += Math.max(1, Math.ceil([...sizes.keys()].filter((n) => n.startsWith("downloads/")).length / 1000)); // size check
  for (const u of todo) requests += requestsFor(statSync(localPath(u)).size);
  const totalBytes = todo.reduce((n, u) => n + statSync(localPath(u)).size, 0);

  console.log(`listing: ${sizes.size} names under downloads/ and dvc/files/md5/ in ${counter.n} request(s)`);
  console.log(`files: ${Object.keys(d.files).length} (${d.upload.length} new uploads planned), sheets: ${Object.values(s.sheets).reduce((n, x) => n + x.items.length, 0)} (${s.upload.length} to upload as packed)`);
  console.log(`to upload now: ${todo.length} file(s), ${totalBytes} bytes; superseded: ${superseded.length}; expired to delete: ${stale.length}`);
  console.log(`requests: ${requests} listing and upload, ${stale.length ? 1 : 0} delete`);
  if (dryRun) {
    for (const u of todo) console.log(`would upload ${u.local} -> ${u.to}`);
    return;
  }

  for (const u of todo) {
    const file = localPath(u);
    const bytes = statSync(file).size;
    console.log(`upload ${u.to} (${bytes} bytes)`);
    await put(u.to, createReadStream(file), {
      access: "public",
      addRandomSuffix: false,
      allowOverwrite: false, // the store itself refuses to replace a file
      cacheControlMaxAge: ONE_YEAR,
      multipart: bytes > MULTIPART_MIN,
    });
    u.bytes = bytes;
  }
  if (todo.length) {
    // One listing confirms every upload by size (names carry the hash, so size is what can differ).
    const after = new Map();
    await listSizes(list, "downloads/", after, { n: 0 });
    for (const u of todo) {
      if (after.get(u.to) !== u.bytes) throw new Error(`${u.to}: listed ${after.get(u.to)} bytes, uploaded ${u.bytes}`);
    }
  }

  const gone = new Set(stale);
  const keep = superseded.filter((x) => !gone.has(x.url));
  if (stale.length) await del(stale);
  writeFileSync(MANIFEST, JSON.stringify(build(index, d.files, s.sheets, keep, now, BASE), null, 1) + "\n");
  console.log(`wrote ${MANIFEST}; commit it and deploy.`);
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  main().catch((e) => {
    console.error(e.message);
    process.exit(1);
  });
}
