#!/usr/bin/env node
// Publishes the GIS downloads (`sfzmaps downloads` output in pipeline/data/downloads) to the Blob
// store and writes lib/downloads.json, which the /data page reads. Nothing is ever overwritten or
// (apart from superseded downloads/ files after 30 days) deleted.
//
//   node --env-file=.env.local scripts/publish-downloads.mjs                # needs BLOB_READ_WRITE_TOKEN
//   ... scripts/publish-downloads.mjs --dry-run                             # lists, prints the plan, uploads nothing
//   ... scripts/publish-downloads.mjs --local http://localhost:8787         # manifest for the local tile server; no token, no requests
//   ... scripts/publish-downloads.mjs --skip-dvc-check                      # emergencies only
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

const days = (a, b) => Math.round((Date.parse(b) - Date.parse(a)) / DAY);

/**
 * What to publish for the vector and citywide files. A file whose fingerprint matches the
 * manifest keeps its url and date; a changed one goes to a new hashed folder dated `today`, and
 * its old urls (file and sidecars) are superseded. Returns every superseded url so far.
 */
export function planDownloads(index, manifest, today, base = BASE) {
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
      return `${base}/${to}`;
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
  return { upload, files, superseded };
}

/**
 * What to publish for the per-sheet GeoTIFFs. `listing` is the set of names on the store (it needs
 * only `has`). A sheet at `dvc/files/md5/<2>/<30>` links there; any other is uploaded under
 * downloads/sheets/<md5>/. A sheet's date moves only when its md5 does; a series' date is the
 * latest of its sheets'. An old downloads/sheets url whose md5 changed is superseded.
 */
export function planSheets(index, manifest, listing, today, base = BASE) {
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
      if (!listing.has(dvc)) {
        name = `downloads/sheets/${s.md5}/${s.name}`;
        upload.push({ key: `${series}/sheet-${s.sheet}`, local: `${series}/${s.sheet}/warped_paper.tif`, to: name, root: "work", bytes: s.bytes });
      }
      const same = before && before.md5 === s.md5;
      if (before && !same && before.url?.includes("/downloads/sheets/")) superseded.push({ url: before.url, since: today });
      return { sheet: s.sheet, name: s.name, url: `${base}/${name}`, bytes: s.bytes, md5: s.md5, updated: same ? before.updated : today };
    });
    const latest = items.map((i) => i.updated).sort().at(-1) ?? old?.updated ?? today;
    sheets[series] = { updated: latest, readme: index.readme?.[series] ?? "", items };
  }
  return { upload, sheets, superseded };
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

const pathnameOf = (url) => decodeURIComponent(new URL(url).pathname.replace(/^\/+/, ""));
const localPath = (u) => path.join(u.root === "work" ? WORK : DOWNLOADS, u.local);
const requestsFor = (bytes) => (bytes > MULTIPART_MIN ? 2 + Math.ceil(bytes / PART_BYTES) : 1);

function localManifest(index, base, today) {
  const everything = { has: () => true };
  const d = planDownloads(index, {}, today, base);
  const s = planSheets(index, {}, everything, today, base);
  return { d, s };
}

function build(index, d, s, today, base) {
  return {
    generated: today, base, license: index.license, files: d.files, sheets: s.sheets, superseded: d.superseded,
  };
}

function argValue(flag) {
  const i = process.argv.indexOf(flag);
  return i >= 0 ? process.argv[i + 1] : undefined;
}

async function main() {
  const dryRun = process.argv.includes("--dry-run");
  const local = argValue("--local")?.replace(/\/+$/, "");
  const today = new Date().toISOString().slice(0, 10);
  const index = JSON.parse(readFileSync(path.join(DOWNLOADS, "index.json"), "utf8"));

  if (local) {
    const { d, s } = localManifest(index, local, today);
    writeFileSync(MANIFEST, JSON.stringify(build(index, { ...d, superseded: [] }, s, today, local), null, 1) + "\n");
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
  const { list, put, del } = await import("@vercel/blob");
  const sizes = new Map();
  const counter = { n: 0 };
  await listSizes(list, "downloads/", sizes, counter);
  await listSizes(list, "dvc/files/md5/", sizes, counter);

  const d = planDownloads(index, manifest, today);
  const s = planSheets(index, manifest, sizes, today);
  const superseded = [...d.superseded, ...s.superseded];
  const todo = [...d.upload, ...s.upload].filter((u) => !sizes.has(u.to));
  const stale = expired(superseded, today).filter((u) => pathnameOf(u).startsWith("downloads/"));
  let requests = counter.n;
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
  writeFileSync(MANIFEST, JSON.stringify(build(index, { ...d, superseded: keep }, s, today, BASE), null, 1) + "\n");
  console.log(`wrote ${MANIFEST}; commit it and deploy.`);
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  main().catch((e) => {
    console.error(e.message);
    process.exit(1);
  });
}
