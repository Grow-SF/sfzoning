#!/usr/bin/env node
// Uploads content-addressed files to the Blob store, for `sfzmaps ship` (answer packs and DVC
// objects). It never overwrites: a name is the hash of its bytes, so a file already there is
// skipped. Before uploading, each local file's hash is checked against the one its name claims;
// afterwards the upload is confirmed by its size, without downloading it.
//
//   node --env-file=.env.local scripts/blob-put.mjs < manifest.json     # needs BLOB_READ_WRITE_TOKEN
//
// manifest.json: [{"file": "/abs/local/path", "name": "dvc/files/md5/ab/cdef...", "alg": "md5"|"sha256",
//                  "hash": "<hex>", "contentType": "application/octet-stream"}]
// Prints one JSON line per file: {"name", "status": "uploaded"|"present"|"error", "error"?}.
// Existence checks add a query string, so asking for a file before it exists never leaves a
// cached 404 at its real URL (the same trick as publish-tiles.mjs).

import { createHash } from "node:crypto";
import { createReadStream, statSync } from "node:fs";
import { fileURLToPath } from "node:url";

export const BASE = (process.env.NEXT_PUBLIC_TILES_BASE_URL || "https://zwgxwpoxqzfauqfg.public.blob.vercel-storage.com").replace(/\/+$/, "");
const ONE_YEAR = 31_536_000;
const NAME = /^(dvc\/files\/md5\/[0-9a-f]{2}\/[0-9a-f]{30}(\.dir)?|cache\/packs\/[0-9a-f]{64}\.(zip|keys\.gz)|cache\/needed\/[0-9a-f]{64}\.keys\.gz)$/;

export const hashFile = (file, alg) =>
  new Promise((resolve, reject) => {
    const h = createHash(alg);
    createReadStream(file).on("data", (d) => h.update(d)).on("end", () => resolve(h.digest("hex"))).on("error", reject);
  });

/** The size of the published file at `name`, or null if it isn't there (cache-busted HEAD). */
export async function remoteSize(name, fetchImpl = fetch) {
  const r = await fetchImpl(`${BASE}/${name}?check=${Date.now()}`, { method: "HEAD" });
  if (r.status === 404) return null;
  if (!r.ok) throw new Error(`${name}: HEAD ${r.status}`);
  return Number(r.headers.get("content-length"));
}

const pause = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

/** Upload one manifest entry unless it's already there. Injectable for tests. */
export async function putOne(entry, { put, fetchImpl = fetch, wait = pause } = {}) {
  const { file, name, alg, hash } = entry;
  if (!NAME.test(name)) throw new Error(`${name}: not a content-addressed name this uploader accepts`);
  const hex = name.startsWith("dvc/") ? name.slice("dvc/files/md5/".length).replace("/", "").replace(/\.dir$/, "") : hash;
  if (hash !== hex) throw new Error(`${name}: manifest hash ${hash} doesn't match the name`);
  if ((await hashFile(file, alg)) !== hash) throw new Error(`${name}: ${file} doesn't hash to its name`);
  const size = statSync(file).size;
  const there = await remoteSize(name, fetchImpl);
  if (there !== null) {
    if (there !== size) throw new Error(`${name}: already published with ${there} bytes, local has ${size}`);
    return "present";
  }
  try {
    await put(name, createReadStream(file), {
      access: "public",
      addRandomSuffix: false,
      allowOverwrite: false,
      cacheControlMaxAge: ONE_YEAR,
      contentType: entry.contentType || "application/octet-stream",
      multipart: size > 50_000_000,
    });
  } catch (e) {
    // a retry after a put that landed (read-after-put lag): the same bytes are there
    if ((await remoteSize(name, fetchImpl)) !== size) throw e;
  }
  for (const ms of [1000, 2000, 4000, 8000, 16000, 0]) {
    if ((await remoteSize(name, fetchImpl)) === size) return "uploaded";
    if (ms) await wait(ms);
  }
  throw new Error(`${name}: uploaded, but the store doesn't show ${size} bytes`);
}

async function main() {
  const chunks = [];
  for await (const c of process.stdin) chunks.push(c);
  const manifest = JSON.parse(Buffer.concat(chunks).toString() || "[]");
  const { put } = await import("@vercel/blob");
  let failed = 0;
  for (const entry of manifest) {
    try {
      console.log(JSON.stringify({ name: entry.name, status: await putOne(entry, { put }) }));
    } catch (e) {
      failed += 1;
      console.log(JSON.stringify({ name: entry.name, status: "error", error: e.message }));
    }
  }
  process.exit(failed ? 1 : 0);
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  main().catch((e) => {
    console.error(e.message);
    process.exit(1);
  });
}
