#!/usr/bin/env node
// Uploads content-addressed files to the Blob store, for `sfzmaps ship` (answer packs and DVC
// objects). It is given only files a listing (blob-list.mjs) showed as missing, and makes one
// request per upload: no existence checks before or after (ship lists again to confirm). It never
// overwrites: a name is the hash of its bytes, and each local file's hash is checked against the
// name before it is sent.
//
//   node --env-file=.env.local scripts/blob-put.mjs < manifest.json     # needs BLOB_READ_WRITE_TOKEN
//
// manifest.json: [{"file": "/abs/local/path", "name": "dvc/files/md5/ab/cdef...", "alg": "md5"|"sha256",
//                  "hash": "<hex>", "contentType": "application/octet-stream"}]
// Prints one JSON line per file: {"name", "status": "uploaded"|"error", "error"?}.

import { createHash } from "node:crypto";
import { createReadStream, statSync } from "node:fs";
import { fileURLToPath } from "node:url";

const ONE_YEAR = 31_536_000;
const NAME = /^(dvc\/files\/md5\/[0-9a-f]{2}\/[0-9a-f]{30}(\.dir)?|cache\/packs\/[0-9a-f]{64}\.(zip|keys\.gz)|cache\/(needed|shipped)\/[0-9a-f]{64}\.(keys\.gz|json))$/;

export const hashFile = (file, alg) =>
  new Promise((resolve, reject) => {
    const h = createHash(alg);
    createReadStream(file).on("data", (d) => h.update(d)).on("end", () => resolve(h.digest("hex"))).on("error", reject);
  });

/** Upload one manifest entry (already known to be missing). Injectable for tests. */
export async function putOne(entry, { put } = {}) {
  const { file, name, alg, hash } = entry;
  if (!NAME.test(name)) throw new Error(`${name}: not a content-addressed name this uploader accepts`);
  const hex = name.startsWith("dvc/") ? name.slice("dvc/files/md5/".length).replace("/", "").replace(/\.dir$/, "") : hash;
  if (hash !== hex) throw new Error(`${name}: manifest hash ${hash} doesn't match the name`);
  if ((await hashFile(file, alg)) !== hash) throw new Error(`${name}: ${file} doesn't hash to its name`);
  const size = statSync(file).size;
  await put(name, createReadStream(file), {
    access: "public",
    addRandomSuffix: false,
    allowOverwrite: false,
    cacheControlMaxAge: ONE_YEAR,
    contentType: entry.contentType || "application/octet-stream",
    multipart: size > 50_000_000,
  });
  return "uploaded";
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
