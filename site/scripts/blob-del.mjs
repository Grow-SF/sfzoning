#!/usr/bin/env node
// Deletes unused content-addressed blobs, for `python -m sfzmaps.prune` (which decides what is
// unused from one listing and the shipped manifests, and lists again afterwards to confirm). It
// sends BATCH URLs per del() call (the SDK takes an array; one request), one batch at a time,
// paced to at most DELETES_PER_S deletes a second: Vercel counts every deleted blob as one
// advanced operation (free, but limited to 75 a second on Pro). It only accepts names in the
// layouts `sfzmaps ship` writes under dvc/ and cache/, so tiles and stats can never be deleted.
// No retries: on any error it stops, and prune can simply run again.
//
//   node --env-file=.env.local scripts/blob-del.mjs < names.json     # needs BLOB_READ_WRITE_TOKEN
//
// names.json: [{"name": "dvc/files/md5/ab/cdef...", "url": "https://<store>.public.blob.vercel-storage.com/<name>"}]
// Prints one JSON line per batch: {"deleted": <count>, "first": "<name>"}.

import { fileURLToPath } from "node:url";

export const BATCH = 100; // sfzmaps.prune.BATCH must match
export const DELETES_PER_S = 50; // sfzmaps.prune.DELETES_PER_S must match
const NAME = /^(dvc\/files\/md5\/[0-9a-f]{2}\/[0-9a-f]{30}(\.dir)?|cache\/packs\/[0-9a-f]{64}\.(zip|keys\.gz)|cache\/datapacks\/[0-9a-f]{64}\.zip|cache\/(needed|shipped)\/[0-9a-f]{64}\.(keys\.gz|json))$/;
const HOST = /^[a-z0-9]+\.public\.blob\.vercel-storage\.com$/;

/** The URLs to delete, each checked against its name; throws on anything else. */
export function checkedUrls(entries) {
  return entries.map(({ name, url }) => {
    if (!NAME.test(name)) throw new Error(`${name}: not a content-addressed name this script deletes`);
    const u = new URL(url);
    if (u.protocol !== "https:" || !HOST.test(u.host) || u.pathname !== `/${name}` || u.search) {
      throw new Error(`${url}: not the Blob store URL of ${name}`);
    }
    return url;
  });
}

/** Delete in batches, paced; del, sleep and log are injectable for tests. */
export async function deleteAll(entries, { del, sleep = (ms) => new Promise((r) => setTimeout(r, ms)), log = console.log }) {
  const urls = checkedUrls(entries);
  for (let i = 0; i < urls.length; i += BATCH) {
    const batch = urls.slice(i, i + BATCH);
    const start = Date.now();
    await del(batch);
    log(JSON.stringify({ deleted: batch.length, first: entries[i].name }));
    const wait = (batch.length * 1000) / DELETES_PER_S - (Date.now() - start);
    if (i + BATCH < urls.length && wait > 0) await sleep(wait);
  }
}

async function main() {
  const chunks = [];
  for await (const c of process.stdin) chunks.push(c);
  const entries = JSON.parse(Buffer.concat(chunks).toString() || "[]");
  const { del } = await import("@vercel/blob");
  await deleteAll(entries, { del });
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  main().catch((e) => {
    console.error(e.message);
    process.exit(1);
  });
}
