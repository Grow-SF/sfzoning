#!/usr/bin/env node
// Lists what the Blob store holds under the given prefixes, for `sfzmaps ship`: one `list` call
// per 1,000 names (the API's page size), never a request per file.
//
//   node --env-file=.env.local scripts/blob-list.mjs dvc/files/md5/ cache/packs/    # needs BLOB_READ_WRITE_TOKEN
//
// Prints one JSON line per blob: {"name": "<pathname>", "size": <bytes>}.

import { fileURLToPath } from "node:url";

/** Every blob under prefix, following the cursor; list(opts) is @vercel/blob's. */
export async function listAll(list, prefix) {
  const out = [];
  let cursor;
  do {
    const page = await list({ prefix, cursor, limit: 1000 });
    for (const b of page.blobs) out.push({ name: b.pathname, size: b.size });
    cursor = page.hasMore ? page.cursor : undefined;
  } while (cursor);
  return out;
}

async function main() {
  const { list } = await import("@vercel/blob");
  for (const prefix of process.argv.slice(2)) {
    for (const b of await listAll(list, prefix)) console.log(JSON.stringify(b));
  }
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  main().catch((e) => {
    console.error(e.message);
    process.exit(1);
  });
}
