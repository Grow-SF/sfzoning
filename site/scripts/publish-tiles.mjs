#!/usr/bin/env node
// Uploads changed tiles to the Blob store without ever overwriting a file, so pages already open
// keep working. A changed file goes up as <name>-<hash of its bytes>.<ext>, and lib/published.json
// (committed and deployed with the site) maps its usual path to that name. Unchanged files stay put.
//
//   node --env-file=.env.local scripts/publish-tiles.mjs          # needs BLOB_READ_WRITE_TOKEN
//   ... scripts/publish-tiles.mjs --dry-run                       # says what it would do
//   ... scripts/publish-tiles.mjs --skip-dvc-check                # emergencies only (see below)
//
// It refuses to publish unless DVC says pipeline/data matches the checkout's dvc.lock: data from
// another checkout, or built before a code or input change, is how stale lot cards went live
// (2026-11-01). Build with `uv run sfzmaps build` first.
//
// Then commit lib/published.json and deploy. Until that deploy, the live site reads the old files.

import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { createReadStream, readFileSync, statSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const SITE = fileURLToPath(new URL("..", import.meta.url));
const OUT = path.resolve(process.env.TILES_DIR || path.join(SITE, "../pipeline/data/out"));
const MANIFEST = path.join(SITE, "lib/published.json");
export const BASE = (process.env.NEXT_PUBLIC_TILES_BASE_URL || "https://zwgxwpoxqzfauqfg.public.blob.vercel-storage.com").replace(/\/+$/, "");
export const ONE_YEAR = 31_536_000;

/** `a/stats.json` + hash → `a/stats-<hash>.json`. Only the last extension moves past the hash. */
export function hashedName(file, hash) {
  const ext = path.posix.extname(file);
  return `${file.slice(0, file.length - ext.length)}-${hash}${ext}`;
}

/**
 * What to do with one file. `published` is the file's name in the manifest, or undefined if it's
 * still at its usual path; `remoteHash` is the hash of the bytes at that name, or null if absent.
 */
export function plan(file, localHash, published, remoteHash) {
  if (remoteHash === localHash) return { action: "keep" };
  if (remoteHash === null && published === undefined) return { action: "upload", to: file, map: false };
  return { action: "upload", to: hashedName(file, localHash.slice(0, 12)), map: true };
}

const sha = (file) =>
  new Promise((resolve, reject) => {
    const h = createHash("sha256");
    createReadStream(file).on("data", (d) => h.update(d)).on("end", () => resolve(h.digest("hex"))).on("error", reject);
  });

async function remoteSha(name) {
  // A query string keeps this check out of the CDN's entry for the real URL: asking for a file
  // just before it exists would otherwise leave readers a cached 404.
  const r = await fetch(`${BASE}/${name}?check=${Date.now()}`);
  if (r.status === 404) return null;
  if (!r.ok) throw new Error(`${name}: ${r.status}`);
  const h = createHash("sha256");
  for await (const chunk of r.body) h.update(chunk);
  return h.digest("hex");
}

export const DVC = ["uv", "tool", "run", "--from", "dvc==3.67.1", "dvc"];

/**
 * Why the pipeline data can't be published, or null when `dvc status --json` (its stdout) says
 * every stage matches dvc.lock.
 */
export function dvcProblem(statusJson) {
  let stale;
  try {
    stale = Object.keys(JSON.parse(statusJson || "{}"));
  } catch {
    return "could not read `dvc status --json`";
  }
  if (stale.length === 0) return null;
  return `pipeline data doesn't match dvc.lock (${stale.slice(0, 6).join(", ")}${stale.length > 6 ? ", ..." : ""}); ` +
    "run `uv run sfzmaps build` in pipeline/ and commit dvc.lock";
}

/** Why publishing must stop (DVC's verdict, then whether dvc.lock is committed), or null. */
export function checkPipelineData() {
  const pipeline = path.join(SITE, "../pipeline");
  const [cmd, ...args] = DVC;
  let status;
  try {
    status = execFileSync(cmd, [...args, "status", "--json"], { cwd: pipeline, encoding: "utf8", stdio: ["ignore", "pipe", "pipe"] });
  } catch (e) {
    const why = e.code === "ENOENT" ? "uv isn't installed" : String(e.stderr || e.message).trim().split("\n").slice(-2).join(" ");
    return `could not run dvc status (${why})`;
  }
  const problem = dvcProblem(status);
  if (problem) return problem;
  const lock = execFileSync("git", ["status", "--porcelain", "--", "dvc.lock"], { cwd: pipeline, encoding: "utf8" });
  return lock.trim() ? "pipeline/dvc.lock has uncommitted changes; commit it, so the deploy names the data it shows" : null;
}

const pause = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

/**
 * Whether an upload reads back as `hash`. A read straight after `put` can see the store lagging
 * (a 404 or a miss), so it gets a few tries, about 30 seconds in all, before it counts as a mismatch.
 */
export async function uploadedBytesMatch(read, hash, wait = pause) {
  for (const ms of [1000, 2000, 4000, 8000, 16000]) {
    if ((await read()) === hash) return true;
    await wait(ms);
  }
  return (await read()) === hash;
}

/** Every file the site reads that the pipeline can change, by its usual path. */
function filesToPublish() {
  const pointer = JSON.parse(readFileSync(path.join(OUT, "shared/districts.json"), "utf8"));
  const series = pointer.eras;
  return [
    ...series.map((s) => `${s}/districts/stats.json`),
    pointer.archive,
    pointer.table,
    pointer.cards,
    "shared/districts.json", // last: it names the three above
  ];
}

async function main() {
  const dryRun = process.argv.includes("--dry-run");
  if (process.argv.includes("--skip-dvc-check")) {
    console.warn("WARNING: --skip-dvc-check: publishing without checking the data matches dvc.lock");
  } else {
    const problem = checkPipelineData();
    if (problem && !dryRun) throw new Error(`refusing to publish: ${problem}`);
    if (problem) console.warn(`(a real publish would refuse: ${problem})`);
  }
  const manifest = JSON.parse(readFileSync(MANIFEST, "utf8"));
  const { put } = await import("@vercel/blob");
  for (const file of filesToPublish()) {
    const local = path.join(OUT, file);
    const hash = await sha(local);
    const published = manifest[file];
    const step = plan(file, hash, published, await remoteSha(published ?? file));
    if (step.action === "keep") {
      console.log(`same     ${file}`);
      continue;
    }
    console.log(`${dryRun ? "would upload" : "upload"} ${file} -> ${step.to} (${statSync(local).size} bytes)`);
    if (dryRun) continue;
    if ((await remoteSha(step.to)) !== hash) {
      await put(step.to, createReadStream(local), {
        access: "public",
        addRandomSuffix: false,
        allowOverwrite: false, // the store itself refuses to replace a file
        cacheControlMaxAge: ONE_YEAR,
        multipart: statSync(local).size > 50_000_000,
      });
      if (!(await uploadedBytesMatch(() => remoteSha(step.to), hash))) {
        throw new Error(`${step.to}: uploaded bytes don't match`);
      }
    }
    if (step.map) {
      manifest[file] = step.to;
      // Saved as it goes, so an interrupted run loses nothing.
      writeFileSync(MANIFEST, JSON.stringify(manifest, Object.keys(manifest).sort(), 1) + "\n");
    }
  }
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  main().catch((e) => {
    console.error(e.message);
    process.exit(1);
  });
}
