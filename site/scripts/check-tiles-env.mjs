#!/usr/bin/env node
// prebuild guard: a production build must point at real, https tiles. Without
// NEXT_PUBLIC_TILES_BASE_URL the site falls back to the local dev tile server
// (http://127.0.0.1:8787), which works on no one else's machine. `next dev` never runs
// this, so the localhost default still serves development.
//
// Production means VERCEL is set or NODE_ENV=production. ALLOW_LOCAL_TILES=1 lets a local
// production build (`ALLOW_LOCAL_TILES=1 npm run build`) use the local tile server.
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { BASE } from "./publish-tiles.mjs";

/** @returns {string | null} an error message, or null when the build may go ahead. */
export function checkTilesEnv(env) {
  const production = !!env.VERCEL || env.NODE_ENV === "production";
  if (!production || env.ALLOW_LOCAL_TILES === "1") return null;
  const url = (env.NEXT_PUBLIC_TILES_BASE_URL || "").trim();
  if (!url) {
    return (
      "NEXT_PUBLIC_TILES_BASE_URL is not set, so this production build would load map tiles from " +
      "http://127.0.0.1:8787. Set it to the https base URL of the published tiles " +
      "(or ALLOW_LOCAL_TILES=1 to test a production build against the local tile server)."
    );
  }
  let parsed;
  try {
    parsed = new URL(url);
  } catch {
    return `NEXT_PUBLIC_TILES_BASE_URL is not a valid URL: "${url}".`;
  }
  if (parsed.protocol !== "https:") {
    return (
      `NEXT_PUBLIC_TILES_BASE_URL must be https for a production build, got "${url}" ` +
      "(ALLOW_LOCAL_TILES=1 allows a local tile server)."
    );
  }
  return null;
}

/**
 * A production build must not ship a --local downloads manifest (links to 127.0.0.1) or one for
 * another store. @returns {string | null} an error message, or null.
 */
export function checkDownloadsManifest(env, manifest, base = BASE) {
  const production = !!env.VERCEL || env.NODE_ENV === "production";
  if (!production || env.ALLOW_LOCAL_TILES === "1") return null;
  const fix = "restore it with `git checkout site/lib/downloads.json`";
  if (manifest.local) return `lib/downloads.json is a local manifest (links to a local server); ${fix}.`;
  const urls = [
    ...Object.values(manifest.files ?? {}).flatMap((f) => [f.url, ...Object.values(f.extra ?? {})]),
    ...Object.values(manifest.sheets ?? {}).flatMap((s) => (s.items ?? []).map((i) => i.url)),
  ];
  const bad = urls.find((u) => !u.startsWith(`${base}/`));
  return bad ? `lib/downloads.json has a url outside ${base} (${bad}); ${fix}.` : null;
}

if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) {
  // Pick up .env.production(.local) the way `next build` will, so the check sees the same value.
  const { loadEnvConfig } = (await import("@next/env")).default; // CommonJS module
  loadEnvConfig(process.cwd(), false, { info: () => {}, error: console.error });
  const manifest = JSON.parse(readFileSync(new URL("../lib/downloads.json", import.meta.url), "utf8"));
  const err = checkTilesEnv(process.env) || checkDownloadsManifest(process.env, manifest);
  if (err) {
    console.error(`\nBuild stopped: ${err}\n`);
    process.exit(1);
  }
}
