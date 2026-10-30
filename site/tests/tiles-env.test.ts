import { describe, expect, it } from "vitest";
import { spawnSync } from "node:child_process";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { checkTilesEnv } from "../scripts/check-tiles-env.mjs";

const SCRIPT = path.resolve(__dirname, "../scripts/check-tiles-env.mjs");

describe("checkTilesEnv", () => {
  it("lets dev and plain local builds through", () => {
    expect(checkTilesEnv({})).toBeNull();
    expect(checkTilesEnv({ NODE_ENV: "development" })).toBeNull();
  });
  it("fails a production build with no tiles URL", () => {
    expect(checkTilesEnv({ NODE_ENV: "production" })).toMatch(/NEXT_PUBLIC_TILES_BASE_URL is not set/);
    expect(checkTilesEnv({ VERCEL: "1" })).toMatch(/127\.0\.0\.1:8787/);
  });
  it("fails a production build with a non-https tiles URL", () => {
    expect(checkTilesEnv({ VERCEL: "1", NEXT_PUBLIC_TILES_BASE_URL: "http://tiles.example.com" })).toMatch(/must be https/);
    expect(checkTilesEnv({ VERCEL: "1", NEXT_PUBLIC_TILES_BASE_URL: "not a url" })).toMatch(/not a valid URL/);
  });
  it("accepts an https tiles URL", () => {
    expect(checkTilesEnv({ VERCEL: "1", NEXT_PUBLIC_TILES_BASE_URL: "https://tiles.sfzoning.com" })).toBeNull();
  });
  it("ALLOW_LOCAL_TILES=1 permits a local production build", () => {
    expect(checkTilesEnv({ NODE_ENV: "production", ALLOW_LOCAL_TILES: "1" })).toBeNull();
    expect(checkTilesEnv({ NODE_ENV: "production", ALLOW_LOCAL_TILES: "true" })).not.toBeNull();
  });
});

describe("check-tiles-env script", () => {
  // Run from an empty directory so no local .env file can change the outcome.
  const cwd = mkdtempSync(path.join(tmpdir(), "tiles-env-"));
  const run = (env: Record<string, string>) => {
    const childEnv: Record<string, string> = { PATH: process.env.PATH ?? "", ...env };
    return spawnSync(process.execPath, [SCRIPT], { cwd, env: childEnv as unknown as NodeJS.ProcessEnv, encoding: "utf8" });
  };
  it("exits 1 with a clear message on Vercel without a tiles URL", () => {
    const r = run({ VERCEL: "1" });
    expect(r.status).toBe(1);
    expect(r.stderr).toMatch(/Build stopped: NEXT_PUBLIC_TILES_BASE_URL is not set/);
  });
  it("exits 0 for an https URL, and for ALLOW_LOCAL_TILES=1", () => {
    expect(run({ VERCEL: "1", NEXT_PUBLIC_TILES_BASE_URL: "https://tiles.example.com" }).status).toBe(0);
    expect(run({ NODE_ENV: "production", ALLOW_LOCAL_TILES: "1" }).status).toBe(0);
  });
});
