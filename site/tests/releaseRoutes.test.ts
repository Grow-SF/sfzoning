import { describe, expect, it } from "vitest";
import { releaseTextResponse, releaseZipResponse } from "../lib/releaseRoutes";
import { RELEASE_ID } from "../lib/releases.mjs";
import { allReleases } from "../lib/releaseData";
import { extraFiles, releaseFiles } from "../lib/releaseFiles";
import history from "../lib/download-versions.json";

const u = (p: string) => new URL(`https://x.test/data/releases/${p}`);
const ID = "2026-11-18";

describe("release text routes", () => {
  it("urls with format=shp lists only shp zips", async () => {
    const r = releaseTextResponse("urls", ID, u(`${ID}/urls.txt?format=shp`));
    const lines = (await r.text()).trim().split("\n");
    expect(lines).toHaveLength(30);
    expect(lines.every((l) => l.endsWith(".shp.zip"))).toBe(true);
  });
  it("sums covers files and extras in sha256sum format", async () => {
    const r = releaseTextResponse("sums", ID, u(`${ID}/SHA256SUMS`));
    const lines = (await r.text()).trim().split("\n");
    const rel = allReleases()[0];
    expect(lines).toHaveLength(releaseFiles(rel, history as never).length + extraFiles(rel).length);
    expect(lines.every((l) => /^[0-9a-f]{64} {2}\S+$/.test(l))).toBe(true);
  });
  it("latest redirects to the newest id keeping the query", () => {
    const r = releaseTextResponse("urls", "latest", u("latest/urls.txt?format=shp"));
    expect(r.status).toBe(307);
    expect(r.headers.get("location")).toMatch(/\/data\/releases\/2026-11-18\/urls\.txt\?format=shp$/);
    expect(r.headers.get("cache-control")).toBe("no-store");
  });
  it("a bad filter is 400", () => {
    expect(releaseTextResponse("urls", ID, u(`${ID}/urls.txt?format=`)).status).toBe(400);
  });
  it("unknown ids are 404, including a well-formed one not in the record", () => {
    expect(releaseTextResponse("urls", "2026-11-17", u("x")).status).toBe(404);
    expect(RELEASE_ID.test("2026-11-18-2")).toBe(true);
    expect(releaseTextResponse("urls", "2026-11-18-2", u("x")).status).toBe(404);
    expect(releaseTextResponse("urls", "nope", u("x")).status).toBe(404);
  });
  it("manifest is JSON with filtered files and cff ignores the filter", async () => {
    const m = await releaseTextResponse("manifest", ID, u(`${ID}/manifest.json?format=shp`)).json();
    expect(m.id).toBe(ID);
    expect(m.files.every((f: { fmt: string }) => f.fmt === "shp")).toBe(true);
    expect(m.extras).toEqual([]);
    const c = releaseTextResponse("cff", ID, u(`${ID}/CITATION.cff?format=`));
    expect(c.status).toBe(200);
    expect(await c.text()).toContain(`version: "${ID}"`);
  });
});

describe("release zip route", () => {
  it("redirects a release zip to Blob", () => {
    const r = releaseZipResponse(ID, `sfzoning-${ID}-shp.zip`);
    expect(r.status).toBe(308);
    expect(r.headers.get("location")).toMatch(new RegExp(`/downloads/releases/${ID}/sfzoning-${ID}-shp\\.zip\\?download=1$`));
    expect(r.headers.get("cache-control")).toBe("public, max-age=31536000, immutable");
  });
  it("404s an unknown zip or release", () => {
    expect(releaseZipResponse(ID, `sfzoning-${ID}-geotiff.zip`).status).toBe(404);
    expect(releaseZipResponse("2026-11-17", "sfzoning-2026-11-17.zip").status).toBe(404);
  });
});
