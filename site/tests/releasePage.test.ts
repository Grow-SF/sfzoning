import { describe, expect, it } from "vitest";
import history from "../lib/download-versions.json";
import manifest from "../lib/downloads.json";
import { allReleases } from "../lib/releaseData";
import { changesView, releaseView, type ReleaseView, type Row } from "../lib/releaseView";
import type { Manifest, Version } from "../lib/downloadRows";
import type { ReleaseRecord } from "../lib/releases.mjs";
import { chipFilter, downloadCommand, filterQuery, formatZip } from "../lib/releaseFilter";
import { ERAS } from "../lib/eras";

const H = history as unknown as { versions: Version[] };
const M = manifest as unknown as Manifest;
const current = allReleases().at(-1)!;

const urlsOf = (v: ReleaseView): string[] =>
  v.editions.flatMap((e) =>
    e.rows.flatMap((r: Row) => (r.type === "file" ? [r.latestUrl, r.historyUrl] : [r.zipUrl ?? "", r.listUrl])),
  );

describe("releaseView", () => {
  const v = releaseView(current, H, M);

  it("counts the committed release: 469 files in 15 editions", () => {
    expect(v.id).toBe("2026-11-18");
    expect(v.latest).toBe(true);
    expect(v.totals.files).toBe(469);
    expect(v.totals.editions).toBe(15);
    expect(v.editions).toHaveLength(15);
  });

  it("lists 1921's lots first, as GeoJSON, GeoPackage, Shapefile", () => {
    const e = v.editions.find((x) => x.era === "1921")!;
    const first = e.rows.slice(0, 3).map((r) => (r.type === "file" ? `${r.file.layer}.${r.file.fmt}` : r.type));
    expect(first).toEqual(["lots.geojson", "lots.gpkg", "lots.shp"]);
  });

  it("groups 1921's 16 sheets in one row with the current sheet zip and CSV", () => {
    const g = v.editions.find((x) => x.era === "1921")!.rows.find((r) => r.type === "sheets");
    if (g?.type !== "sheets") throw new Error("no sheet group for 1921");
    expect(g.count).toBe(16);
    expect(g.sheets).toHaveLength(16);
    expect(g.zipUrl).toBe("/data/files/1921/sfzoning-1921-sheets-latest.zip");
    expect(g.listUrl).toBe("/data/sheets/1921");
  });

  it("an older release links no current-only sheet zip or CSV", () => {
    const old = { ...current, id: "2026-11-04", date: "2026-11-04" };
    const record: ReleaseRecord = { releases: [old, current] };
    const o = releaseView(old, H, M, record);
    expect(o.latest).toBe(false);
    expect(urlsOf(o).some((u) => u.includes("/data/sheets/") || u.includes("sheets-latest"))).toBe(false);
    const g = o.editions.find((x) => x.era === "1921")!.rows.find((r) => r.type === "sheets");
    if (g?.type !== "sheets") throw new Error("no sheet group for 1921");
    expect(g.zipUrl).toBeUndefined();
    expect(g.listUrl).toBe("/data/releases/2026-11-04/urls.txt?layer=sheets&edition=1921");
    expect(releaseView(current, H, M, record).previousId).toBe("2026-11-04");
  });
});

describe("chipFilter", () => {
  const eras = ERAS.map((e) => e.id);
  const present = eras.filter((e) => e !== "today");
  const on = { format: ["shp"], layer: ["lots", "districts", "scans", "sheets"], edition: present };

  it("a release without an edition still reads as every edition when all its chips are on", () => {
    const f = chipFilter(on, present);
    expect(formatZip(f)).toBe("shp");
    expect(filterQuery(f)).toBe("?format=shp");
  });
  it("turning a present edition off narrows; the absent one stays out of the query", () => {
    const f = chipFilter({ ...on, edition: present.filter((e) => e !== "1948") }, present);
    expect(formatZip(f)).toBeNull();
    expect(f.edition.has("today")).toBe(true);
    expect(filterQuery(f)).toContain("edition=");
    expect(filterQuery(f)).not.toContain("1948");
  });
});

describe("changesView", () => {
  it("the first release has no predecessor and adds every file", () => {
    const first = allReleases()[0];
    const c = changesView(first.id, H)!;
    expect(c.first).toBe(true);
    expect(c.added).toHaveLength(releaseView(first, H, M).totals.files);
    expect(c.changed).toEqual([]);
    expect(c.removed).toEqual([]);
    expect(changesView("nope", H)).toBeUndefined();
  });
  it("across two releases: a re-issued file is changed, a dropped one removed, the rest untouched", () => {
    const [kept, renamed, dropped] = current.names.map((n) => H.versions.find((v) => v.name === n)!);
    const oldName = renamed.name.replace(/-[0-9a-f]{8}\./, "-00000000.");
    const hist = { versions: [...H.versions, { ...renamed, name: oldName, date: "2026-11-04" }] };
    const older = { ...current, id: "2026-11-04", date: "2026-11-04", names: [kept.name, oldName, dropped.name] };
    const newer = { ...current, names: [kept.name, renamed.name] };
    const c = changesView(newer.id, hist, { releases: [older, newer] })!;
    expect(c.first).toBe(false);
    expect(c.previousId).toBe("2026-11-04");
    expect(c.added).toEqual([]);
    expect(c.changed).toHaveLength(1);
    expect(c.changed[0].from.name).toBe(oldName);
    expect(c.changed[0].to.name).toBe(renamed.name);
    expect(c.removed.map((f) => f.name)).toEqual([dropped.name]);
  });
});

describe("downloadCommand", () => {
  it("follows redirects so the apex 308 to www still works", () => {
    const c = downloadCommand("https://sfzoning.com/data/releases/x/urls.txt");
    expect(c).toContain("curl -sL '");
    expect(c).toBe("curl -sL 'https://sfzoning.com/data/releases/x/urls.txt' | wget -nc -i -");
  });
});
