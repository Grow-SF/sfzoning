import { DATA_CREDIT } from "./credits";
import { ERAS } from "./eras";
import { fileUrl, scanSeries, type Version } from "./downloadRows";
import { EXTS, latestName, naturalCompare } from "./downloadNames.mjs";
import type { Extra, Release } from "./releases.mjs";

export type Fmt = "geojson" | "gpkg" | "shp" | "cog";
export type Kind = "lots" | "districts" | "scans" | "sheets";
export type ReleaseFile = {
  series: string; era: string; layer: string; kind: Kind; fmt: Fmt;
  name: string; bytes: number; sha256: string; date: string;
};

const SITE = "https://sfzoning.com";
const abs = (series: string, name: string) => SITE + fileUrl(series, name);
const FMT_ORDER: Fmt[] = ["geojson", "gpkg", "shp", "cog"];

/** Each era's series in listing order: district series first, then its scan series (zoning, height, special use). */
const ERA_SERIES = ERAS.map((era) => ({
  id: era.id,
  series: [...new Set([era.districtSeries ?? era.id, ...scanSeries(era).map((s) => s.series)])],
}));

/** The era a series belongs to, as a district series or as one of its scan series. */
export function eraOf(series: string): string | undefined {
  return ERA_SERIES.find((e) => e.series.includes(series))?.id;
}

const kindOf = (layer: string): Kind =>
  layer === "lots" ? "lots" : layer === "districts" ? "districts" : layer === "citywide" ? "scans" : "sheets";
const KIND_ORDER: Kind[] = ["lots", "districts", "scans", "sheets"];

/** The release's files in ERAS order: per series lots, districts (geojson, gpkg, shp), then citywide, then sheets by number. */
export function releaseFiles(release: Release, history: { versions: Version[] }): ReleaseFile[] {
  const byName = new Map(history.versions.map((v) => [v.name, v]));
  const out: { f: ReleaseFile; e: number; s: number }[] = [];
  for (const name of release.names) {
    const v = byName.get(name);
    if (!v) throw new Error(`release ${release.id}: ${name} has no history entry`);
    const era = eraOf(v.series);
    if (!era) throw new Error(`release ${release.id}: ${name} is in series ${v.series}, which no era has`);
    const e = ERA_SERIES.findIndex((x) => x.id === era);
    out.push({
      e, s: ERA_SERIES[e].series.indexOf(v.series),
      f: { series: v.series, era, layer: v.layer, kind: kindOf(v.layer), fmt: v.fmt as Fmt, name, bytes: v.bytes, sha256: v.sha256, date: v.date },
    });
  }
  out.sort(
    (a, b) =>
      a.e - b.e || a.s - b.s ||
      KIND_ORDER.indexOf(a.f.kind) - KIND_ORDER.indexOf(b.f.kind) ||
      FMT_ORDER.indexOf(a.f.fmt) - FMT_ORDER.indexOf(b.f.fmt) ||
      naturalCompare(a.f.layer, b.f.layer),
  );
  return out.map((x) => x.f);
}

/** The README/LICENSE sidecars of a release, each with its era. */
export function extraFiles(release: Release): (Extra & { era: string })[] {
  return release.extras.map((x) => {
    const era = eraOf(x.series);
    if (!era) throw new Error(`release ${release.id}: ${x.name} is in series ${x.series}, which no era has`);
    return { ...x, era };
  });
}

const key = (f: ReleaseFile) => `${f.series}/${f.layer}/${f.fmt}`;

/** What differs from the previous release, matching files by series, layer and format. */
export function releaseChanges(prev: ReleaseFile[] | null, cur: ReleaseFile[]) {
  const before = new Map((prev ?? []).map((f) => [key(f), f]));
  const now = new Map(cur.map((f) => [key(f), f]));
  const added: ReleaseFile[] = [];
  const changed: { from: ReleaseFile; to: ReleaseFile }[] = [];
  for (const f of cur) {
    const was = before.get(key(f));
    if (!was) added.push(f);
    else if (was.name !== f.name) changed.push({ from: was, to: f });
  }
  const removed = (prev ?? []).filter((f) => !now.has(key(f)));
  return { added, changed, removed };
}

/** One absolute url per line: the files, then the sidecars. */
export function urlsTxt(files: ReleaseFile[], extras: Extra[]): string {
  return [...files.map((f) => abs(f.series, f.name)), ...extras.map((x) => abs(x.series, x.name))].map((u) => u + "\n").join("");
}

/** The release for scripts: every file with its versioned `url` and its `latest` link, the sidecars and the zips. */
export function releaseManifest(release: Release, files: ReleaseFile[], extras: Extra[], license: string) {
  return {
    id: release.id,
    date: release.date,
    license,
    credit: DATA_CREDIT,
    files: files.map((f) => ({
      series: f.series, layer: f.layer, fmt: f.fmt, name: f.name, bytes: f.bytes, sha256: f.sha256,
      url: abs(f.series, f.name),
      latest: abs(f.series, latestName({ series: f.series, layer: f.layer, ext: EXTS[f.fmt] })),
    })),
    extras: extras.map((x) => ({ series: x.series, name: x.name, sha256: x.sha256, url: abs(x.series, x.name) })),
    zips: release.zips,
  };
}
