/**
 * Release records: dated snapshots of the whole dataset. Pure functions, shared by the publish
 * script (Node) and the site.
 *
 * @typedef {{ series: string; name: string; sha256: string }} Extra
 * @typedef {"all" | "geojson" | "gpkg" | "shp"} ZipKind
 * @typedef {{ name: string; bytes: number; sha256: string }} ZipEntry
 * @typedef {{ id: string; date: string; names: string[]; extras: Extra[]; zips: Record<ZipKind, ZipEntry> }} Release
 * @typedef {{ releases: Release[] }} ReleaseRecord
 */

/** A release id: the date, plus -2, -3, ... for later releases on the same day. */
export const RELEASE_ID = /^\d{4}-\d{2}-\d{2}(-[2-9]|-[1-9]\d+)?$/;

const byText = (a, b) => (a < b ? -1 : a > b ? 1 : 0);

/**
 * Every current file name (files and sheets), sorted, and the README/LICENSE sidecars of the
 * files that carry them, sorted by name.
 * @returns {{ names: string[]; extras: Extra[] }}
 */
export function currentNames(manifest) {
  const names = [];
  const extras = [];
  for (const f of Object.values(manifest.files ?? {})) {
    names.push(f.name);
    for (const [key, name] of Object.entries(f.extra ?? {})) {
      extras.push({ series: f.series, name, sha256: f.extraSha256?.[key] });
    }
  }
  for (const s of Object.values(manifest.sheets ?? {})) {
    for (const i of s.items) names.push(i.name);
  }
  names.sort(byText);
  extras.sort((a, b) => byText(a.name, b.name));
  return { names, extras };
}

/** False only when the last release has exactly these names and extras (hashes included). */
export function releaseChanged(record, names, extras) {
  const last = record.releases[record.releases.length - 1];
  if (!last) return true;
  const same = (a, b, eq) => a.length === b.length && a.every((x, i) => eq(x, b[i]));
  return !(
    same(last.names, names, (x, y) => x === y) &&
    same(last.extras, extras, (x, y) => x.series === y.series && x.name === y.name && x.sha256 === y.sha256)
  );
}

/** `date`, or `date-2`, `date-3`... when earlier releases already took it. */
export function nextReleaseId(record, date) {
  const taken = new Set(record.releases.map((r) => r.id));
  if (!taken.has(date)) return date;
  for (let n = 2; ; n++) if (!taken.has(`${date}-${n}`)) return `${date}-${n}`;
}

/** @returns {Record<ZipKind, string>} */
export function zipNames(id) {
  return {
    all: `sfzoning-${id}.zip`,
    geojson: `sfzoning-${id}-geojson.zip`,
    gpkg: `sfzoning-${id}-gpkg.zip`,
    shp: `sfzoning-${id}-shp.zip`,
  };
}

export function releaseZipPath(id, name) {
  return `downloads/releases/${id}/${name}`;
}

/**
 * The file names each zip holds. `all` is every name plus every sidecar; the format zips hold the
 * names whose history entry has that format.
 * @returns {Record<ZipKind, string[]>}
 */
export function zipMembers(release, history) {
  const fmtOf = new Map(history.versions.map((v) => [v.name, v.fmt]));
  const withFmt = (fmt) => release.names.filter((n) => fmtOf.get(n) === fmt);
  return {
    all: [...release.names, ...release.extras.map((e) => e.name)],
    geojson: withFmt("geojson"),
    gpkg: withFmt("gpkg"),
    shp: withFmt("shp"),
  };
}

/** `<sha256>  <name>\n` per entry, in the order given (the format `sha256sum -c` reads). */
export function sha256Sums(entries) {
  return entries.map((e) => `${e.sha256}  ${e.name}\n`).join("");
}
