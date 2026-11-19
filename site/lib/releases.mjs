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
 * Throws when a sidecar has no hash.
 * @param {{ files?: Record<string, any>; sheets?: Record<string, any> }} manifest
 * @returns {{ names: string[]; extras: Extra[] }}
 */
export function currentNames(manifest) {
  const names = [];
  const extras = [];
  for (const [fileKey, f] of Object.entries(manifest.files ?? {})) {
    names.push(f.name);
    for (const [kind, name] of Object.entries(f.extra ?? {})) {
      const sha256 = f.extraSha256?.[kind];
      if (!sha256) throw new Error(`${fileKey}: sidecar ${kind} has no extraSha256 hash`);
      extras.push({ series: f.series, name, sha256 });
    }
  }
  for (const s of Object.values(manifest.sheets ?? {})) {
    for (const i of s.items) names.push(i.name);
  }
  names.sort(byText);
  extras.sort((a, b) => byText(a.name, b.name));
  return { names, extras };
}

/**
 * False only when the last release has exactly these names and extras (hashes included).
 * @param {ReleaseRecord} record
 * @param {string[]} names
 * @param {Extra[]} extras
 * @returns {boolean}
 */
export function releaseChanged(record, names, extras) {
  const last = record.releases[record.releases.length - 1];
  if (!last) return true;
  const same = (a, b, eq) => a.length === b.length && a.every((x, i) => eq(x, b[i]));
  return !(
    same(last.names, names, (x, y) => x === y) &&
    same(last.extras, extras, (x, y) => x.series === y.series && x.name === y.name && x.sha256 === y.sha256)
  );
}

/**
 * `date`, or `date-2`, `date-3`... when earlier releases already took it.
 * @param {ReleaseRecord} record
 * @param {string} date
 * @returns {string}
 */
export function nextReleaseId(record, date) {
  const taken = new Set(record.releases.map((r) => r.id));
  if (!taken.has(date)) return date;
  for (let n = 2; ; n++) if (!taken.has(`${date}-${n}`)) return `${date}-${n}`;
}

/**
 * @param {string} id
 * @returns {Record<ZipKind, string>}
 */
export function zipNames(id) {
  return {
    all: `sfzoning-${id}.zip`,
    geojson: `sfzoning-${id}-geojson.zip`,
    gpkg: `sfzoning-${id}-gpkg.zip`,
    shp: `sfzoning-${id}-shp.zip`,
  };
}

/**
 * @param {string} id
 * @param {string} name
 * @returns {string}
 */
export function releaseZipPath(id, name) {
  return `downloads/releases/${id}/${name}`;
}

/**
 * The file names each zip holds. `all` is every name plus every sidecar; the format zips hold the
 * names whose history entry has that format. Throws when a name has no history entry.
 * @param {Release} release
 * @param {{ versions: { name: string; fmt: string }[] }} history
 * @returns {Record<ZipKind, string[]>}
 */
export function zipMembers(release, history) {
  const fmtOf = new Map(history.versions.map((v) => [v.name, v.fmt]));
  for (const n of release.names) {
    if (!fmtOf.has(n)) throw new Error(`release ${release.id}: ${n} has no history entry`);
  }
  const withFmt = (fmt) => release.names.filter((n) => fmtOf.get(n) === fmt);
  return {
    all: [...release.names, ...release.extras.map((e) => e.name)],
    geojson: withFmt("geojson"),
    gpkg: withFmt("gpkg"),
    shp: withFmt("shp"),
  };
}

/**
 * `<sha256>  <name>\n` per entry, in the order given (the format `sha256sum -c` reads).
 * @param {{ name: string; sha256: string }[]} entries
 * @returns {string}
 */
export function sha256Sums(entries) {
  return entries.map((e) => `${e.sha256}  ${e.name}\n`).join("");
}
