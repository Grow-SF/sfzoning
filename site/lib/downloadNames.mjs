/**
 * Names for the GIS downloads. Pure functions, shared by the publish script (Node) and the site.
 *
 *   versioned: sfzoning-<series>-<layer>-<YYYY-MM-DD>-<hash8>.<ext>   (+ optional .sha256)
 *   latest:    sfzoning-<series>-<layer>-latest.<ext>
 *
 * Layers: lots, districts, citywide, sheet-<n>, sheets (the sheet zip; latest only).
 * <hash8> is the first 8 lowercase hex digits of the file's SHA-256.
 */

export const EXTS = { geojson: "geojson.zip", gpkg: "gpkg.zip", shp: "shp.zip", cog: "tif" };

const FILE_EXTS = ["geojson.zip", "gpkg.zip", "shp.zip", "tif", "README.txt", "LICENSE.txt"];
const SHEET_EXTS = ["zip"];
const DATE = /^(\d{4})-(\d{2})-(\d{2})$/;
const HASH8 = /^[0-9a-f]{8}$/;

/** Real calendar dates only: 2026-02-30 and 2026-13-04 are rejected. */
function isDate(s) {
  const m = DATE.exec(s);
  if (!m) return false;
  const [y, mo, d] = [Number(m[1]), Number(m[2]), Number(m[3])];
  const t = new Date(Date.UTC(y, mo - 1, d));
  return t.getUTCFullYear() === y && t.getUTCMonth() === mo - 1 && t.getUTCDate() === d;
}

/** The ext a layer may carry: the sheet zip is `zip` only, and nothing else is. */
const extsFor = (layer) => (layer === "sheets" ? SHEET_EXTS : FILE_EXTS);

export function versionedName({ series, layer, date, sha256, ext }) {
  const name = `sfzoning-${series}-${layer}-${date}-${String(sha256).toLowerCase().slice(0, 8)}.${ext}`;
  if (!parseName(series, name)) throw new Error(`not a valid download name: ${name}`);
  return name;
}

export function latestName({ series, layer, ext }) {
  const name = `sfzoning-${series}-${layer}-latest.${ext}`;
  if (!parseName(series, name)) throw new Error(`not a valid download name: ${name}`);
  return name;
}

/**
 * Parse a file name for one series. Returns null for anything that is not a name this module
 * would make: wrong series, unknown layer or ext, impossible date, uppercase hash, path pieces.
 */
export function parseName(series, name) {
  if (typeof series !== "string" || typeof name !== "string") return null;
  if (!series || /[/\\]|\.\./.test(series)) return null;
  const prefix = `sfzoning-${series}-`;
  if (!name.startsWith(prefix)) return null;
  const rest = name.slice(prefix.length);
  const layerRe = "(lots|districts|citywide|sheets|sheet-(?:p\\d{2,}|\\d{2,}))";

  let m = new RegExp(`^${layerRe}-latest\\.(.+)$`).exec(rest);
  if (m) {
    const [, layer, ext] = m;
    if (!extsFor(layer).includes(ext)) return null;
    return { layer, ext, latest: true, date: null, hash8: null, checksum: false };
  }

  m = new RegExp(`^${layerRe}-(\\d{4}-\\d{2}-\\d{2})-([0-9a-f]{8})\\.(.+?)(\\.sha256)?$`).exec(rest);
  if (!m) return null;
  const [, layer, date, hash8, ext, checksum] = m;
  if (layer === "sheets") return null; // the sheet zip has no dated name
  if (!isDate(date) || !HASH8.test(hash8)) return null;
  if (!extsFor(layer).includes(ext)) return null;
  return { layer, ext, latest: false, date, hash8, checksum: checksum !== undefined };
}

export function blobPath(series, name) {
  return `downloads/${series}/${name}`;
}

const naturalParts = (s) => s.match(/\d+|\D+/g) ?? [];

/** Digit runs compare as numbers; everything else compares as text. */
export function naturalCompare(a, b) {
  const pa = naturalParts(a);
  const pb = naturalParts(b);
  for (let i = 0; i < Math.min(pa.length, pb.length); i++) {
    const x = pa[i];
    const y = pb[i];
    const xd = /^\d/.test(x);
    const yd = /^\d/.test(y);
    if (xd && yd && Number(x) !== Number(y)) return Number(x) - Number(y);
    if (!(xd && yd) && x !== y) return x < y ? -1 : 1;
  }
  if (pa.length !== pb.length) return pa.length - pb.length;
  return a < b ? -1 : a > b ? 1 : 0;
}

/**
 * Sheet number for every id. Known entries are kept as they are, including ids no longer present,
 * so a number is never reused. A new id with a -pNN suffix gets pNN; any other new id gets the next
 * two-digit number after the largest numeric one in `known` (natural ID order for the first ones).
 */
export function assignSheetNumbers(ids, known) {
  const out = { ...known };
  const fresh = [];
  for (const id of ids) {
    if (Object.hasOwn(known, id)) continue;
    const p = /-p(\d{2})$/.exec(id);
    if (p) out[id] = `p${p[1]}`;
    else fresh.push(id);
  }
  let next = 0;
  for (const v of Object.values(known)) if (/^\d+$/.test(v)) next = Math.max(next, Number(v));
  for (const id of fresh.sort(naturalCompare)) {
    next += 1;
    out[id] = String(next).padStart(2, "0");
  }
  return out;
}
