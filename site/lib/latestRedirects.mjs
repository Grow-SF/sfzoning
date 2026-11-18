/**
 * Stable links: /data/latest/<series>/<filename> redirects to the file's current url, so a
 * link in a paper or a script keeps working when a file is republished under a new hash.
 * Served by app/data/latest/[series]/[file]/route.ts (one handler, not one redirect rule per file).
 */
export function latestRedirects(manifest, toUrl) {
  const out = [];
  const add = (series, name, url) =>
    out.push({ source: `/data/latest/${series}/${name}`, destination: url, permanent: false });
  for (const f of Object.values(manifest.files ?? {})) {
    add(f.series, f.name, toUrl(f.series, f.name));
    for (const name of Object.values(f.extra ?? {})) add(f.series, name, toUrl(f.series, name));
  }
  for (const [series, s] of Object.entries(manifest.sheets ?? {})) {
    for (const i of s.items ?? []) add(series, i.name, toUrl(series, i.name));
  }
  return out;
}

const lookups = new WeakMap();

/** The current url for /data/latest/<series>/<file>, or null when there is no such file. */
export function latestUrl(manifest, series, file, toUrl) {
  let map = lookups.get(manifest);
  if (!map) {
    map = new Map(latestRedirects(manifest, toUrl).map((r) => [r.source, r.destination]));
    lookups.set(manifest, map);
  }
  return map.get(`/data/latest/${series}/${file}`) ?? null;
}
