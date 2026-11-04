/**
 * Stable links: /data/latest/<series>/<filename> redirects to the file's current url, so a
 * link in a paper or a script keeps working when a file is republished under a new hash.
 * Plain JS because next.config.mjs imports it.
 */
const base = (url) => url.split("?")[0].split("/").at(-1);

export function latestRedirects(manifest) {
  const out = [];
  const add = (series, name, url) =>
    out.push({ source: `/data/latest/${series}/${name}`, destination: url, permanent: false });
  for (const f of Object.values(manifest.files ?? {})) {
    add(f.series, base(f.url), f.url);
    for (const [name, url] of Object.entries(f.extra ?? {})) add(f.series, name, url);
  }
  for (const [series, s] of Object.entries(manifest.sheets ?? {})) {
    for (const i of s.items ?? []) add(series, i.name, i.url);
  }
  return out;
}
