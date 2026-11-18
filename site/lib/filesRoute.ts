// The one files route, /data/files/<series>/<name>: versioned names redirect to their place on
// Blob, `latest` names redirect to the current versioned name (or stream the sheet zip).
import { EXTS, blobPath, parseName } from "./downloadNames.mjs";
import { fileUrl, type Manifest } from "./downloadRows";
import { sheetsZipResponse } from "./sheetZip";

type FetchFn = (url: string, init?: RequestInit) => Promise<Response>;

const notFound = () => new Response("Not found", { status: 404 });
const redirect = (status: 307 | 308, location: string, cache: string) =>
  new Response(null, { status, headers: { location, "cache-control": cache } });

/** A current file of the manifest, under the layer and ext a latest name would use. */
type Current = { layer: string; ext: string; name: string; local?: string };

function currents(m: Manifest, series: string): Current[] {
  const out: Current[] = [];
  for (const f of Object.values(m.files)) {
    if (f.series !== series) continue;
    out.push({ layer: f.layer, ext: EXTS[f.fmt as keyof typeof EXTS], name: f.name, local: f.local && `downloads/${f.local}` });
    for (const [kind, name] of Object.entries(f.extra ?? {})) {
      const rel = f.extraLocal?.[kind];
      out.push({ layer: f.layer, ext: kind, name, local: rel && `downloads/${rel}` });
    }
  }
  for (const i of m.sheets[series]?.items ?? []) out.push({ layer: `sheet-${i.number}`, ext: "tif", name: i.name, local: i.local });
  return out;
}

/**
 * The response for /data/files/<series>/<name>. Only a series in the manifest and a name that
 * parseName accepts get any further; everything else is a 404 before anything is fetched.
 */
export async function filesResponse(
  manifest: Manifest,
  series: string,
  name: string,
  fetchFn: FetchFn = fetch,
  signal?: AbortSignal,
): Promise<Response> {
  const known = manifest.sheets[series] !== undefined || Object.values(manifest.files).some((f) => f.series === series);
  if (!known) return notFound();
  const p = parseName(series, name) as { layer: string; ext: string; latest: boolean } | null;
  if (!p) return notFound();
  const { base } = manifest;

  if (manifest.local) {
    // A local manifest names only what it has, at the path it can be read from.
    const hit = currents(manifest, series).find((c) => (p.latest ? c.layer === p.layer && c.ext === p.ext : c.name === name));
    if (!base || !hit?.local) return notFound();
    return redirect(307, `${base}/${hit.local}`, "no-store");
  }

  if (p.latest) {
    if (p.layer === "sheets") return sheetsZipResponse(manifest, series, fetchFn, signal);
    const hits = currents(manifest, series).filter((c) => c.layer === p.layer && c.ext === p.ext);
    // A sidecar's latest name is ambiguous when one layer has several formats.
    if (hits.length !== 1) return notFound();
    return redirect(307, fileUrl(series, hits[0].name), "no-store");
  }
  // Older versions stay valid, so a versioned name needs no entry in the current manifest.
  if (!base) return notFound();
  return redirect(308, `${base}/${blobPath(series, name)}`, "public, max-age=31536000, immutable");
}
