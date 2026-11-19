// Handlers for the release bulk routes, so tests can call them without Next.
import { citations } from "./citation.mjs";
import history from "./download-versions.json";
import manifest from "./downloads.json";
import { findRelease } from "./releaseData";
import { matches, parseFilter, type Filter } from "./releaseFilter";
import { extraFiles, releaseFiles, releaseManifest, urlsTxt } from "./releaseFiles";
import { RELEASE_ID, releaseZipPath, sha256Sums, zipNames } from "./releases.mjs";
import type { Version } from "./downloadRows";

export type TextKind = "urls" | "sums" | "manifest" | "cff";

const FILE: Record<TextKind, string> = { urls: "urls.txt", sums: "SHA256SUMS", manifest: "manifest.json", cff: "CITATION.cff" };
const TEXT = "text/plain; charset=utf-8";
const notFound = () => new Response("Not found", { status: 404 });

/** The text output `kind` of release `id`, narrowed by the query's filter (`cff` ignores it). `latest` redirects to the newest id. */
export function releaseTextResponse(kind: TextKind, id: string, url: URL): Response {
  if (id !== "latest" && !RELEASE_ID.test(id)) return notFound();
  const release = findRelease(id);
  if (!release) return notFound();
  if (id === "latest") {
    return new Response(null, {
      status: 307,
      headers: { location: `/data/releases/${release.id}/${FILE[kind]}${url.search}`, "cache-control": "no-store" },
    });
  }
  if (kind === "cff") return new Response(citations(release).cff, { headers: { "content-type": TEXT } });

  const parsed = parseFilter(url.searchParams);
  if ("error" in parsed) return new Response(parsed.error, { status: 400, headers: { "content-type": TEXT } });
  const f: Filter = parsed.filter;

  const files = releaseFiles(release, history as unknown as { versions: Version[] }).filter((x) => matches(x, f));
  const extras = extraFiles(release).filter((x) => matches({ fmt: "cog", kind: "scans", era: x.era }, f));
  if (kind === "urls") return new Response(urlsTxt(files, extras), { headers: { "content-type": TEXT } });
  if (kind === "sums") return new Response(sha256Sums([...files, ...extras]), { headers: { "content-type": TEXT } });
  return Response.json(releaseManifest(release, files, extras, (manifest as { license: string }).license));
}

/** A release zip redirects to its place on Blob; a name that is not one of the release's zips is a 404. */
export function releaseZipResponse(id: string, file: string): Response {
  if (!RELEASE_ID.test(id) || !findRelease(id)) return notFound();
  const { base } = manifest as { base?: string };
  if (!base || !Object.values(zipNames(id)).includes(file)) return notFound();
  return new Response(null, {
    status: 308,
    headers: { location: `${base}/${releaseZipPath(id, file)}?download=1`, "cache-control": "public, max-age=31536000, immutable" },
  });
}
