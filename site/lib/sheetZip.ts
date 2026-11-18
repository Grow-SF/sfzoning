// The sheet zip, served by the files route (sfzoning-<series>-sheets-latest.zip): an edition's
// sheets as a zip built while it streams, each sheet read from its versioned file on Blob.
import { makeZip, predictLength } from "client-zip";
import type { Manifest, ManifestSheet } from "./downloadRows";
import { blobPath, latestName } from "./downloadNames.mjs";

/** Just what the sheet downloads need: the license and the series' sheets. */
export type SheetSource = Pick<Manifest, "sheets" | "license" | "base">;

/** Where a sheet is read from: its place on the store (a --local manifest names its own path under the base). */
function sheetUrl(src: SheetSource, series: string, item: ManifestSheet): string {
  return item.local ? `${src.base}/${item.local}` : `${src.base}/${blobPath(series, item.name)}`;
}
export type ZipEntry = { name: string; input: string | ReadableStream<Uint8Array>; lastModified: Date; size?: number };
type FetchFn = (url: string, init?: RequestInit) => Promise<Response>;
type Options = { signal?: AbortSignal; onSheetDone?: (done: number) => void };

const day = (d: string) => new Date(`${d}T12:00:00Z`);
const message = (e: unknown) => String((e as Error)?.message ?? e);

function license(src: SheetSource): string {
  if (!src.license) throw new Error("The license text is missing, so the zip cannot be built.");
  return src.license;
}

/** A sheet's body, failing if it is not exactly the size the manifest says. */
function sized(body: ReadableStream<Uint8Array>, item: ManifestSheet): ReadableStream<Uint8Array> {
  let n = 0;
  const wrong = (size: string) => new Error(`Could not download sheet ${item.name}: size ${size}, expected ${item.bytes}`);
  return body.pipeThrough(
    new TransformStream<Uint8Array, Uint8Array>({
      transform(chunk, c) {
        n += chunk.length;
        if (n > item.bytes) throw wrong(`at least ${n}`);
        c.enqueue(chunk);
      },
      flush() {
        if (n !== item.bytes) throw wrong(String(n));
      },
    }),
  );
}

/** README.txt and LICENSE.txt, then each sheet, fetched only when the zip asks for it. */
export async function* zipEntries(
  src: SheetSource,
  series: string,
  fetchFn: FetchFn = fetch,
  { signal, onSheetDone }: Options = {},
): AsyncGenerator<ZipEntry> {
  const s = src.sheets[series];
  if (!s) throw new Error(`No sheets for ${series}`);
  const text = license(src);
  yield { name: "README.txt", input: s.readme, lastModified: day(s.updated) };
  yield { name: "LICENSE.txt", input: text, lastModified: day(s.updated) };
  let done = 0;
  for (const item of s.items) {
    let res: Response;
    try {
      res = await fetchFn(sheetUrl(src, series, item), { signal });
    } catch (e) {
      if (signal?.aborted) throw e;
      throw new Error(`Could not download sheet ${item.name}: ${message(e)}`);
    }
    if (!res.ok) throw new Error(`Could not download sheet ${item.name}: HTTP ${res.status}`);
    // The zip's Content-Length is predicted from the manifest, so a sheet of another size must stop
    // the download rather than end in a broken or truncated file. Only the decoded bytes are
    // counted: Blob may send the sheet compressed (br), so its Content-Length can be the
    // compressed size, and cross-origin the Content-Encoding header that would say so is hidden.
    if (!res.body) throw new Error(`Could not download sheet ${item.name}: empty response`);
    yield { name: item.name, input: sized(res.body, item), lastModified: day(item.updated), size: item.bytes };
    // The generator resumes once client-zip has read the whole sheet.
    onSheetDone?.(++done);
  }
}

/** The exact size of the zip, so the download can show a total and send a Content-Length. */
export function zipLength(src: SheetSource, series: string): number {
  const s = src.sheets[series];
  const enc = new TextEncoder();
  return Number(
    predictLength([
      { name: "README.txt", size: enc.encode(s.readme).length },
      { name: "LICENSE.txt", size: enc.encode(license(src)).length },
      ...s.items.map((i) => ({ name: i.name, size: i.bytes })),
    ]),
  );
}

const missing = () => new Response("No such sheets.", { status: 404 });

/** `attachment` with an ASCII filename and, for other characters, its RFC 5987 form. */
function attachment(name: string): string {
  const ascii = name.replace(/[^\x20-\x7e]|["\\]/g, "_");
  return ascii === name
    ? `attachment; filename="${name}"`
    : `attachment; filename="${ascii}"; filename*=UTF-8''${encodeURIComponent(name)}`;
}

/**
 * The series' sheets as one zip, built while it streams (README, LICENSE, then each sheet fetched
 * from Blob in turn). Its exact length is known up front, so the browser shows a normal download
 * with progress. A sheet that fails or has the wrong size errors the stream, so the browser marks
 * the download failed instead of saving a short file.
 */
export function sheetsZipResponse(src: SheetSource, series: string, fetchFn: FetchFn = fetch, signal?: AbortSignal): Response {
  if (!src.sheets[series] || !src.license) return missing();
  return new Response(makeZip(zipEntries(src, series, fetchFn, { signal })), {
    headers: {
      "content-type": "application/zip",
      "content-disposition": attachment(latestName({ series, layer: "sheets", ext: "zip" })),
      "content-length": String(zipLength(src, series)),
      "cache-control": "no-store",
    },
  });
}
