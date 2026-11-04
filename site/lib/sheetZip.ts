// Sheet downloads built in the browser: one sheet as a file, or an edition's sheets as a zip that
// streams straight to disk (nothing is stored on the server and the zip is never held in memory).
import { makeZip, predictLength } from "client-zip";
import type { Manifest, ManifestSheet } from "./downloadRows";

/** Just what the sheet downloads need: the license and the series' sheets. */
export type SheetSource = Pick<Manifest, "sheets" | "license">;
export type ZipEntry = { name: string; input: string | ReadableStream<Uint8Array>; lastModified: Date; size?: number };
export type Progress = { sheetsDone: number; sheets: number; bytes: number; totalBytes: number };
/** Where the zip went: a file the reader picked, or the browser's own downloads (service worker). */
export type Saved = "file" | "browser";
type FetchFn = (url: string, init?: RequestInit) => Promise<Response>;
type Options = { signal?: AbortSignal; onSheetDone?: (done: number) => void };

const day = (d: string) => new Date(`${d}T12:00:00Z`);
const TIMEOUT_MS = 30_000;
const abortError = (s?: AbortSignal) => s?.reason ?? new DOMException("Aborted", "AbortError");
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
      res = await fetchFn(item.url, { signal });
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

/** `p`, unless the signal aborts or `ms` pass first (then rejects with `timeoutMessage`). */
function within<T>(p: Promise<T>, ms: number, timeoutMessage: string, signal?: AbortSignal): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    if (signal?.aborted) return reject(abortError(signal));
    const onAbort = () => reject(abortError(signal));
    const t = setTimeout(() => reject(new Error(timeoutMessage)), ms);
    signal?.addEventListener("abort", onAbort, { once: true });
    p.then(resolve, reject).finally(() => {
      clearTimeout(t);
      signal?.removeEventListener("abort", onAbort);
    });
  });
}

type SavePicker = (o: { suggestedName: string; types?: unknown[] }) => Promise<{
  createWritable(): Promise<WritableStream<Uint8Array>>;
}>;

/**
 * Builds the zip and saves it: into a file the reader picks where the browser can (Chrome, Edge),
 * otherwise through /download-sw.js, which answers a download URL with the stream (Firefox, Safari).
 * Resolves when the last byte is written (or handed to the browser); rejects with the failing
 * sheet's name, or an AbortError.
 */
export async function saveSheetsZip(
  src: SheetSource,
  series: string,
  { signal, onProgress }: { signal?: AbortSignal; onProgress?: (p: Progress) => void } = {},
): Promise<Saved> {
  const filename = `${series}-sheets.zip`;
  const totalBytes = zipLength(src, series);
  const progress: Progress = { sheetsDone: 0, sheets: src.sheets[series].items.length, bytes: 0, totalBytes };
  const report = () => onProgress?.({ ...progress });

  const picker = (window as unknown as { showSaveFilePicker?: SavePicker }).showSaveFilePicker;
  // Ask for the file first: the picker needs the click's user activation, and a cancelled picker
  // should not start any fetches.
  const handle = picker ? await picker({ suggestedName: filename }) : null;
  signal?.throwIfAborted();

  const zip = makeZip(
    zipEntries(src, series, fetch, { signal, onSheetDone: (n) => ((progress.sheetsDone = n), report()) }),
  ).pipeThrough(
    new TransformStream<Uint8Array, Uint8Array>({
      transform(chunk, c) {
        progress.bytes += chunk.length;
        report();
        c.enqueue(chunk);
      },
    }),
  );
  report();

  if (handle) {
    await zip.pipeTo(await handle.createWritable(), { signal });
    return "file";
  }
  await streamThroughWorker(zip, filename, totalBytes, signal);
  return "browser";
}

/**
 * Hands the stream to the service worker and opens its download URL in a hidden iframe.
 * Any failure (a sheet, Cancel, the browser cancelling) is passed to the worker as an error so the
 * browser's download fails too, instead of hanging or ending as a short "complete" file.
 */
async function streamThroughWorker(
  zip: ReadableStream<Uint8Array>,
  filename: string,
  size: number,
  signal?: AbortSignal,
): Promise<void> {
  if (!("serviceWorker" in navigator)) throw new Error("This browser cannot save the zip; download the sheets one by one.");
  const reg = await within(
    navigator.serviceWorker.register("/download-sw.js", { scope: "/data/" }),
    TIMEOUT_MS, "The download helper did not start. Reload the page and try again.", signal,
  );
  const worker = await within(activeWorker(reg), TIMEOUT_MS, "The download helper did not start. Reload the page and try again.", signal);

  const reader = zip.getReader();
  let settle!: { resolve: () => void; reject: (e: unknown) => void };
  const finished = new Promise<void>((resolve, reject) => (settle = { resolve, reject }));
  let controller: ReadableStreamDefaultController<Uint8Array> | null = null; // transferred stream
  let port: MessagePort | null = null; // chunk-by-chunk fallback
  let failed = false;
  const fail = (e: unknown) => {
    if (failed) return;
    failed = true;
    settle.reject(e);
    reader.cancel(e).catch(() => {});
    try {
      controller?.error(e);
    } catch {
      // already closed or errored
    }
    // The worker errors its stream and closes the port itself.
    port?.postMessage({ error: message(e) });
  };
  const onAbort = () => fail(abortError(signal));

  // Pulls one chunk at a time, so the network waits for the disk rather than filling memory.
  const pull = async (): Promise<Uint8Array | null> => {
    if (failed) throw new Error("The download stopped.");
    try {
      const { done, value } = await reader.read();
      if (failed) throw new Error("The download stopped.");
      if (done) settle.resolve();
      return done ? null : value;
    } catch (e) {
      fail(e);
      throw e;
    }
  };

  // The worker answers "ok" once it holds the stream and "started" once the browser asked for it.
  const status = new MessageChannel();
  let onOk!: () => void, onStarted!: () => void;
  const ok = new Promise<void>((r) => (onOk = r));
  const started = new Promise<void>((r) => (onStarted = r));
  status.port1.onmessage = (e) => (e.data === "ok" ? onOk() : e.data === "started" ? onStarted() : undefined);

  const id = crypto.randomUUID();
  const meta = { id, filename, size, ack: status.port2 };
  try {
    const stream = new ReadableStream<Uint8Array>({
      start(c) {
        controller = c;
      },
      async pull(c) {
        const chunk = await pull();
        if (chunk) c.enqueue(chunk);
        else c.close();
      },
      cancel() {
        fail(new Error("The download was cancelled in the browser."));
      },
    });
    worker.postMessage({ ...meta, stream }, [stream, status.port2]);
  } catch (e) {
    if (!(e instanceof DOMException && e.name === "DataCloneError")) throw e;
    // This browser cannot transfer streams: send chunks over a port as the worker asks for them.
    controller = null;
    const ch = new MessageChannel();
    port = ch.port1;
    port.onmessage = async (m) => {
      if (m.data === "cancel") return fail(new Error("The download was cancelled in the browser."));
      if (failed) return;
      try {
        const chunk = await pull();
        // Copied, not transferred: a chunk may be a view into a buffer client-zip still uses.
        port!.postMessage(chunk ? { chunk } : { done: true });
      } catch {
        // fail() has already sent the error.
      }
    };
    worker.postMessage({ ...meta, port: ch.port2 }, [ch.port2, status.port2]);
  }

  signal?.addEventListener("abort", onAbort, { once: true });
  let frame: HTMLIFrameElement | null = null;
  // Browsers stop an idle worker after a while; a message now and then keeps this one saving.
  const keepAlive = setInterval(() => worker.postMessage({ ping: true }), 10_000);
  try {
    await within(ok, TIMEOUT_MS, "The download helper did not answer. Reload the page and try again.", signal);
    frame = document.createElement("iframe");
    frame.hidden = true;
    frame.src = `/data/__download/${id}`;
    document.body.appendChild(frame);
    await within(started, TIMEOUT_MS, "The browser did not start the download. Reload the page and try again.", signal);
    await finished;
  } catch (e) {
    fail(e);
    throw e;
  } finally {
    clearInterval(keepAlive);
    signal?.removeEventListener("abort", onAbort);
    status.port1.close();
    // Leave the iframe a moment so the browser can finish handing the file to its downloads.
    const f = frame;
    if (f) setTimeout(() => f.remove(), 60_000);
  }
}

function activeWorker(reg: ServiceWorkerRegistration): Promise<ServiceWorker> {
  if (reg.active) return Promise.resolve(reg.active);
  const w = (reg.installing ?? reg.waiting)!;
  return new Promise((resolve, reject) => {
    w.addEventListener("statechange", () => {
      if (w.state === "activated") resolve(w);
      else if (w.state === "redundant") reject(new Error("The download helper failed to start."));
    });
  });
}

/** One sheet: fetched into memory (≤ ~60 MB) and saved under its own name. */
export async function saveSheet(item: ManifestSheet, fetchFn: FetchFn = fetch): Promise<void> {
  const res = await fetchFn(item.url);
  if (!res.ok) throw new Error(`Could not download sheet ${item.name}: HTTP ${res.status}`);
  const url = URL.createObjectURL(await res.blob());
  const a = document.createElement("a");
  a.href = url;
  a.download = item.name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 60_000);
}
