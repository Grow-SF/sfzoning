// Sheet downloads built in the browser: one sheet as a file, or an edition's sheets as a zip that
// streams straight to disk (nothing is stored on the server and the zip is never held in memory).
import { makeZip, predictLength } from "client-zip";
import type { Manifest, ManifestSheet } from "./downloadRows";

/** Just what the sheet downloads need: the license and the series' sheets. */
export type SheetSource = Pick<Manifest, "sheets" | "license">;
export type ZipEntry = { name: string; input: string | Response; lastModified: Date; size?: number };
export type Progress = { sheetsDone: number; sheets: number; bytes: number; totalBytes: number };
type FetchFn = (url: string, init?: RequestInit) => Promise<Response>;
type Options = { signal?: AbortSignal; onSheetDone?: (done: number) => void };

const day = (d: string) => new Date(`${d}T12:00:00Z`);

/** README.txt and LICENSE.txt, then each sheet, fetched only when the zip asks for it. */
export async function* zipEntries(
  src: SheetSource,
  series: string,
  fetchFn: FetchFn = fetch,
  { signal, onSheetDone }: Options = {},
): AsyncGenerator<ZipEntry> {
  const s = src.sheets[series];
  if (!s) throw new Error(`No sheets for ${series}`);
  yield { name: "README.txt", input: s.readme, lastModified: day(s.updated) };
  yield { name: "LICENSE.txt", input: src.license ?? "", lastModified: day(s.updated) };
  let done = 0;
  for (const item of s.items) {
    let res: Response;
    try {
      res = await fetchFn(item.url, { signal });
    } catch (e) {
      if (signal?.aborted) throw e;
      throw new Error(`Could not download sheet ${item.name}: ${(e as Error).message}`);
    }
    if (!res.ok) throw new Error(`Could not download sheet ${item.name}: HTTP ${res.status}`);
    yield { name: item.name, input: res, lastModified: day(item.updated), size: item.bytes };
    // The generator resumes once client-zip has read the whole sheet.
    onSheetDone?.(++done);
  }
}

/** The exact size of the zip, so the download can show a total. */
export function zipLength(src: SheetSource, series: string): number {
  const s = src.sheets[series];
  const enc = new TextEncoder();
  return Number(
    predictLength([
      { name: "README.txt", size: enc.encode(s.readme).length },
      { name: "LICENSE.txt", size: enc.encode(src.license ?? "").length },
      ...s.items.map((i) => ({ name: i.name, size: i.bytes })),
    ]),
  );
}

type SavePicker = (o: { suggestedName: string; types?: unknown[] }) => Promise<{
  createWritable(): Promise<WritableStream<Uint8Array>>;
}>;

/**
 * Builds the zip and saves it: into a file the reader picks where the browser can (Chrome, Edge),
 * otherwise through /download-sw.js, which answers a download URL with the stream (Firefox, Safari).
 * Resolves when the last byte is written; rejects with the failing sheet's name, or an AbortError.
 */
export async function saveSheetsZip(
  src: SheetSource,
  series: string,
  { signal, onProgress }: { signal?: AbortSignal; onProgress?: (p: Progress) => void } = {},
): Promise<void> {
  const filename = `${series}-sheets.zip`;
  const totalBytes = zipLength(src, series);
  const progress: Progress = { sheetsDone: 0, sheets: src.sheets[series].items.length, bytes: 0, totalBytes };
  const report = () => onProgress?.({ ...progress });

  const picker = (window as unknown as { showSaveFilePicker?: SavePicker }).showSaveFilePicker;
  // Ask for the file first: the picker needs the click's user activation, and a cancelled picker
  // should not start any fetches.
  const handle = picker ? await picker({ suggestedName: filename }) : null;

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
    return;
  }
  await streamThroughWorker(zip, filename, totalBytes, signal);
}

/** Hands the stream to the service worker and opens its download URL in a hidden iframe. */
async function streamThroughWorker(
  zip: ReadableStream<Uint8Array>,
  filename: string,
  size: number,
  signal?: AbortSignal,
): Promise<void> {
  if (!("serviceWorker" in navigator)) throw new Error("This browser cannot save the zip; download the sheets one by one.");
  const worker = await activeWorker(await navigator.serviceWorker.register("/download-sw.js", { scope: "/data/" }));
  const reader = zip.getReader();
  let settle!: { resolve: () => void; reject: (e: unknown) => void };
  const finished = new Promise<void>((resolve, reject) => (settle = { resolve, reject }));
  const onAbort = () => {
    reader.cancel(signal?.reason).catch(() => {});
    settle.reject(signal?.reason ?? new DOMException("Aborted", "AbortError"));
  };
  signal?.addEventListener("abort", onAbort, { once: true });

  // Pulls one chunk at a time, so the network waits for the disk rather than filling memory.
  const pull = async (): Promise<Uint8Array | null> => {
    try {
      // A cancelled download must fail in the browser, not end as a short "complete" file.
      if (signal?.aborted) throw signal.reason ?? new DOMException("Aborted", "AbortError");
      const { done, value } = await reader.read();
      if (signal?.aborted) throw signal.reason ?? new DOMException("Aborted", "AbortError");
      if (done) settle.resolve();
      return done ? null : value;
    } catch (e) {
      settle.reject(e);
      throw e;
    }
  };

  const id = crypto.randomUUID();
  const port = await sendStream(worker, id, filename, size, pull);
  const frame = document.createElement("iframe");
  frame.hidden = true;
  frame.src = `/data/__download/${id}`;
  document.body.appendChild(frame);
  // Browsers stop an idle worker after a while; a message now and then keeps this one saving.
  const keepAlive = setInterval(() => worker.postMessage({ ping: true }), 10_000);
  try {
    await finished;
  } finally {
    clearInterval(keepAlive);
    signal?.removeEventListener("abort", onAbort);
    port?.close();
    // Leave the iframe a moment so the browser can finish handing the file to its downloads.
    setTimeout(() => frame.remove(), 60_000);
  }
}

/**
 * Gives the worker the stream: transferred whole where the browser can transfer streams,
 * otherwise chunk by chunk over a MessagePort as the worker asks for them.
 */
async function sendStream(
  worker: ServiceWorker,
  id: string,
  filename: string,
  size: number,
  pull: () => Promise<Uint8Array | null>,
): Promise<MessagePort | null> {
  const ack = new MessageChannel();
  const acked = new Promise<void>((resolve) => (ack.port1.onmessage = () => resolve()));
  try {
    const stream = new ReadableStream<Uint8Array>({
      async pull(c) {
        const chunk = await pull();
        if (chunk) c.enqueue(chunk);
        else c.close();
      },
    });
    worker.postMessage({ id, filename, size, stream, ack: ack.port2 }, [stream, ack.port2]);
    await acked;
    return null;
  } catch (e) {
    if (!(e instanceof DOMException && e.name === "DataCloneError")) throw e;
  }
  const ch = new MessageChannel();
  ch.port1.onmessage = async () => {
    try {
      const chunk = await pull();
      // Copied, not transferred: a chunk may be a view into a buffer client-zip still uses.
      if (chunk) ch.port1.postMessage({ chunk });
      else ch.port1.postMessage({ done: true });
    } catch (e) {
      ch.port1.postMessage({ error: String((e as Error)?.message ?? e) });
    }
  };
  const ack2 = new MessageChannel();
  const acked2 = new Promise<void>((resolve) => (ack2.port1.onmessage = () => resolve()));
  worker.postMessage({ id, filename, size, port: ch.port2, ack: ack2.port2 }, [ch.port2, ack2.port2]);
  await acked2;
  return ch.port1;
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
