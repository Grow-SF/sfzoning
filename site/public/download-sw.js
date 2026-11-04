// Saves the sheet zips that /data builds in the page (lib/sheetZip.ts), for browsers without
// showSaveFilePicker. The page posts a stream (or a port to pull chunks from) under an id, then
// opens /data/__download/<id>; this worker answers that request with the stream as an attachment.
// It does nothing else: every other request goes to the network untouched.
const pending = new Map();

self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", (e) => e.waitUntil(self.clients.claim()));

self.addEventListener("message", (e) => {
  const { id, filename, size, stream, port, ack } = e.data || {};
  if (!id || !(stream || port)) return;
  pending.set(id, { filename, size, ack, stream: stream || fromPort(port) });
  // Forget a download nobody opened.
  setTimeout(() => pending.delete(id), 60_000);
  if (ack) ack.postMessage("ok");
});

/**
 * A stream that asks the page for one chunk each time the download wants more. The page may send
 * { error } at any time (a failed sheet, Cancel); the stream then errors so the browser's download
 * fails instead of waiting forever. A port that closes before { done } is an error too.
 */
function fromPort(port) {
  let waiting = null; // the pull in progress, if any
  let failure = null; // an error that arrived while no pull was waiting
  let finished = false;
  const finish = () => {
    finished = true;
    port.close();
  };
  port.onmessage = (e) => {
    const m = e.data || {};
    if (m.error && !waiting) {
      failure = m.error;
      return;
    }
    const w = waiting;
    waiting = null;
    if (w) w(m);
  };
  port.addEventListener("close", () => {
    if (finished) return;
    const err = { error: "The page closed before the download finished." };
    if (waiting) waiting(err);
    else failure = err.error;
  });
  return new ReadableStream({
    pull(c) {
      if (failure) {
        c.error(new Error(failure));
        finish();
        return;
      }
      return new Promise((resolve) => {
        waiting = (m) => {
          if (m.error) {
            c.error(new Error(m.error));
            finish();
          } else if (m.done) {
            c.close();
            finish();
          } else c.enqueue(m.chunk);
          resolve();
        };
        port.postMessage("pull");
      });
    },
    cancel() {
      if (!finished) port.postMessage("cancel");
      finish();
    },
  });
}

self.addEventListener("fetch", (e) => {
  const url = new URL(e.request.url);
  const m = url.origin === self.location.origin && url.pathname.match(/^\/data\/__download\/([\w-]+)$/);
  if (!m) return;
  const d = pending.get(m[1]);
  if (!d) {
    e.respondWith(new Response("This download has expired. Go back and start it again.", { status: 404 }));
    return;
  }
  pending.delete(m[1]);
  const name = String(d.filename).replace(/["\\\r\n]/g, "");
  const headers = {
    "Content-Type": "application/zip",
    "Content-Disposition": `attachment; filename="${name}"`,
    "Cache-Control": "no-store",
  };
  if (d.size) headers["Content-Length"] = String(d.size);
  e.respondWith(new Response(d.stream, { headers }));
  if (d.ack) d.ack.postMessage("started");
});
