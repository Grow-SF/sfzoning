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
  pending.set(id, { filename, size, stream: stream || fromPort(port) });
  // Forget a download nobody opened.
  setTimeout(() => pending.delete(id), 60_000);
  if (ack) ack.postMessage("ok");
});

/** A stream that asks the page for one chunk each time the download wants more. */
function fromPort(port) {
  let waiting = null;
  port.onmessage = (e) => {
    const w = waiting;
    waiting = null;
    if (w) w(e.data);
  };
  return new ReadableStream({
    pull(c) {
      return new Promise((resolve) => {
        waiting = (m) => {
          if (m.error) c.error(new Error(m.error));
          else if (m.done) (c.close(), port.close());
          else c.enqueue(m.chunk);
          resolve();
        };
        port.postMessage("pull");
      });
    },
    cancel() {
      port.close();
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
});
