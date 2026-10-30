#!/usr/bin/env node
// Static tile server with HTTP Range + CORS, for local development.
// Serves the repo's pipeline output (../pipeline/data/out from site/) by default.
// TILES_DIR overrides the directory; PORT overrides 8787; HOST overrides 127.0.0.1 (0.0.0.0 for LAN).
import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const DEFAULT_TILES_DIR = fileURLToPath(new URL("../../pipeline/data/out", import.meta.url));
const DIR = path.resolve(process.env.TILES_DIR || DEFAULT_TILES_DIR);
const PORT = Number(process.env.PORT || 8787);
const HOST = process.env.HOST || "127.0.0.1";

const CORS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "Range, If-Match, If-None-Match",
  "Access-Control-Expose-Headers": "Content-Range, Content-Length, ETag, Accept-Ranges",
};

const server = http.createServer((req, res) => {
  if (req.method === "OPTIONS") {
    res.writeHead(204, { ...CORS, "Access-Control-Allow-Methods": "GET, HEAD, OPTIONS" });
    return res.end();
  }
  let rel;
  try {
    rel = decodeURIComponent(new URL(req.url, "http://x").pathname);
  } catch {
    res.writeHead(400, CORS);
    return res.end();
  }
  const file = path.resolve(DIR, "." + rel);
  if (file !== DIR && !file.startsWith(DIR + path.sep)) {
    res.writeHead(403, CORS);
    return res.end("forbidden");
  }
  fs.stat(file, (err, st) => {
    if (err || !st.isFile()) {
      res.writeHead(404, CORS);
      return res.end("not found");
    }
    const headers = { ...CORS, "Accept-Ranges": "bytes", "Content-Type": "application/octet-stream" };
    const m = /^bytes=(\d*)-(\d*)$/.exec(req.headers.range || "");
    let start = 0;
    let end = st.size - 1;
    let status = 200;
    if (m && (m[1] || m[2])) {
      if (m[1]) {
        start = Number(m[1]);
        if (m[2]) end = Math.min(Number(m[2]), end);
      } else {
        start = Math.max(0, st.size - Number(m[2]));
      }
      if (start > end || start >= st.size) {
        res.writeHead(416, { ...CORS, "Content-Range": `bytes */${st.size}` });
        return res.end();
      }
      status = 206;
      headers["Content-Range"] = `bytes ${start}-${end}/${st.size}`;
    }
    headers["Content-Length"] = end - start + 1;
    res.writeHead(status, headers);
    if (req.method === "HEAD") return res.end();
    fs.createReadStream(file, { start, end }).pipe(res);
  });
});

server.listen(PORT, HOST, () => console.log(`serving ${DIR} on http://${HOST}:${PORT}`));
