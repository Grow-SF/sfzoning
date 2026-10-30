// Bundlers mangle maplibre-gl 6's module worker; serve it as static files instead.
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const src = path.join(root, "node_modules/maplibre-gl/dist");
const dest = path.join(root, "public/maplibre");
fs.mkdirSync(dest, { recursive: true });
for (const f of ["maplibre-gl-worker.mjs", "maplibre-gl-shared.mjs"]) fs.copyFileSync(path.join(src, f), path.join(dest, f));
