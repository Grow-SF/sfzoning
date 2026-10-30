#!/usr/bin/env node
// Converts docs/research/zoning-districts.csv into public/data/rules.json for the parcel
// popup. Runs on predev/prebuild. Usage: node scripts/build-rules.mjs [csv] [out]
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

/** RFC 4180 CSV: quoted fields may hold commas, newlines and doubled quotes. */
export function parseCsv(text) {
  const rows = [];
  let row = [];
  let field = "";
  let quoted = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (quoted) {
      if (ch === '"') {
        if (text[i + 1] === '"') {
          field += '"';
          i++;
        } else quoted = false;
      } else field += ch;
    } else if (ch === '"') quoted = true;
    else if (ch === ",") {
      row.push(field);
      field = "";
    } else if (ch === "\n" || ch === "\r") {
      if (ch === "\r" && text[i + 1] === "\n") i++;
      row.push(field);
      rows.push(row);
      row = [];
      field = "";
    } else field += ch;
  }
  if (field !== "" || row.length) {
    row.push(field);
    rows.push(row);
  }
  return rows.filter((r) => r.some((f) => f.trim() !== ""));
}

const year = (v) => {
  const n = Number(String(v).trim());
  return Number.isInteger(n) ? n : null;
};

/** CSV text → array of rules (see lib/rules.ts for the schema). */
export function buildRules(csvText) {
  const [header, ...rows] = parseCsv(csvText);
  const col = (name) => {
    const i = header.indexOf(name);
    if (i < 0) throw new Error(`zoning-districts.csv: missing column ${name}`);
    return i;
  };
  const C = Object.fromEntries(
    ["era_start", "era_end", "code", "name", "category", "allowed_uses", "max_units_per_lot", "height_limit_ft", "notes", "source", "confidence"].map(
      (k) => [k, col(k)],
    ),
  );
  const out = [];
  for (const r of rows) {
    const start = year(r[C.era_start]);
    const code = (r[C.code] ?? "").trim();
    // "(boundary rule)" style rows describe how to read maps, not a district.
    if (start === null || !code || code.startsWith("(")) continue;
    const conf = (r[C.confidence] ?? "").trim().toLowerCase();
    out.push({
      era_start: start,
      era_end: year(r[C.era_end]),
      code,
      name: (r[C.name] ?? "").trim(),
      category: (r[C.category] ?? "").trim(),
      allowed_uses: (r[C.allowed_uses] ?? "").trim(),
      max_units_per_lot: (r[C.max_units_per_lot] ?? "").trim(),
      height_limit_ft: (r[C.height_limit_ft] ?? "").trim(),
      notes: (r[C.notes] ?? "").trim(),
      source: (r[C.source] ?? "").trim(),
      confidence: ["high", "medium", "low"].includes(conf) ? conf : "low",
    });
  }
  return out;
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const here = path.dirname(fileURLToPath(import.meta.url));
  const csv = path.resolve(process.argv[2] ?? path.join(here, "../../docs/research/zoning-districts.csv"));
  const out = path.resolve(process.argv[3] ?? path.join(here, "../public/data/rules.json"));
  const rules = buildRules(fs.readFileSync(csv, "utf8"));
  fs.mkdirSync(path.dirname(out), { recursive: true });
  fs.writeFileSync(out, JSON.stringify(rules) + "\n");
  console.log(`build-rules: ${rules.length} rules → ${path.relative(process.cwd(), out)}`);
}
