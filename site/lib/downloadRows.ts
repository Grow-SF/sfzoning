import { ERAS, type Era } from "./eras";
import { latestRedirects } from "./latestRedirects.mjs";

export { latestRedirects };

export type ManifestFile = {
  series: string; layer: string; fmt: string; url: string; bytes: number; sha256: string;
  fingerprint: string; updated: string; extra?: Record<string, string>;
};
export type ManifestSheet = { sheet: string; name: string; url: string; bytes: number; md5: string; updated: string };
export type Manifest = {
  files: Record<string, ManifestFile>;
  sheets: Record<string, { updated: string; readme: string; items: ManifestSheet[] }>;
  superseded: unknown[];
  license?: string;
  generated?: string;
  base?: string;
  local?: boolean;
};

export type Cell = { fmt: string; label: "GeoJSON" | "GeoPackage" | "Shapefile"; url: string; bytes: number; updated: string };
export type ScanCell = {
  series: string;
  label: "Zoning" | "Height" | "Special use" | "";
  citywide?: Omit<Cell, "label">;
  sheets?: { series: string; count: number; bytes: number; updated: string };
};
export type Row = { era: Era; lots: Cell[]; districts: Cell[]; scans: ScanCell[] };

const VECTOR_FORMATS: [string, Cell["label"]][] = [
  ["geojson", "GeoJSON"],
  ["gpkg", "GeoPackage"],
  ["shp", "Shapefile"],
];

function vectorCells(m: Manifest, series: string, layer: string): Cell[] {
  const cells: Cell[] = [];
  for (const [fmt, label] of VECTOR_FORMATS) {
    const f = m.files[`${series}/${layer}.${fmt}`];
    if (f) cells.push({ fmt, label, url: f.url, bytes: f.bytes, updated: f.updated });
  }
  return cells;
}

function scanSeries(era: Era): { series: string; label: ScanCell["label"] }[] {
  return era.scanSeries.flatMap<{ series: string; label: ScanCell["label"] }>((s) =>
    s.endsWith("-zoning")
      ? [
          { series: s, label: "Zoning" as const },
          { series: s.replace(/-zoning$/, "-height"), label: "Height" as const },
          { series: s.replace(/-zoning$/, "-su"), label: "Special use" as const },
        ]
      : [{ series: s, label: "" as const }],
  );
}

function scanCell(m: Manifest, series: string, label: ScanCell["label"]): ScanCell {
  const cell: ScanCell = { series, label };
  const c = m.files[`${series}/citywide.cog`];
  if (c) cell.citywide = { fmt: "cog", url: c.url, bytes: c.bytes, updated: c.updated };
  const s = m.sheets[series];
  if (s && s.items.length) {
    cell.sheets = {
      series,
      count: s.items.length,
      bytes: s.items.reduce((n, i) => n + i.bytes, 0),
      updated: s.updated,
    };
  }
  return cell;
}

/** One table row per era, in ERAS order; anything not published is an empty cell. */
export function rows(m: Manifest, eras: Era[] = ERAS): Row[] {
  return eras.map((era) => {
    const vs = era.districtSeries ?? era.id;
    return {
      era,
      lots: vectorCells(m, vs, "lots"),
      districts: vectorCells(m, vs, "districts"),
      scans: scanSeries(era).map((s) => scanCell(m, s.series, s.label)),
    };
  });
}

/** "3.2 MB", "0.7 GB" (from 500 MB up): decimal units, one decimal place. */
export function formatBytes(n: number): string {
  if (n >= 5e8) return `${(n / 1e9).toFixed(1)} GB`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)} MB`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)} kB`;
  return `${n} B`;
}

/** The latest `updated` date among files and sheets ("" when nothing is published). */
export function lastUpdated(m: Manifest): string {
  const dates = [
    ...Object.values(m.files).map((f) => f.updated),
    ...Object.values(m.sheets).map((s) => s.updated),
  ];
  return dates.sort().at(-1) ?? "";
}

const csvField = (v: string | number) => (/[",\n]/.test(String(v)) ? `"${String(v).replace(/"/g, '""')}"` : String(v));

/** The per-sheet index of a series as CSV. */
export function sheetsCsv(m: Manifest, series: string): string {
  const header = "sheet,name,url,bytes,md5,updated";
  const lines = (m.sheets[series]?.items ?? []).map((i) =>
    [i.sheet, i.name, i.url, i.bytes, i.md5, i.updated].map(csvField).join(","),
  );
  return [header, ...lines].join("\n");
}
