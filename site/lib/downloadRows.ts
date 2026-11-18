import { DATA_CREDIT, DATA_LICENSE, RUMSEY_CREDIT, RUMSEY_LICENSE } from "./credits";
import { ERAS, type Era } from "./eras";
import { EXTS, latestName } from "./downloadNames.mjs";

/** The site path of a published file; the files route redirects it to Blob. The one definition of that path. */
export const fileUrl = (series: string, name: string): string => `/data/files/${series}/${name}`;


export type ManifestFile = {
  series: string; layer: string; fmt: string; name: string; bytes: number; sha256: string;
  fingerprint: string; updated: string; extra?: Record<"README.txt" | "LICENSE.txt", string>;
  /** --local manifests only: the path under the base the file can be read from. */
  local?: string; extraLocal?: Record<string, string>;
};
export type ManifestSheet = {
  sheet: string; number: string; name: string; bytes: number; md5: string; sha256: string; updated: string; local?: string;
};
export type Manifest = {
  files: Record<string, ManifestFile>;
  sheets: Record<string, { updated: string; readme: string; items: ManifestSheet[] }>;
  license?: string;
  generated?: string;
  base?: string;
  local?: boolean;
};
export type Version = { series: string; layer: string; fmt: string; name: string; date: string; bytes: number; sha256: string };

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
    if (f) cells.push({ fmt, label, url: fileUrl(series, f.name), bytes: f.bytes, updated: f.updated });
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
  if (c) cell.citywide = { fmt: "cog", url: fileUrl(series, c.name), bytes: c.bytes, updated: c.updated };
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

const csvField = (v: string | number) => (/[",\r\n]/.test(String(v)) ? `"${String(v).replace(/"/g, '""')}"` : String(v));

/** The license of a series' sheets: the project's, plus the Rumsey imagery's for 1948. */
export const sheetLicense = (series: string) =>
  series === "1948" ? `${DATA_LICENSE}; 1948 imagery ${RUMSEY_LICENSE}` : DATA_LICENSE;

/** The credit a series' sheets ask for: sfzoning.com, plus the Rumsey collection for 1948. */
export const sheetCredit = (series: string) =>
  series === "1948" ? `${DATA_CREDIT}; imagery: ${RUMSEY_CREDIT}` : DATA_CREDIT;

const SITE = "https://sfzoning.com";
const abs = (series: string, name: string) => SITE + fileUrl(series, name);

/** The latest name of one current sheet. */
export const sheetLatestName = (series: string, i: ManifestSheet): string =>
  latestName({ series, layer: `sheet-${i.number}`, ext: "tif" });

/** The per-sheet index of a series as CSV. A single sheet carries no README, so each row names its license and credit. */
export function sheetsCsv(m: Manifest, series: string): string {
  const header = "sheet,number,name,url,latest_url,bytes,md5,sha256,updated,license,credit";
  const lines = (Object.hasOwn(m.sheets, series) ? m.sheets[series].items : []).map((i) =>
    [
      i.sheet, i.number, i.name, abs(series, i.name), abs(series, sheetLatestName(series, i)),
      i.bytes, i.md5, i.sha256, i.updated, sheetLicense(series), sheetCredit(series),
    ].map(csvField).join(","),
  );
  return [header, ...lines].join("\n");
}

/** The manifest for scripts: each current file and sheet with its absolute `url` (versioned) and `latest` links. */
export function downloadsJson(m: Manifest) {
  return {
    ...m,
    files: Object.fromEntries(
      Object.entries(m.files).map(([k, f]) => [
        k,
        { ...f, url: abs(f.series, f.name), latest: abs(f.series, latestName({ series: f.series, layer: f.layer, ext: EXTS[f.fmt as keyof typeof EXTS] })) },
      ]),
    ),
    sheets: Object.fromEntries(
      Object.entries(m.sheets).map(([series, s]) => [
        series,
        {
          ...s,
          latest: abs(series, latestName({ series, layer: "sheets", ext: "zip" })),
          items: s.items.map((i) => ({ ...i, url: abs(series, i.name), latest: abs(series, sheetLatestName(series, i)) })),
        },
      ]),
    ),
  };
}
