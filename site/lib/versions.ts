import { fileUrl, type Manifest, type Version } from "./downloadRows";
import { naturalCompare } from "./downloadNames.mjs";

export type VersionHistory = { versions: Version[] };

/** The one note on every versions page (the /data page carries the same sentence). */
export const VERSIONS_NOTE = "Use the latest links to always get the newest data. A dated link always returns exactly that version.";

const FORMAT_LABELS: Record<string, string> = { geojson: "GeoJSON", gpkg: "GeoPackage", shp: "Shapefile", cog: "GeoTIFF" };

const LAYER_ORDER = ["lots", "districts", "citywide"];

export type VersionFile = {
  fmt: string;
  label: string;
  name: string;
  url: string;
  bytes: number;
  sha256: string;
  checksumUrl: string;
};
export type VersionRow = { date: string; latest: boolean; files: VersionFile[] };

/** The name a layer's format carries in the current manifest, or undefined when none is published. */
function currentName(m: Manifest, series: string, layer: string, fmt: string): string | undefined {
  if (layer.startsWith("sheet-")) {
    const number = layer.slice("sheet-".length);
    const sheets = Object.hasOwn(m.sheets, series) ? m.sheets[series].items : [];
    return sheets.find((i) => i.number === number)?.name;
  }
  const key = `${series}/${layer}.${fmt}`;
  return Object.hasOwn(m.files, key) ? m.files[key].name : undefined;
}

/**
 * Every release of one layer, newest first. Entries of one release are adjacent and share a date; a format
 * already in the row starts a new row, so two releases on the same day stay two rows.
 */
export function versionRows(history: VersionHistory, manifest: Manifest, series: string, layer: string): VersionRow[] {
  const groups: Version[][] = [];
  for (const v of history.versions) {
    if (v.series !== series || v.layer !== layer) continue;
    const last = groups.at(-1);
    if (last && last[0].date === v.date && !last.some((x) => x.fmt === v.fmt)) last.push(v);
    else groups.push([v]);
  }
  return groups.reverse().map((g) => ({
    date: g[0].date,
    latest: g.some((v) => currentName(manifest, series, layer, v.fmt) === v.name),
    files: g.map((v) => ({
      fmt: v.fmt,
      label: FORMAT_LABELS[v.fmt] ?? v.fmt,
      name: v.name,
      url: fileUrl(series, v.name),
      bytes: v.bytes,
      sha256: v.sha256,
      checksumUrl: fileUrl(series, `${v.name}.sha256`),
    })),
  }));
}

/** The layers of one series that have any release: lots, districts, citywide, then sheets in natural order. */
export function seriesLayers(history: VersionHistory, series: string): string[] {
  const layers = [...new Set(history.versions.filter((v) => v.series === series).map((v) => v.layer))];
  const rank = (l: string) => {
    const i = LAYER_ORDER.indexOf(l);
    return i === -1 ? LAYER_ORDER.length : i;
  };
  return layers.sort((a, b) => rank(a) - rank(b) || naturalCompare(a, b));
}

/** Every series and layer with a release, once each: the static params of the versions pages. */
export function versionPairs(history: VersionHistory): { series: string; layer: string }[] {
  const seen = new Map<string, { series: string; layer: string }>();
  for (const { series, layer } of history.versions) seen.set(`${series}/${layer}`, { series, layer });
  return [...seen.values()];
}

/** A layer as a person names it: "Lots", "Citywide scan", "Sheet 01". */
export function layerLabel(layer: string): string {
  if (layer.startsWith("sheet-")) return `Sheet ${layer.slice("sheet-".length)}`;
  if (layer === "citywide") return "Citywide scan";
  return layer.charAt(0).toUpperCase() + layer.slice(1);
}
