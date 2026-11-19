// What the release page shows, worked out from the record: pure, so tests check it without React.
import { citations, type Citations } from "./citation.mjs";
import { ERAS } from "./eras";
import { fileUrl, scanSeries, versionsUrl, type Manifest, type Version } from "./downloadRows";
import { EXTS, latestName } from "./downloadNames.mjs";
import { isLatest, previousRelease } from "./releaseData";
import { releaseFiles, type ReleaseFile } from "./releaseFiles";
import type { Release, ReleaseRecord } from "./releases.mjs";

export type FileRow = { type: "file"; file: ReleaseFile; label: string; latestUrl: string; historyUrl: string };
export type SheetRow = {
  type: "sheets"; series: string; era: string; label: string; count: number; bytes: number; date: string;
  sheets: (ReleaseFile & { label: string; latestUrl: string; historyUrl: string })[];
  /** The ready-made sheet zip; only the newest release's sheets are the ones it holds. */
  zipUrl?: string;
  listUrl: string;
};
export type Row = FileRow | SheetRow;
export type Edition = { era: string; label: string; rows: Row[] };
export type ReleaseView = {
  id: string; date: string; latest: boolean; previousId?: string;
  totals: { files: number; bytes: number; editions: number };
  citations: Citations;
  editions: Edition[];
};

/** "Scan", or "Zoning scan" / "Height scan" / "Special-use scan" for the books with three. */
function scanLabel(series: string): string {
  for (const era of ERAS) {
    const s = scanSeries(era).find((x) => x.series === series);
    if (s) return s.label === "" ? "Scan" : s.label === "Special use" ? "Special-use scan" : `${s.label} scan`;
  }
  return "Scan";
}

function fileLabel(f: ReleaseFile): string {
  if (f.kind === "lots") return "Lots";
  if (f.kind === "districts") return "Districts";
  if (f.kind === "scans") return `${scanLabel(f.series)}, citywide`;
  return `Sheet ${f.layer.replace(/^sheet-/, "")}`;
}

const links = (f: ReleaseFile) => ({
  latestUrl: fileUrl(f.series, latestName({ series: f.series, layer: f.layer, ext: EXTS[f.fmt] })),
  historyUrl: versionsUrl(f.series, f.layer),
});

/** One release as the page shows it: per edition, its files, with each scan series' sheets as one group row. */
export function releaseView(
  release: Release,
  history: { versions: Version[] },
  manifest: Manifest,
  record?: ReleaseRecord,
): ReleaseView {
  const latest = isLatest(release.id, record);
  const files = releaseFiles(release, history);
  const editions: Edition[] = [];
  for (const era of ERAS) {
    const rows: Row[] = [];
    for (const f of files.filter((x) => x.era === era.id)) {
      if (f.kind !== "sheets") {
        rows.push({ type: "file", file: f, label: fileLabel(f), ...links(f) });
        continue;
      }
      const last = rows.at(-1);
      const sheet = { ...f, label: fileLabel(f), ...links(f) };
      if (last?.type === "sheets" && last.series === f.series) {
        last.sheets.push(sheet);
        last.count++;
        last.bytes += f.bytes;
        if (f.date > last.date) last.date = f.date;
        continue;
      }
      const current = latest && manifest.license && manifest.sheets[f.series]?.items.length;
      rows.push({
        type: "sheets", series: f.series, era: era.id, label: `${scanLabel(f.series)}, sheets`,
        count: 1, bytes: f.bytes, date: f.date, sheets: [sheet],
        ...(current ? { zipUrl: fileUrl(f.series, latestName({ series: f.series, layer: "sheets", ext: "zip" })) } : {}),
        listUrl: current
          ? `/data/sheets/${f.series}`
          : `/data/releases/${release.id}/urls.txt?layer=sheets&edition=${era.id}`,
      });
    }
    if (rows.length) editions.push({ era: era.id, label: era.label, rows });
  }
  return {
    id: release.id,
    date: release.date,
    latest,
    previousId: previousRelease(release.id, record)?.id,
    totals: { files: files.length, bytes: files.reduce((n, f) => n + f.bytes, 0), editions: editions.length },
    citations: citations(release),
    editions,
  };
}
