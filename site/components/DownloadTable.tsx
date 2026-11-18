import { formatBytes, rows, type Cell, type FileLinks, type Manifest, type ScanCell } from "@/lib/downloadRows";
import SheetDownload from "./SheetDownload";
import styles from "./downloads.module.css";

/** One file: its latest link (what the label downloads), its dated name for citing, and its versions page. */
function FileItem({ label, file }: { label: string; file: FileLinks & { bytes: number; updated: string } }) {
  const datedName = file.versionedUrl.slice(file.versionedUrl.lastIndexOf("/") + 1);
  return (
    <span className={styles.item}>
      <a href={file.latestUrl}>{label}</a>
      <span className={styles.meta}>
        {formatBytes(file.bytes)} · {file.updated}
      </span>
      <span className={styles.meta}>
        <a href={file.versionedUrl}>{datedName}</a> · <a href={file.versionsUrl}>versions</a>
      </span>
    </span>
  );
}

function Links({ cells }: { cells: Cell[] }) {
  if (!cells.length) return <span className={styles.none}>Not yet published</span>;
  return (
    <span className={styles.cell}>
      {cells.map((c) => (
        <FileItem key={c.fmt} label={c.label} file={c} />
      ))}
    </span>
  );
}

function Scan({ scan, manifest }: { scan: ScanCell; manifest: Manifest }) {
  return (
    <div className={styles.scan}>
      {scan.label && <span className={styles.label}>{scan.label}</span>}
      {scan.citywide && (
        <FileItem label="Citywide image (GeoTIFF)" file={scan.citywide} />
      )}
      {/* The client component gets only this series' sheets, not the whole manifest. */}
      {scan.sheets && (
        <SheetDownload series={scan.series} sheets={manifest.sheets[scan.series]} license={manifest.license} />
      )}
    </div>
  );
}

export default function DownloadTable({ manifest }: { manifest: Manifest }) {
  return (
    <table className={styles.table}>
      <caption className={styles.caption}>Downloads by map edition</caption>
      <thead>
        <tr>
          <th scope="col">Edition</th>
          <th scope="col">Lots</th>
          <th scope="col">Districts</th>
          <th scope="col">Scans</th>
        </tr>
      </thead>
      <tbody>
        {rows(manifest).map((r) => {
          const scans = r.scans.filter((s) => s.citywide || s.sheets);
          return (
            <tr key={r.era.id}>
              <th scope="row" className={styles.era}>
                {r.era.label}
              </th>
              <td data-label="Lots">
                <Links cells={r.lots} />
              </td>
              <td data-label="Districts">
                <Links cells={r.districts} />
              </td>
              <td data-label="Scans">
                {scans.length ? (
                  scans.map((s) => <Scan key={s.series} scan={s} manifest={manifest} />)
                ) : (
                  <span className={styles.none}>{r.scans.length ? "Not yet published" : "No scan"}</span>
                )}
              </td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}
