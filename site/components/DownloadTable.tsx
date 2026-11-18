import { formatBytes, rows, type Cell, type Manifest, type ScanCell } from "@/lib/downloadRows";
import SheetDownload from "./SheetDownload";
import styles from "./downloads.module.css";

function Links({ cells }: { cells: Cell[] }) {
  if (!cells.length) return <span className={styles.none}>Not yet published</span>;
  return (
    <span className={styles.cell}>
      {cells.map((c) => (
        <span key={c.fmt} className={styles.item}>
          <a href={c.url}>{c.label}</a>
          <span className={styles.meta}>
            {formatBytes(c.bytes)} · {c.updated}
          </span>
        </span>
      ))}
    </span>
  );
}

function Scan({ scan, manifest }: { scan: ScanCell; manifest: Manifest }) {
  return (
    <div className={styles.scan}>
      {scan.label && <span className={styles.label}>{scan.label}</span>}
      {scan.citywide && (
        <span className={styles.item}>
          <a href={scan.citywide.url}>Citywide image (GeoTIFF)</a>
          <span className={styles.meta}>
            {formatBytes(scan.citywide.bytes)} · {scan.citywide.updated}
          </span>
        </span>
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
