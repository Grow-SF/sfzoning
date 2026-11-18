// DRAFT COPY — NOT REVIEWED BY THE OWNER. Link and license wording are first drafts.
import { DATA_CREDIT, RUMSEY_CREDIT } from "@/lib/credits";
import { formatBytes, type Manifest } from "@/lib/downloadRows";
import styles from "./downloads.module.css";

type Props = { series: string; sheets: Manifest["sheets"][string]; license?: string };

/**
 * The sheets of one series: all of them as a zip, or one at a time. Both are ordinary downloads
 * served by the site (app/data/sheets/[series]/zip and .../sheet/[sheet]), so the browser shows
 * its usual progress and saves to the downloads folder.
 */
export default function SheetDownload({ series, sheets, license }: Props) {
  const total = sheets.items.reduce((n, i) => n + i.bytes, 0);
  const base = `/data/sheets/${encodeURIComponent(series)}`;
  return (
    <span className={styles.item}>
      {license ? (
        <a className={styles.button} href={`${base}/zip`} download={`${series}-sheets.zip`}>
          All {sheets.items.length} sheets ({formatBytes(total)}, zip)
        </a>
      ) : (
        // No license text to put in the zip: offer the sheets one at a time only.
        <span>
          {sheets.items.length} sheets ({formatBytes(total)})
        </span>
      )}
      <span className={styles.meta}>
        {sheets.updated} · <a href={base}>sheet list (CSV)</a>
      </span>
      <details className={styles.sheets}>
        <summary>One sheet at a time</summary>
        <span className={styles.meta}>
          A single sheet comes without a README. It is licensed{" "}
          <a href="https://creativecommons.org/licenses/by-nc-sa/4.0/">CC BY-NC-SA 4.0</a>; credit it as “{DATA_CREDIT}”.
          {series === "1948" && (
            <>
              {" "}The 1948 images are from the {RUMSEY_CREDIT}, licensed{" "}
              <a href="https://creativecommons.org/licenses/by-nc-sa/3.0/">CC BY-NC-SA 3.0</a>.
            </>
          )}
        </span>
        <ul className={styles.sheetList}>
          {sheets.items.map((i) => (
            <li key={i.sheet}>
              <a href={`${base}/sheet/${encodeURIComponent(i.sheet)}`} download={i.name}>
                {i.name}
              </a>{" "}
              <span className={styles.size}>{formatBytes(i.bytes)}</span>
            </li>
          ))}
        </ul>
      </details>
    </span>
  );
}
