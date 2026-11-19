// DRAFT COPY — NOT REVIEWED BY THE OWNER. Wording is a first draft.
import type { Metadata } from "next";
import Link from "next/link";
import SiteFooter from "@/components/SiteFooter";
import styles from "@/components/releases.module.css";
import { formatBytes, type Version } from "@/lib/downloadRows";
import historyJson from "@/lib/download-versions.json";
import { allReleases } from "@/lib/releaseData";
import { releaseFiles } from "@/lib/releaseFiles";

const history = historyJson as unknown as { versions: Version[] };

export const metadata: Metadata = {
  title: "All releases · Download the data · A History of San Francisco’s Zoning",
  description: "Every data release of San Francisco’s zoning lots, districts and map scans, newest first.",
  alternates: { canonical: "/data/releases" },
};

export default function ReleasesIndex() {
  const rows = allReleases()
    .map((r, i, all) => {
      const files = releaseFiles(r, history);
      return { id: r.id, date: r.date, n: files.length, bytes: files.reduce((s, f) => s + f.bytes, 0), latest: i === all.length - 1 };
    })
    .reverse();
  return (
    <>
      <main className={styles.page}>
        <p className={styles.back}>
          <Link href="/data">← Download the data</Link>
        </p>
        <header>
          <p className={styles.kicker}>Open data</p>
          <h1 className={styles.title}>All releases</h1>
          <p className={styles.lede}>
            Each release is kept as it was published, so work that cites one can find the same files.
          </p>
        </header>
        <div className={styles.wrap}>
          <table className={styles.idx}>
            <thead>
              <tr>
                <th>Release</th>
                <th>Date</th>
                <th className={styles.num}>Files</th>
                <th className={styles.num}>Total size</th>
                <th>Changes</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.id}>
                  <td className={styles.rid}>
                    <Link href={r.latest ? "/data" : `/data/releases/${r.id}`}>{r.id}</Link>
                    {r.latest && <span className={styles.tag}>Latest</span>}
                  </td>
                  <td>{r.date}</td>
                  <td className={styles.num}>{r.n}</td>
                  <td className={styles.num}>{formatBytes(r.bytes)}</td>
                  <td>
                    <Link href={`/data/releases/${r.id}/changes`}>Changes</Link>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </main>
      <SiteFooter />
    </>
  );
}
