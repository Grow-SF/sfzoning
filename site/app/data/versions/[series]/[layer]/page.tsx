import type { Metadata } from "next";
import { notFound } from "next/navigation";
import SiteFooter from "@/components/SiteFooter";
import { formatBytes, type Manifest } from "@/lib/downloadRows";
import manifestJson from "@/lib/downloads.json";
import historyJson from "@/lib/download-versions.json";
import { layerLabel, VERSIONS_NOTE, versionPairs, versionRows, type VersionHistory } from "@/lib/versions";
import styles from "@/components/endmatter.module.css";
import tableStyles from "@/components/downloads.module.css";

const manifest = manifestJson as unknown as Manifest;
const history = historyJson as unknown as VersionHistory;

export const dynamicParams = false;

export function generateStaticParams() {
  return versionPairs(history);
}

type Params = { series: string; layer: string };

export async function generateMetadata({ params }: { params: Promise<Params> }): Promise<Metadata> {
  const { series, layer } = await params;
  const title = `${series} ${layerLabel(layer).toLowerCase()}: every version · A History of San Francisco’s Zoning`;
  const description = `Every released version of the ${series} ${layerLabel(layer).toLowerCase()} download, with its date, size and SHA-256.`;
  return {
    title,
    description,
    alternates: { canonical: `/data/versions/${series}/${layer}` },
    openGraph: { title, description, url: `/data/versions/${series}/${layer}` },
  };
}

export default async function VersionsPage({ params }: { params: Promise<Params> }) {
  const { series, layer } = await params;
  const rows = versionRows(history, manifest, series, layer);
  if (!rows.length) notFound();
  return (
    <>
      <main className={styles.page}>
        <p className={styles.kicker}>Open data</p>
        <h1 className={styles.title}>
          {series} {layerLabel(layer).toLowerCase()}
        </h1>
        <p className={styles.lede}>Every version released, newest first.</p>

        <section className={styles.block}>
          <table className={tableStyles.table}>
            <caption className={tableStyles.caption}>
              Versions of the {series} {layerLabel(layer).toLowerCase()}
            </caption>
            <thead>
              <tr>
                <th scope="col">Released</th>
                <th scope="col">Downloads</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={`${r.date}-${r.files[0].name}`}>
                  <th scope="row" className={tableStyles.era}>
                    <time dateTime={r.date}>{r.date}</time>
                    {r.latest && <span className={tableStyles.meta}>(latest)</span>}
                  </th>
                  <td data-label="Downloads">
                    <span className={tableStyles.cell}>
                      {r.files.map((f) => (
                        <span key={f.name} className={tableStyles.item}>
                          <a href={f.url}>{f.label}</a>{" "}
                          <span className={tableStyles.meta}>
                            {formatBytes(f.bytes)} · hash{" "}
                            <a href={f.checksumUrl}>{f.sha256.slice(0, 8)}</a>
                          </span>
                          <span className={`${tableStyles.meta} ${tableStyles.hash}`}>SHA-256 {f.sha256}</span>
                        </span>
                      ))}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>

        <section className={styles.block}>
          <p>{VERSIONS_NOTE}</p>
          <p>
            <a href={`/data/versions/${series}`} className={styles.back}>
              All {series} layers
            </a>{" "}
            · <a href="/data" className={styles.back}>Download the data</a>
          </p>
        </section>
      </main>
      <SiteFooter />
    </>
  );
}
