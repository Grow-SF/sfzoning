import type { Metadata } from "next";
import { notFound } from "next/navigation";
import SiteFooter from "@/components/SiteFooter";
import historyJson from "@/lib/download-versions.json";
import { layerLabel, seriesLayers, VERSIONS_NOTE, versionPairs, type VersionHistory } from "@/lib/versions";
import styles from "@/components/endmatter.module.css";

const history = historyJson as unknown as VersionHistory;

export const dynamicParams = false;

export function generateStaticParams() {
  return [...new Set(versionPairs(history).map((p) => p.series))].map((series) => ({ series }));
}

export async function generateMetadata({ params }: { params: Promise<{ series: string }> }): Promise<Metadata> {
  const { series } = await params;
  const title = `${series} versions · A History of San Francisco’s Zoning`;
  const description = `Every released version of the ${series} download, layer by layer.`;
  return {
    title,
    description,
    alternates: { canonical: `/data/versions/${series}` },
    openGraph: { title, description, url: `/data/versions/${series}` },
  };
}

export default async function SeriesVersionsPage({ params }: { params: Promise<{ series: string }> }) {
  const { series } = await params;
  const layers = seriesLayers(history, series);
  if (!layers.length) notFound();
  return (
    <>
      <main className={styles.page}>
        <p className={styles.kicker}>Open data</p>
        <h1 className={styles.title}>{series} versions</h1>
        <p className={styles.lede}>Every released version, layer by layer.</p>

        <section className={styles.block}>
          <ul className={styles.sourceList}>
            {layers.map((layer) => (
              <li key={layer}>
                <a href={`/data/versions/${series}/${layer}`}>{layerLabel(layer)}</a>
              </li>
            ))}
          </ul>
        </section>

        <section className={styles.block}>
          <p>{VERSIONS_NOTE}</p>
          <p>
            <a href="/data" className={styles.back}>
              Download the data
            </a>
          </p>
        </section>
      </main>
      <SiteFooter />
    </>
  );
}
