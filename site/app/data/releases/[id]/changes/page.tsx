// DRAFT COPY — NOT REVIEWED BY THE OWNER. Wording is a first draft.
import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import SiteFooter from "@/components/SiteFooter";
import styles from "@/components/releases.module.css";
import historyJson from "@/lib/download-versions.json";
import { fileUrl, formatBytes, type Version } from "@/lib/downloadRows";
import { ERAS } from "@/lib/eras";
import { allReleases } from "@/lib/releaseData";
import type { Fmt, ReleaseFile } from "@/lib/releaseFiles";
import { changesView, fileLabel } from "@/lib/releaseView";

export const dynamicParams = false;

const history = historyJson as unknown as { versions: Version[] };
const FMT_LABEL: Record<Fmt, string> = { geojson: "GeoJSON", gpkg: "GeoPackage", shp: "Shapefile", cog: "GeoTIFF" };
const eraLabel = (id: string) => ERAS.find((e) => e.id === id)?.label ?? id;

type Params = { id: string };

export function generateStaticParams() {
  return allReleases().map((r) => ({ id: r.id }));
}

export async function generateMetadata({ params }: { params: Promise<Params> }): Promise<Metadata> {
  const { id } = await params;
  const title = `Changes in release ${id} · Download the data · A History of San Francisco’s Zoning`;
  const url = `/data/releases/${id}/changes`;
  return { title, alternates: { canonical: url }, openGraph: { title, url } };
}

function FileRow({ f, was }: { f: ReleaseFile; was?: boolean }) {
  return (
    <tr className={was ? styles.was : undefined}>
      <td className={styles.layer}>{was ? "was" : `${eraLabel(f.era)} · ${fileLabel(f)}`}</td>
      <td>
        <span className={`${styles.fmt} ${styles[f.fmt]}`}>{FMT_LABEL[f.fmt]}</span>
      </td>
      <td className={styles.file}>
        <a href={fileUrl(f.series, f.name)} title={f.name}>
          {f.name}
        </a>
      </td>
      <td className={styles.num}>{formatBytes(f.bytes)}</td>
      <td className={styles.date}>{f.date}</td>
      <td className={styles.hash}>{f.sha256.slice(0, 8)}</td>
    </tr>
  );
}

function Section({ title, n, children }: { title: string; n: number; children: React.ReactNode }) {
  return (
    <section>
      <h2 className={styles.eral} style={{ display: "block", marginTop: "2rem" }}>
        {title}{" "}
        <span className={styles.eram}>
          {n} file{n === 1 ? "" : "s"}
        </span>
      </h2>
      {n === 0 ? (
        <p className={styles.nothing}>None.</p>
      ) : (
        <div className={styles.wrap}>
          <table className={styles.table}>
            <colgroup>
              <col style={{ width: "16rem" }} />
              <col className={styles.cf} />
              <col />
              <col className={styles.cs} />
              <col className={styles.cd} />
              <col className={styles.ch} />
            </colgroup>
            <thead>
              <tr>
                <th>Layer</th>
                <th>Format</th>
                <th>File</th>
                <th className={styles.num}>Size</th>
                <th className={styles.date}>Updated</th>
                <th className={styles.hash}>SHA-256</th>
              </tr>
            </thead>
            <tbody>{children}</tbody>
          </table>
        </div>
      )}
    </section>
  );
}

export default async function ChangesPage({ params }: { params: Promise<Params> }) {
  const c = changesView((await params).id, history);
  if (!c) notFound();
  return (
    <>
      <main className={styles.page}>
        <p className={styles.back}>
          <Link href={`/data/releases/${c.id}`}>← Release {c.id}</Link> · <Link href="/data/releases">All releases</Link>
        </p>
        <header>
          <p className={styles.kicker}>Open data</p>
          <h1 className={styles.title}>Changes in release {c.id}</h1>
          <p className={styles.lede}>
            {c.first
              ? "This is the first release."
              : `What this release added, changed and removed since release ${c.previousId}.`}
          </p>
        </header>
        <Section title="Added" n={c.added.length}>
          {c.added.map((f) => (
            <FileRow key={f.name} f={f} />
          ))}
        </Section>
        <Section title="Changed" n={c.changed.length}>
          {c.changed.flatMap(({ from, to }) => [
            <FileRow key={to.name} f={to} />,
            <FileRow key={`was-${from.name}`} f={from} was />,
          ])}
        </Section>
        <Section title="Removed" n={c.removed.length}>
          {c.removed.map((f) => (
            <FileRow key={f.name} f={f} />
          ))}
        </Section>
      </main>
      <SiteFooter />
    </>
  );
}
