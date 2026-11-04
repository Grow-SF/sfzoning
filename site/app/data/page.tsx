// DRAFT COPY — NOT REVIEWED BY THE OWNER. Intro, licence and credit wording are first drafts.
import type { Metadata } from "next";
import { notFound } from "next/navigation";
import { RUMSEY_CREDIT } from "@/components/Credits";
import DownloadTable from "@/components/DownloadTable";
import SiteFooter from "@/components/SiteFooter";
import { lastUpdated, type Manifest } from "@/lib/downloadRows";
import manifestJson from "@/lib/downloads.json";
import styles from "@/components/endmatter.module.css";
import tableStyles from "@/components/downloads.module.css";

const manifest = manifestJson as unknown as Manifest;
const TITLE = "Download the data · A History of San Francisco’s Zoning";
const DESCRIPTION =
  "San Francisco’s zoning lots and districts from 1921 to today as GeoJSON, GeoPackage and Shapefile, plus the georeferenced map scans.";

export const metadata: Metadata = {
  title: TITLE,
  description: DESCRIPTION,
  alternates: { canonical: "/data" },
  openGraph: { title: TITLE, description: DESCRIPTION, url: "/data" },
};

export default function DataPage() {
  if (!Object.keys(manifest.files).length) notFound();
  const seriesWithSheets = Object.keys(manifest.sheets);
  return (
    <>
      <main className={styles.page}>
        <p className={styles.kicker}>Open data</p>
        <h1 className={styles.title}>Download the data</h1>
        <p className={styles.lede}>
          For researchers, planners, journalists and anyone who wants to put San Francisco’s past zoning in a GIS
          program of their own.
        </p>

        <section className={styles.block}>
          <h2 className={styles.heading}>What is here</h2>
          <p>
            <strong>Lots</strong> are today’s parcels, each tagged with the district the historic map put it in.{" "}
            <strong>Districts</strong> are those same lots dissolved into one shape per district. <strong>Scans</strong>{" "}
            are the historic map pages themselves, aligned to today’s streets and left as scanned: one citywide image,
            and the individual sheets.
          </p>
          <p>
            Lots and districts come as GeoJSON, GeoPackage and Shapefile, in WGS84 (EPSG:4326). The GeoPackage also
            carries a copy in California State Plane (zone III, feet). Each download includes a README and the license.
          </p>
          <p>
            This is a reconstruction from scanned maps, read by a program and checked by people, not an official
            record. Read <a href="/about">how the maps were made</a> and how accurate they are before relying on them.
          </p>
        </section>

        <section className={styles.block}>
          <h2 className={styles.heading}>License and credit</h2>
          <p>
            The data is licensed{" "}
            <a href="https://creativecommons.org/licenses/by-nc-sa/4.0/">CC BY-NC-SA 4.0</a>: share and adapt it for
            non-commercial use, with credit, under the same license. When you use it, credit it as: “Based on
            sfzoning.com by Steven Bacio for GrowSF (https://growsf.org)”.
          </p>
          <p>
            Lot lines are DataSF parcels (San Francisco Open Data). The 1948 atlas images are from the {RUMSEY_CREDIT}.
          </p>
        </section>

        <section className={styles.block}>
          <h2 className={styles.heading}>Files</h2>
          <p>
            Data last updated <time dateTime={lastUpdated(manifest)}>{lastUpdated(manifest)}</time>. Sizes are
            for the downloads as they are.
          </p>
          <DownloadTable manifest={manifest} />
          <p className={tableStyles.more}>
            <a href="/data/downloads.json">All files and checksums (JSON)</a>
            {seriesWithSheets.map((s) => (
              <a key={s} href={`/data/sheets/${s}`}>
                Sheets {s} (CSV)
              </a>
            ))}
          </p>
        </section>
      </main>
      <SiteFooter />
    </>
  );
}
