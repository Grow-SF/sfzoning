import type { Metadata } from "next";
import { notFound } from "next/navigation";
import ReleasePage from "@/components/ReleasePage";
import { findRelease } from "@/lib/releaseData";

const TITLE = "Download the data · A History of San Francisco’s Zoning";
const DESCRIPTION =
  "San Francisco’s zoning lots and districts from 1921 to today as GeoJSON, GeoPackage and Shapefile, plus the georeferenced map scans.";

export const metadata: Metadata = {
  title: TITLE,
  description: DESCRIPTION,
  alternates: { canonical: "/data" },
  openGraph: { title: TITLE, description: DESCRIPTION, url: "/data" },
};

/** /data is the newest release. */
export default function DataPage() {
  const release = findRelease("latest");
  if (!release) notFound();
  return <ReleasePage release={release} />;
}
