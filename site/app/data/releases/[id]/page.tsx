import type { Metadata } from "next";
import { notFound } from "next/navigation";
import ReleasePage from "@/components/ReleasePage";
import { allReleases, findRelease, isLatest } from "@/lib/releaseData";

export const dynamicParams = false;

export function generateStaticParams() {
  return allReleases().map((r) => ({ id: r.id }));
}

type Params = { id: string };

export async function generateMetadata({ params }: { params: Promise<Params> }): Promise<Metadata> {
  const { id } = await params;
  const title = `Release ${id} · Download the data · A History of San Francisco’s Zoning`;
  const description = `Data release ${id}: San Francisco’s zoning lots and districts from 1921 to today as GeoJSON, GeoPackage and Shapefile, plus the georeferenced map scans.`;
  const url = `/data/releases/${id}`;
  return {
    title,
    description,
    // The newest release is the same page as /data.
    alternates: { canonical: isLatest(id) ? "/data" : url },
    openGraph: { title, description, url },
  };
}

export default async function ReleaseIdPage({ params }: { params: Promise<Params> }) {
  const release = findRelease((await params).id);
  if (!release) notFound();
  return <ReleasePage release={release} />;
}
