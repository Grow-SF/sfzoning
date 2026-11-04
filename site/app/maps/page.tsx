import type { Metadata } from "next";
import Explorer from "@/components/Explorer";
import SiteFooter from "@/components/SiteFooter";

const TITLE = "All maps · A History of San Francisco’s Zoning";

const DESCRIPTION =
  "Every historic zoning map of San Francisco from 1921 to today, laid over the same streets. Pick an edition and look up any lot.";

export const metadata: Metadata = {
  title: TITLE,
  description: DESCRIPTION,
  alternates: { canonical: "/maps" },
  openGraph: { title: TITLE, description: DESCRIPTION, url: "/maps" },
};

export default function AllMaps() {
  return (
    <main>
      <Explorer eager />
      <SiteFooter />
    </main>
  );
}
