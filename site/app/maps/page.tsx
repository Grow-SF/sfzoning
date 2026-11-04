import type { Metadata } from "next";
import Explorer from "@/components/Explorer";
import SiteFooter from "@/components/SiteFooter";

const TITLE = "All maps · A History of San Francisco’s Zoning";

export const metadata: Metadata = {
  title: TITLE,
  alternates: { canonical: "/maps" },
  openGraph: { title: TITLE, url: "/maps" },
};

export default function AllMaps() {
  return (
    <main>
      <Explorer eager />
      <SiteFooter />
    </main>
  );
}
