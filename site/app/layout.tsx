import type { Metadata, Viewport } from "next";
import { Libre_Franklin, Newsreader } from "next/font/google";
import "./globals.css";

// next/font self-hosts both families at build time; no runtime font CDN.
const serif = Newsreader({
  subsets: ["latin"],
  variable: "--font-serif",
  display: "swap",
  style: ["normal", "italic"],
  axes: ["opsz"],
});
const sans = Libre_Franklin({ subsets: ["latin"], variable: "--font-sans", display: "swap" });

const TITLE = "A History of San Francisco’s Zoning";
const DESCRIPTION =
  "A century of San Francisco zoning, told on the city’s own maps: how land where apartments were legal in 1921 came to be zoned for houses.";

// Share cards: app/opengraph-image.jpg (today's zoning, from the site's own map) is picked up
// by file convention. NEXT_PUBLIC_SITE_URL makes its URL absolute; it defaults to the domain.
export const metadata: Metadata = {
  metadataBase: new URL(process.env.NEXT_PUBLIC_SITE_URL || "https://sfzoning.com"),
  title: TITLE,
  description: DESCRIPTION,
  openGraph: { title: TITLE, description: DESCRIPTION, url: "/", siteName: "sfzoning.com", locale: "en_US", type: "article" },
  twitter: { card: "summary_large_image", title: TITLE, description: DESCRIPTION },
  alternates: { canonical: "/" },
};

export const viewport: Viewport = { themeColor: "#efebe2" };

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${serif.variable} ${sans.variable}`}>
      <body>{children}</body>
    </html>
  );
}
