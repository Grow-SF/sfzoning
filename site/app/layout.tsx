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

export const metadata: Metadata = {
  title: "How San Francisco was zoned",
  description: "A century of San Francisco zoning, told on the original maps.",
};

export const viewport: Viewport = { themeColor: "#efebe2" };

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${serif.variable} ${sans.variable}`}>
      <body>{children}</body>
    </html>
  );
}
