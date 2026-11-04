import manifest from "./lib/downloads.json" with { type: "json" };
import { latestRedirects } from "./lib/latestRedirects.mjs";

/** @type {import("next").NextConfig} */
export default {
  reactStrictMode: true,
  allowedDevOrigins: ["127.0.0.1", ...(process.env.DEV_LAN_HOST ? [process.env.DEV_LAN_HOST] : [])],
  devIndicators: false,
  // Stable /data/latest/<series>/<file> links to the current published files.
  redirects: async () => latestRedirects(manifest),
};
