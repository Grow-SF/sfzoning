/** @type {import("next").NextConfig} */
export default {
  reactStrictMode: true,
  allowedDevOrigins: ["127.0.0.1"],
  devIndicators: false,
  // Always defined, so the bundler inlines it and minification drops the test-only map hooks
  // from production bundles. (An unset NEXT_PUBLIC_ variable is left as a runtime lookup.)
  env: { NEXT_PUBLIC_E2E: process.env.NEXT_PUBLIC_E2E === "1" ? "1" : "" },
};
