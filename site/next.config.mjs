/** @type {import("next").NextConfig} */
export default {
  reactStrictMode: true,
  allowedDevOrigins: ["127.0.0.1", ...(process.env.DEV_LAN_HOST ? [process.env.DEV_LAN_HOST] : [])],
  devIndicators: false,
};
