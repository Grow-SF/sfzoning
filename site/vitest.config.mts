import { defineConfig } from "vitest/config";
import { fileURLToPath } from "node:url";

// .mts: an ESM config, so Vite loads it natively (no "ESM syntax in a CommonJS file" warning).
export default defineConfig({
  resolve: { alias: { "@": fileURLToPath(new URL(".", import.meta.url)) } },
  test: { include: ["tests/**/*.test.ts"] },
});
