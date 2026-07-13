import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";

const root = dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  build: {
    target: "esnext",
    rollupOptions: {
      input: {
        main: resolve(root, "index.html"),
        leaderboard: resolve(root, "leaderboard.html"),
      },
    },
  },
  esbuild: { target: "esnext" },
  optimizeDeps: { esbuildOptions: { target: "esnext" } },
  server: {
    host: true,
    port: 5173,
    proxy: { "/api": "http://localhost:8000" },
  },
});
