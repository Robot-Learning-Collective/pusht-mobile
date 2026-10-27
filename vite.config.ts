import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";

const root = dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  define: {
    __SIM_SHA256__: JSON.stringify(createHash("sha256").update(
      readFileSync(resolve(root, "rust/pusht-wasm/pkg/pusht_wasm_bg.wasm")),
    ).digest("hex")),
  },
  build: {
    target: "esnext",
    rollupOptions: {
      input: {
        main: resolve(root, "index.html"),
        leaderboard: resolve(root, "leaderboard.html"),
      },
      output: {
        // Keep Pixi in its own chunk. The entry module uses top-level await
        // (await app.init()), and Pixi loads its renderer via a runtime
        // dynamic import(). If Pixi core is inlined into the entry chunk, that
        // renderer chunk statically depends on the entry — which is suspended
        // mid-await — and the dynamic import deadlocks, so app.init() never
        // resolves and the canvas never mounts (dev works; prod build hangs).
        // Splitting Pixi out breaks that cycle.
        manualChunks(id) {
          if (id.includes("node_modules/pixi.js")) return "pixi";
        },
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
