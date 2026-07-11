import { defineConfig } from "vite";

export default defineConfig({
  build: { target: "esnext" },
  esbuild: { target: "esnext" },
  optimizeDeps: { esbuildOptions: { target: "esnext" } },
  server: { host: true, port: 5173 },
});
