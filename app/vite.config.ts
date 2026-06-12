import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Vite frontend lives in /app (per spec: "gesture-geometry/app/src/*").
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    strictPort: true
  }
});

