import react from "@vitejs/plugin-react";
import { loadEnv } from "vite";
import { defineConfig } from "vitest/config";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, ".", "SOCRATIC_TUTOR_");
  return {
    plugins: [react()],
    server: {
      proxy: {
        "/api": env.SOCRATIC_TUTOR_API_TARGET ?? "http://127.0.0.1:8000",
      },
    },
    test: {
      environment: "jsdom",
      globals: true,
      include: ["src/**/*.test.ts", "src/**/*.test.tsx"],
      setupFiles: "./src/test/setup.ts",
    },
  };
});
