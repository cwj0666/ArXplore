import react from "@vitejs/plugin-react-swc";
import type { ProxyOptions } from "vite";
import { defineConfig } from "vitest/config";

const apiTarget = process.env.API_PROXY_TARGET ?? "http://127.0.0.1:8001";

const apiPaths = [
  "/bootstrap.json",
  "/papers/list.json",
  "/papers/assistant/chat/",
  "/papers/assistant/stream/",
  "^/papers/[^/]+/detail\\.json$",
  "^/papers/[^/]+/analyze/$",
  "^/papers/[^/]+/summary/$",
  "^/papers/[^/]+/chat/$",
  "^/papers/[^/]+/chat/stream/$",
];

const apiProxy: ProxyOptions = {
  target: apiTarget,
  changeOrigin: false,
  configure: (proxyServer) => {
    proxyServer.on("proxyReq", (proxyReq, req) => {
      const clientIp = req.socket.remoteAddress ?? "127.0.0.1";
      proxyReq.setHeader("X-Real-IP", clientIp);
      proxyReq.setHeader("X-Forwarded-For", clientIp);
    });
  },
};

const proxy = Object.fromEntries(apiPaths.map((path) => [path, apiProxy]));

export default defineConfig({
  base: "/",
  plugins: [react()],
  server: {
    host: "0.0.0.0",
    port: 5173,
    proxy,
  },
  preview: {
    proxy,
  },
  test: {
    environment: "jsdom",
    setupFiles: ["./src/test/setup.ts"],
    restoreMocks: true,
    unstubGlobals: true,
  },
});
