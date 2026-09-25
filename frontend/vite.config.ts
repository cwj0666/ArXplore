/// <reference types="node" />
import { defineConfig, type ProxyOptions } from "vite";
import react from "@vitejs/plugin-react-swc";

const djangoTarget = "http://arxplore-django:8001";
const buildBasePath = process.env.VITE_BASE_PATH || "/static/frontend/";

const proxy = (target: string): ProxyOptions => ({
  target,
  changeOrigin: false,
  headers: { host: "localhost" },
  configure: (proxyServer) => {
    proxyServer.on("proxyReq", (proxyReq, req) => {
      const clientIp = req.socket.remoteAddress ?? "127.0.0.1";
      proxyReq.setHeader("X-Real-IP", clientIp);
      proxyReq.setHeader("X-Forwarded-For", clientIp);
    });
  },
});

export default defineConfig(({ command }) => ({
  base: command === "serve" ? "/" : buildBasePath,
  plugins: [react()],
  server: {
    host: "0.0.0.0",
    port: 5173,
    proxy: {
      "/bootstrap.json": proxy(djangoTarget),
      "/auth/": proxy(djangoTarget),
      "/settings/": proxy(djangoTarget),
      "/favorites/": proxy(djangoTarget),
      "/papers/list.json": proxy(djangoTarget),
      "/papers/assistant/chat/": proxy(djangoTarget),
      "/papers/assistant/stream/": proxy(djangoTarget),
      "^/papers/[^/]+/detail\\.json$": proxy(djangoTarget),
      "^/papers/[^/]+/analyze/$": proxy(djangoTarget),
      "^/papers/[^/]+/summary/$": proxy(djangoTarget),
      "^/papers/[^/]+/chat/$": proxy(djangoTarget),
      "^/papers/[^/]+/chat/stream/$": proxy(djangoTarget),
      "/admin/": proxy(djangoTarget),
      "/static/": proxy(djangoTarget),
    }
  }
}));
