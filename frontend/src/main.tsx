import "./shared/styles/tokens.css";
import "./shared/styles/reset.css";

import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { createBrowserRouter } from "react-router";

import { App } from "./app/App";
import { createQueryClient } from "./app/queryClient";
import { routes } from "./app/routes";
import { bootstrapQuery } from "./shared/api/bootstrap";

const container = document.getElementById("root");
if (!container) {
  throw new Error("missing #root element");
}

const queryClient = createQueryClient();
queryClient.query(bootstrapQuery).catch(() => undefined);
const router = createBrowserRouter(routes);

createRoot(container).render(
  <StrictMode>
    <App queryClient={queryClient} router={router} />
  </StrictMode>,
);
