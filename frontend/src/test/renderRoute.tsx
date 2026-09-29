import { QueryClientProvider } from "@tanstack/react-query";
import { render } from "@testing-library/react";
import type { ComponentType } from "react";
import { createMemoryRouter, Outlet, RouterProvider } from "react-router";

import { createQueryClient } from "../app/queryClient";
import { BootstrapContext } from "../shared/api/bootstrap";
import type { BootstrapPayload } from "../shared/api/types";

const TEST_BOOTSTRAP: BootstrapPayload = {
  default_summary_model: "gpt-5-mini",
  available_summary_models: ["gpt-5-mini", "gpt-5"],
};

function BootstrapLayout() {
  return (
    <BootstrapContext value={TEST_BOOTSTRAP}>
      <Outlet />
    </BootstrapContext>
  );
}

function RouteProbe() {
  return <p>other route</p>;
}

export function renderRoute(path: string, Component: ComponentType, initialEntry: string) {
  const router = createMemoryRouter(
    [
      {
        Component: BootstrapLayout,
        children: [
          { path, Component },
          { path: "*", Component: RouteProbe },
        ],
      },
    ],
    { initialEntries: [initialEntry] },
  );
  const queryClient = createQueryClient();
  const view = render(
    <QueryClientProvider client={queryClient}>
      <RouterProvider router={router} />
    </QueryClientProvider>,
  );
  return { router, queryClient, ...view };
}
