import type { RouteObject } from "react-router";

import NotFoundPage from "../pages/NotFoundPage";
import { PapersAliasRedirect } from "./PapersAliasRedirect";
import { RootLayout } from "./RootLayout";
import { RouteErrorFallback } from "./RouteErrorFallback";
import { PreparingScreen } from "./StatusScreen";

export const routes: RouteObject[] = [
  {
    Component: RootLayout,
    ErrorBoundary: RouteErrorFallback,
    HydrateFallback: PreparingScreen,
    children: [
      { path: "/", lazy: async () => ({ Component: (await import("../pages/ListPage")).default }) },
      { path: "/papers/", Component: PapersAliasRedirect },
      {
        path: "/papers/assistant/",
        lazy: async () => ({ Component: (await import("../pages/AssistantPage")).default }),
      },
      {
        path: "/papers/:arxivId/",
        lazy: async () => ({ Component: (await import("../pages/PaperDetailPage")).default }),
      },
      { path: "*", Component: NotFoundPage },
    ],
  },
];
