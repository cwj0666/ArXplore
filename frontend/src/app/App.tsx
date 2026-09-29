import { type QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { DataRouter } from "react-router";
import { RouterProvider } from "react-router/dom";

import { ErrorBoundary } from "./ErrorBoundary";

export function App({ queryClient, router }: { queryClient: QueryClient; router: DataRouter }) {
  return (
    <ErrorBoundary>
      <QueryClientProvider client={queryClient}>
        <RouterProvider router={router} />
      </QueryClientProvider>
    </ErrorBoundary>
  );
}
