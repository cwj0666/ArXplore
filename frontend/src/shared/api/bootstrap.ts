import { queryOptions } from "@tanstack/react-query";
import { createContext, use } from "react";

import { getJson } from "./http";
import type { BootstrapPayload } from "./types";

const BOOTSTRAP_RETRIES = 3;

export const bootstrapQuery = queryOptions({
  queryKey: ["bootstrap"],
  queryFn: ({ signal }) => getJson<BootstrapPayload>("/bootstrap.json", { signal }),
  staleTime: Infinity,
  retry: BOOTSTRAP_RETRIES,
});

export const BootstrapContext = createContext<BootstrapPayload | null>(null);

export function useBootstrap(): BootstrapPayload {
  const bootstrap = use(BootstrapContext);
  if (!bootstrap) {
    throw new Error("useBootstrap must be used inside the bootstrap gate");
  }
  return bootstrap;
}
