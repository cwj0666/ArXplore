import { useQuery } from "@tanstack/react-query";
import type { ReactNode } from "react";

import { BootstrapContext, bootstrapQuery } from "../shared/api/bootstrap";
import { PreparingScreen, StatusScreen } from "./StatusScreen";

export function BootstrapGate({ children }: { children: ReactNode }) {
  const { data, error } = useQuery(bootstrapQuery);

  if (!data) {
    return error ? <StatusScreen message={error.message} role="alert" /> : <PreparingScreen />;
  }

  return <BootstrapContext value={data}>{children}</BootstrapContext>;
}
