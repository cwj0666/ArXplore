import { useEffect } from "react";
import { useRouteError } from "react-router";

import { AppErrorFallback } from "./AppErrorFallback";

export function RouteErrorFallback() {
  const error = useRouteError();

  useEffect(() => {
    console.error("Unhandled UI error", error);
  }, [error]);

  return <AppErrorFallback />;
}
