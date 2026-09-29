import { useEffect } from "react";
import { NavigationType, Outlet, useLocation, useNavigationType } from "react-router";

import { BootstrapGate } from "./BootstrapGate";

export function RootLayout() {
  const { pathname } = useLocation();
  const navigationType = useNavigationType();

  useEffect(() => {
    if (navigationType !== NavigationType.Pop) {
      window.scrollTo(0, 0);
    }
  }, [pathname, navigationType]);

  return (
    <BootstrapGate>
      <Outlet />
    </BootstrapGate>
  );
}
