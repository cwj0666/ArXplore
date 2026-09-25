import { useEffect, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import {
  Navigate,
  Route,
  Routes,
  useLocation,
  useNavigate,
  useNavigationType,
  useSearchParams,
} from "react-router-dom";

import type { SettingsTab } from "./components/account/AccountMenu";
import { SettingsPanel } from "./components/account/SettingsPanel";
import { postLogout } from "./helpers/accountApi";
import { requestJson } from "./helpers/http";
import { buildLoginPath } from "./helpers/loginPath";
import { AssistantPage } from "./pages/assistant";
import { PaperDetailPage } from "./pages/detail";
import { ListPage } from "./pages/list";
import { LoginPage } from "./pages/login/LoginPage";
import { NotFoundPage } from "./pages/not-found/NotFoundPage";
import type { BootstrapPayload } from "./types/app";


async function fetchBootstrap(): Promise<BootstrapPayload> {
  return requestJson<BootstrapPayload>("/bootstrap.json", {
    method: "GET",
    headers: {
      Accept: "application/json",
    },
  });
}


function AssistantRoute({
  session,
  onOpenSettings,
}: {
  session: BootstrapPayload;
  onOpenSettings: (tab?: SettingsTab) => void;
}) {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();

  return (
    <AssistantPage
      session={session}
      initialQuery={searchParams.get("q") ?? ""}
      homeHref="/"
      onRequireLogin={() => navigate(buildLoginPath("/papers/assistant/"))}
      onOpenSettings={() => onOpenSettings("settings")}
    />
  );
}


function App() {
  const queryClient = useQueryClient();
  const navigate = useNavigate();
  const location = useLocation();
  const navigationType = useNavigationType();
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [settingsTab, setSettingsTab] = useState<SettingsTab>("settings");

  const bootstrapQuery = useQuery({
    queryKey: ["bootstrap"],
    queryFn: fetchBootstrap,
    staleTime: Infinity,
  });

  useEffect(() => {
    if (navigationType !== "POP") {
      window.scrollTo(0, 0);
    }
  }, [location.pathname, navigationType]);

  const refreshBootstrap = async () => {
    await queryClient.invalidateQueries({ queryKey: ["bootstrap"] });
    await bootstrapQuery.refetch();
  };

  const openSettings = (tab: SettingsTab = "settings") => {
    setSettingsTab(tab);
    setSettingsOpen(true);
  };

  const handleLogout = async () => {
    try {
      await postLogout();
    } catch {
      return;
    }
    setSettingsOpen(false);
    await refreshBootstrap();
    const staysReadable = Boolean(bootstrapQuery.data?.demo_mode) && location.pathname.startsWith("/papers/");
    if (location.pathname !== "/" && !staysReadable) {
      navigate("/");
    }
  };

  if (bootstrapQuery.isLoading) {
    return (
      <div className="app-shell-status" role="status">
        앱을 준비하는 중입니다.
      </div>
    );
  }

  if (bootstrapQuery.isError || !bootstrapQuery.data) {
    const message =
      bootstrapQuery.error instanceof Error
        ? bootstrapQuery.error.message
        : "앱 초기화에 실패했습니다.";

    return (
      <div className="app-shell-status" role="alert">
        {message}
      </div>
    );
  }

  const session = bootstrapQuery.data;
  const logout = () => void handleLogout();

  return (
    <>
      <SettingsPanel
        open={settingsOpen}
        initialTab={settingsTab}
        session={session}
        onClose={() => setSettingsOpen(false)}
        onSessionChanged={refreshBootstrap}
      />

      <Routes>
        <Route path="/" element={<ListPage session={session} onOpenSettings={openSettings} onLogout={logout} />} />
        <Route path="/login/" element={<LoginPage onAuthSuccess={refreshBootstrap} />} />
        <Route path="/papers/" element={<Navigate replace to="/" />} />
        <Route
          path="/papers/assistant/"
          element={<AssistantRoute session={session} onOpenSettings={openSettings} />}
        />
        <Route
          path="/papers/:arxivId/"
          element={
            <PaperDetailPage
              session={session}
              onRequireLogin={() => navigate(buildLoginPath(`${location.pathname}${location.search}`))}
              onOpenSettings={openSettings}
              onLogout={logout}
            />
          }
        />
        <Route path="*" element={<NotFoundPage />} />
      </Routes>
    </>
  );
}


export default App;
