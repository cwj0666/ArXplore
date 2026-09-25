import { useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { Navigate, Route, Routes, useLocation, useNavigate, useSearchParams } from "react-router-dom";

import { AppHeader } from "./components/account/AppHeader";
import { SettingsPanel } from "./components/account/SettingsPanel";
import { postLogout } from "./helpers/accountApi";
import { requestJson } from "./helpers/http";
import { AssistantPage } from "./pages/assistant";
import { PaperDetailPage } from "./pages/detail";
import { ListPage } from "./pages/list";
import { LoginPage } from "./pages/login/LoginPage";
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
  onOpenSettings: (tab?: "settings" | "favorites") => void;
}) {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();

  return (
    <AssistantPage
      session={session}
      initialQuery={searchParams.get("q") ?? ""}
      homeHref="/"
      onRequireLogin={() => navigate(`/login/?next=${encodeURIComponent("/papers/assistant/")}`)}
      onOpenSettings={() => onOpenSettings("settings")}
    />
  );
}


function App() {
  const queryClient = useQueryClient();
  const navigate = useNavigate();
  const location = useLocation();
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [settingsTab, setSettingsTab] = useState<"settings" | "favorites">("settings");

  const bootstrapQuery = useQuery({
    queryKey: ["bootstrap"],
    queryFn: fetchBootstrap,
    staleTime: Infinity,
  });

  const refreshBootstrap = async () => {
    await queryClient.invalidateQueries({ queryKey: ["bootstrap"] });
    await bootstrapQuery.refetch();
  };

  const openSettings = (tab: "settings" | "favorites" = "settings") => {
    setSettingsTab(tab);
    setSettingsOpen(true);
  };

  const handleLogout = async () => {
    try {
      await postLogout();
    } catch {
      return;
    }
    await refreshBootstrap();
    if (location.pathname !== "/") {
      navigate("/");
    }
  };

  if (bootstrapQuery.isLoading) {
    return <div className="app-shell-status">앱을 준비하는 중입니다.</div>;
  }

  if (bootstrapQuery.isError || !bootstrapQuery.data) {
    const message =
      bootstrapQuery.error instanceof Error
        ? bootstrapQuery.error.message
        : "앱 초기화에 실패했습니다.";

    return <div className="app-shell-status">{message}</div>;
  }

  const session = bootstrapQuery.data;

  return (
    <>
      <AppHeader session={session} onOpenSettings={openSettings} onLogout={() => void handleLogout()} />
      <SettingsPanel
        open={settingsOpen}
        initialTab={settingsTab}
        session={session}
        onClose={() => setSettingsOpen(false)}
        onSessionChanged={refreshBootstrap}
      />

      <Routes>
        <Route
          path="/"
          element={<ListPage session={session} onOpenSettings={openSettings} onLogout={() => void handleLogout()} />}
        />
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
              onRequireLogin={() =>
                navigate(`/login/?next=${encodeURIComponent(`${location.pathname}${location.search}`)}`)
              }
              onOpenSettings={openSettings}
              onLogout={() => void handleLogout()}
            />
          }
        />
      </Routes>
    </>
  );
}


export default App;
