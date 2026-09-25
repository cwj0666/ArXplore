import { useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  Navigate,
  Route,
  Routes,
  useLocation,
  useNavigationType,
  useSearchParams,
} from "react-router-dom";

import { requestJson } from "./helpers/http";
import { AssistantPage } from "./pages/assistant";
import { PaperDetailPage } from "./pages/detail";
import { ListPage } from "./pages/list";
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


function AssistantRoute() {
  const [searchParams] = useSearchParams();

  return <AssistantPage initialQuery={searchParams.get("q") ?? ""} />;
}


function App() {
  const location = useLocation();
  const navigationType = useNavigationType();

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

  const bootstrap = bootstrapQuery.data;

  return (
    <Routes>
      <Route path="/" element={<ListPage />} />
      <Route path="/papers/" element={<Navigate replace to="/" />} />
      <Route path="/papers/assistant/" element={<AssistantRoute />} />
      <Route path="/papers/:arxivId/" element={<PaperDetailPage bootstrap={bootstrap} />} />
      <Route path="*" element={<NotFoundPage />} />
    </Routes>
  );
}


export default App;
