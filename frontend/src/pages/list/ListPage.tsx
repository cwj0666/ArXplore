import { startTransition, useEffect, useState } from "react";
import { Link, useLocation, useNavigate, useSearchParams } from "react-router-dom";

import { AccountMenu, type SettingsTab } from "../../components/account/AccountMenu";
import { ListPagination } from "../../components/list/ListPagination";
import { ListSearchPanel } from "../../components/list/ListSearchPanel";
import { PaperCard } from "../../components/list/PaperCard";
import { toggleFavorite } from "../../helpers/accountApi";
import { ApiError } from "../../helpers/http";
import { buildLoginPath } from "../../helpers/loginPath";
import { fetchPaperList } from "./listApi";
import { buildListHref, buildListSearchParams, type ListParams, normalizeSort, readListParams } from "./listParams";
import type { PaperListResponse, SearchMode, SortOption } from "./listTypes";
import type { BootstrapPayload, FavoriteTogglePayload } from "../../types/app";
import "./listPage.css";

const SORT_OPTIONS = [
  { value: "latest", label: "최신순" },
  { value: "upvotes", label: "추천순" },
] as const;

interface ListPageProps {
  session: BootstrapPayload;
  onOpenSettings: (tab?: SettingsTab) => void;
  onLogout: () => void;
}

export function ListPage({ session, onOpenSettings, onLogout }: ListPageProps) {
  const navigate = useNavigate();
  const location = useLocation();
  const [searchParams, setSearchParams] = useSearchParams();

  // URL이 목록 상태의 원본이다. 뒤로/앞으로 가기가 페이지·정렬·검색어를 그대로 되살린다.
  const params = readListParams(searchParams);
  const { q: query, sort, mode, page } = params;

  const [queryInput, setQueryInput] = useState(query);
  const [listData, setListData] = useState<PaperListResponse | null>(null);
  const [isListLoading, setIsListLoading] = useState(true);
  const [listError, setListError] = useState<string | null>(null);

  useEffect(() => {
    setQueryInput(query);
  }, [query]);

  useEffect(() => {
    const abortController = new AbortController();
    startTransition(() => {
      setIsListLoading(true);
      setListError(null);
    });

    fetchPaperList({ q: query, sort, mode, page }, abortController.signal)
      .then((response) => {
        setListData(response);
      })
      .catch((error: unknown) => {
        if (abortController.signal.aborted) {
          return;
        }
        const message =
          error instanceof Error ? error.message : "목록을 불러오는 중 오류가 발생했습니다.";
        setListError(message);
      })
      .finally(() => {
        if (!abortController.signal.aborted) {
          setIsListLoading(false);
        }
      });

    return () => abortController.abort();
  }, [mode, page, query, sort]);

  const updateParams = (patch: Partial<ListParams>, options: { replace?: boolean } = {}) => {
    const next = buildListSearchParams({ ...params, ...patch });
    if (next.toString() !== searchParams.toString()) {
      setSearchParams(next, { replace: options.replace });
    }
  };

  const requireLogin = () => navigate(buildLoginPath(`${location.pathname}${location.search}`));

  const handleSearchSubmit = () => {
    const trimmed = queryInput.trim();
    if (mode === "search") {
      updateParams({ q: trimmed, page: 1 });
      return;
    }

    if (!trimmed) {
      return;
    }

    navigate(`/papers/assistant/?q=${encodeURIComponent(trimmed)}`);
  };

  const handleSortChange = (nextSort: SortOption) => {
    updateParams({ sort: nextSort, page: 1 });
  };

  const handleModeChange = (nextMode: SearchMode) => {
    updateParams({ mode: nextMode }, { replace: true });
  };

  const handleFavoriteToggle = async (arxivId: string) => {
    let payload: FavoriteTogglePayload;
    try {
      payload = await toggleFavorite(arxivId);
    } catch (error) {
      if (error instanceof ApiError && error.status === 401) {
        requireLogin();
      }
      return;
    }
    setListData((previous) => {
      if (!previous) {
        return previous;
      }
      return {
        ...previous,
        items: previous.items.map((paper) =>
          paper.arxiv_id === arxivId
            ? { ...paper, is_favorited: payload.is_favorited ?? false }
            : paper,
        ),
      };
    });
  };

  const showResultSection = !isListLoading && !listError && !!listData && listData.total_items > 0;
  const totalPages = Math.max(1, listData?.total_pages ?? 1);
  const currentPage = listData?.page ?? page;
  const resultData = showResultSection ? listData : null;

  return (
    <div className="list-page">
      <div className="container">
        <header className="list-hero">
          <div className="list-hero-spacer" aria-hidden="true" />
          <h1>
            <Link to="/">ArXplore</Link>
          </h1>
          <AccountMenu
            className="list-hero-account"
            session={session}
            onOpenSettings={onOpenSettings}
            onLogout={onLogout}
          />
        </header>

        <ListSearchPanel
          mode={mode}
          queryInput={queryInput}
          onModeChange={handleModeChange}
          onQueryInputChange={setQueryInput}
          onSubmit={handleSearchSubmit}
          busy={false}
        />

        {listError ? (
          <div className="no-papers list-error" role="alert">
            {listError}
          </div>
        ) : null}

        {!listError && isListLoading ? (
          <div className="no-papers" role="status">
            불러오는 중...
          </div>
        ) : null}

        {!listError && !isListLoading && (!listData || listData.total_items === 0) ? (
          <div className="no-papers">수집된 논문이 없습니다.</div>
        ) : null}

        {resultData ? (
          <>
            <div className="result-count">
              <span>
                총 {resultData.total_items}개 논문 — {resultData.page} / {resultData.total_pages} 페이지
              </span>
              <select
                aria-label="정렬"
                value={sort}
                onChange={(event) => handleSortChange(normalizeSort(event.target.value))}
              >
                {SORT_OPTIONS.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </div>

            <section className="paper-grid" aria-label="논문 목록">
              {resultData.items.map((paper) => (
                <PaperCard
                  key={paper.arxiv_id}
                  paper={paper}
                  canFavorite={session.is_authenticated}
                  onToggleFavorite={(arxivId) => {
                    void handleFavoriteToggle(arxivId);
                  }}
                  onRequireLogin={requireLogin}
                />
              ))}
            </section>

            <ListPagination
              page={currentPage}
              totalPages={totalPages}
              buildHref={(targetPage) => buildListHref({ ...params, page: targetPage })}
            />
          </>
        ) : null}
      </div>
    </div>
  );
}
