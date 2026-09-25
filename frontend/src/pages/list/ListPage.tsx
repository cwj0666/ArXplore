import { startTransition, useEffect, useMemo, useRef, useState } from "react";
import { useLocation, useNavigate, useSearchParams } from "react-router-dom";
import { toggleFavorite } from "../../helpers/accountApi";
import { ListPagination } from "../../components/list/ListPagination";
import { ListSearchPanel } from "../../components/list/ListSearchPanel";
import { PaperCard } from "../../components/list/PaperCard";
import { fetchPaperList } from "./listApi";
import type { PaperListResponse, SearchMode, SortOption } from "./listTypes";
import type { BootstrapPayload, FavoriteTogglePayload } from "../../types/app";
import "./listPage.css";

const SORT_OPTIONS = [
  { value: "latest", label: "최신순" },
  { value: "upvotes", label: "추천순" },
] as const;

function normalizeSort(raw: string | null): SortOption {
  return raw === "upvotes" ? "upvotes" : "latest";
}

function normalizeMode(raw: string | null): SearchMode {
  return raw === "ai" ? "ai" : "search";
}

function normalizePage(raw: string | null): number {
  const parsed = Number(raw);
  if (!Number.isFinite(parsed) || parsed < 1) {
    return 1;
  }
  return Math.floor(parsed);
}

interface ListPageProps {
  session: BootstrapPayload;
  onOpenSettings: (tab?: "settings" | "favorites") => void;
  onLogout: () => void;
}

export function ListPage({ session, onOpenSettings, onLogout }: ListPageProps) {
  const navigate = useNavigate();
  const location = useLocation();
  const accountMenuRef = useRef<HTMLDivElement | null>(null);
  const [searchParams, setSearchParams] = useSearchParams();
  const [accountMenuOpen, setAccountMenuOpen] = useState(false);

  const initialQuery = searchParams.get("q") ?? "";
  const initialSort = normalizeSort(searchParams.get("sort"));
  const initialMode = normalizeMode(searchParams.get("mode"));
  const initialPage = normalizePage(searchParams.get("page"));

  const [queryInput, setQueryInput] = useState(initialQuery);
  const [query, setQuery] = useState(initialQuery);
  const [sort, setSort] = useState<SortOption>(initialSort);
  const [mode, setMode] = useState<SearchMode>(initialMode);
  const [page, setPage] = useState(initialPage);

  const [listData, setListData] = useState<PaperListResponse | null>(null);
  const [isListLoading, setIsListLoading] = useState(true);
  const [listError, setListError] = useState<string | null>(null);

  useEffect(() => {
    const nextParams = new URLSearchParams();
    if (query) {
      nextParams.set("q", query);
    }
    nextParams.set("sort", sort);
    nextParams.set("mode", mode);
    if (page > 1) {
      nextParams.set("page", String(page));
    }

    if (nextParams.toString() !== searchParams.toString()) {
      setSearchParams(nextParams, { replace: true });
    }
  }, [mode, page, query, searchParams, setSearchParams, sort]);

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

  useEffect(() => {
    setAccountMenuOpen(false);
  }, [location.pathname, location.search]);

  useEffect(() => {
    if (!accountMenuOpen) {
      return;
    }

    const handleClick = (event: MouseEvent) => {
      if (!accountMenuRef.current?.contains(event.target as Node)) {
        setAccountMenuOpen(false);
      }
    };

    document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, [accountMenuOpen]);

  const showResultSection = useMemo(() => {
    return !isListLoading && !listError && !!listData && listData.total_items > 0;
  }, [isListLoading, listData, listError]);

  const handleSearchSubmit = () => {
    const trimmed = queryInput.trim();
    if (mode === "search") {
      setQuery(trimmed);
      setPage(1);
      return;
    }

    if (!trimmed) {
      return;
    }

    navigate(`/papers/assistant/?q=${encodeURIComponent(trimmed)}`);
  };

  const handleSortChange = (nextSort: SortOption) => {
    setSort(nextSort);
    setPage(1);
  };

  const handleModeChange = (nextMode: SearchMode) => {
    setMode(nextMode);
  };

  const handleFavoriteToggle = async (arxivId: string) => {
    let payload: FavoriteTogglePayload;
    try {
      payload = await toggleFavorite(arxivId);
    } catch {
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

  const totalPages = Math.max(1, listData?.total_pages ?? 1);
  const currentPage = listData?.page ?? page;
  const resultData = showResultSection ? listData : null;
  const initial = session.username ? session.username.slice(0, 1).toUpperCase() : "?";

  return (
    <div className="list-page">
      <div className="container">
        <header className="list-hero">
          <div className="list-hero-spacer" aria-hidden="true" />
          <h1>
            <a href="/">ArXplore</a>
          </h1>
          <div className="list-hero-account" ref={accountMenuRef}>
            {!session.is_authenticated ? (
              <a
                className="app-header-login"
                href={`/login/?next=${encodeURIComponent(`${location.pathname}${location.search}`)}`}
              >
                로그인
              </a>
            ) : (
              <>
                <button
                  type="button"
                  className="account-trigger"
                  onClick={() => setAccountMenuOpen((value) => !value)}
                >
                  <span className="account-trigger-badge">{initial}</span>
                  <span>{session.username}</span>
                </button>

                {accountMenuOpen ? (
                  <div className="account-menu">
                    <button type="button" onClick={() => onOpenSettings("settings")}>
                      내 설정
                    </button>
                    <button type="button" onClick={() => onOpenSettings("favorites")}>
                      즐겨찾기
                    </button>
                    <button type="button" onClick={onLogout}>
                      로그아웃
                    </button>
                  </div>
                ) : null}
              </>
            )}
          </div>
        </header>

        <ListSearchPanel
          mode={mode}
          queryInput={queryInput}
          onModeChange={handleModeChange}
          onQueryInputChange={setQueryInput}
          onSubmit={handleSearchSubmit}
          busy={false}
        />

        {listError ? <div className="no-papers list-error">{listError}</div> : null}

        {!listError && isListLoading ? <div className="no-papers">불러오는 중...</div> : null}

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

            <section className="paper-grid">
              {resultData.items.map((paper) => (
                <PaperCard
                  key={paper.arxiv_id}
                  paper={paper}
                  canOpenDetail={session.is_authenticated}
                  canFavorite={session.is_authenticated}
                  onToggleFavorite={(arxivId) => {
                    void handleFavoriteToggle(arxivId);
                  }}
                  onRequireLogin={() =>
                    navigate(`/login/?next=${encodeURIComponent(`${window.location.pathname}${window.location.search}`)}`)
                  }
                />
              ))}
            </section>

            <ListPagination
              page={currentPage}
              totalPages={totalPages}
              query={query}
              sort={sort}
              mode={mode}
              onPageChange={setPage}
            />
          </>
        ) : null}
      </div>
    </div>
  );
}
