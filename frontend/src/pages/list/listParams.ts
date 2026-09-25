import type { SearchMode, SortOption } from "./listTypes";


export interface ListParams {
  q: string;
  sort: SortOption;
  mode: SearchMode;
  page: number;
}


export function normalizeSort(raw: string | null): SortOption {
  return raw === "upvotes" ? "upvotes" : "latest";
}


export function normalizeMode(raw: string | null): SearchMode {
  return raw === "ai" ? "ai" : "search";
}


export function normalizePage(raw: string | null): number {
  const parsed = Number(raw);
  if (!Number.isFinite(parsed) || parsed < 1) {
    return 1;
  }
  return Math.floor(parsed);
}


export function readListParams(searchParams: URLSearchParams): ListParams {
  return {
    q: searchParams.get("q") ?? "",
    sort: normalizeSort(searchParams.get("sort")),
    mode: normalizeMode(searchParams.get("mode")),
    page: normalizePage(searchParams.get("page")),
  };
}


/** 기본값(최신순, 키워드 검색, 1페이지)은 URL에서 생략한다. */
export function buildListSearchParams(params: ListParams): URLSearchParams {
  const next = new URLSearchParams();
  if (params.q) {
    next.set("q", params.q);
  }
  if (params.sort !== "latest") {
    next.set("sort", params.sort);
  }
  if (params.mode !== "search") {
    next.set("mode", params.mode);
  }
  if (params.page > 1) {
    next.set("page", String(params.page));
  }
  return next;
}


export function buildListHref(params: ListParams): string {
  const search = buildListSearchParams(params).toString();
  return search ? `/?${search}` : "/";
}
