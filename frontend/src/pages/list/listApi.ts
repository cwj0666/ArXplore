import { requestJson } from "../../helpers/http";
import type { PaperListResponse, SearchMode, SortOption } from "./listTypes";

interface ListQuery {
  q: string;
  sort: SortOption;
  mode: SearchMode;
  page: number;
}

export async function fetchPaperList(
  query: ListQuery,
  signal?: AbortSignal,
): Promise<PaperListResponse> {
  const params = new URLSearchParams();
  if (query.q) {
    params.set("q", query.q);
  }
  params.set("sort", query.sort);
  params.set("mode", query.mode);
  params.set("page", String(query.page));

  return requestJson<PaperListResponse>(`/papers/list.json?${params.toString()}`, {
    method: "GET",
    signal,
  });
}
