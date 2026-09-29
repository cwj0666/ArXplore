import { queryOptions } from "@tanstack/react-query";

import { getJson } from "../../shared/api/http";
import type { PaperListResponse } from "../../shared/api/types";
import type { ListParams } from "./listParams";

function listUrl({ q, sort, mode, page }: ListParams): string {
  const params = new URLSearchParams();
  if (q) {
    params.set("q", q);
  }
  params.set("sort", sort);
  params.set("mode", mode);
  params.set("page", String(page));
  return `/papers/list.json?${params.toString()}`;
}

export function paperListQuery(params: ListParams) {
  return queryOptions({
    queryKey: ["papers", "list", params],
    queryFn: ({ signal }) => getJson<PaperListResponse>(listUrl(params), { signal, accept: false }),
  });
}
