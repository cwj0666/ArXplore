import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { useNavigate, useSearchParams } from "react-router";

import type { SearchMode, SortOption } from "../../shared/api/types";
import { paperListQuery } from "./api";
import { buildListSearchParams, type ListParams, readListParams } from "./listParams";

export function useListPageState() {
  const navigate = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams();
  const params = readListParams(searchParams);
  const [queryInput, setQueryInput] = useState(params.q);
  const [syncedQuery, setSyncedQuery] = useState(params.q);
  const listQuery = useQuery(paperListQuery(params));

  if (syncedQuery !== params.q) {
    setSyncedQuery(params.q);
    setQueryInput(params.q);
  }

  const updateParams = (patch: Partial<ListParams>, options: { replace?: boolean } = {}) => {
    const next = buildListSearchParams({ ...params, ...patch });
    if (next.toString() !== searchParams.toString()) {
      setSearchParams(next, options.replace ? { replace: true } : {});
    }
  };

  const submitSearch = () => {
    const trimmed = queryInput.trim();
    if (params.mode === "search") {
      updateParams({ q: trimmed, page: 1 });
      return;
    }
    if (!trimmed) {
      return;
    }
    void navigate(`/papers/assistant/?q=${encodeURIComponent(trimmed)}`);
  };

  const changeSort = (sort: SortOption) => {
    updateParams({ sort, page: 1 });
  };

  const changeMode = (mode: SearchMode) => {
    updateParams({ mode }, { replace: true });
  };

  return { params, queryInput, setQueryInput, listQuery, submitSearch, changeSort, changeMode };
}
