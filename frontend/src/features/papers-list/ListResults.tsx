import type { UseQueryResult } from "@tanstack/react-query";

import type { PaperListResponse, SortOption } from "../../shared/api/types";
import { buildListHref, type ListParams, normalizeSort } from "./listParams";
import { ListPagination } from "./ListPagination";
import styles from "./ListResults.module.css";
import { PaperCard } from "./PaperCard";

const LIST_ERROR_FALLBACK = "목록을 불러오는 중 오류가 발생했습니다.";

const SORT_OPTIONS: { value: SortOption; label: string }[] = [
  { value: "latest", label: "최신순" },
  { value: "upvotes", label: "추천순" },
];

interface ListResultsProps {
  params: ListParams;
  listQuery: UseQueryResult<PaperListResponse>;
  onSortChange: (sort: SortOption) => void;
}

export function ListResults({ params, listQuery, onSortChange }: ListResultsProps) {
  const { data, error } = listQuery;

  if (!data) {
    return error ? (
      <div className={styles.noPapers} role="alert">
        {error.message || LIST_ERROR_FALLBACK}
      </div>
    ) : (
      <div className={styles.noPapers} role="status">
        불러오는 중...
      </div>
    );
  }

  if (data.total_items === 0) {
    return (
      <div className={styles.noPapers}>
        {data.query ? "검색 결과가 없습니다." : "수집된 논문이 없습니다."}
      </div>
    );
  }

  return (
    <>
      <div className={styles.resultCount}>
        <span>
          총 {data.total_items}개 논문 — {data.page} / {data.total_pages} 페이지
        </span>
        <select
          aria-label="정렬"
          value={params.sort}
          onChange={(event) => {
            onSortChange(normalizeSort(event.target.value));
          }}
        >
          {SORT_OPTIONS.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
      </div>

      <section className={styles.grid} aria-label="논문 목록">
        {data.items.map((paper) => (
          <PaperCard key={paper.arxiv_id} paper={paper} />
        ))}
      </section>

      <ListPagination
        page={data.page}
        totalPages={Math.max(1, data.total_pages)}
        buildHref={(targetPage) => buildListHref({ ...params, page: targetPage })}
      />
    </>
  );
}
