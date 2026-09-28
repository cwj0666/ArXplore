import { Link } from "react-router";

import styles from "./ListPagination.module.css";

interface ListPaginationProps {
  page: number;
  totalPages: number;
  buildHref: (page: number) => string;
}

function clampPage(page: number, totalPages: number): number {
  return Math.min(Math.max(page, 1), totalPages);
}

const arrowClass = styles.arrow;
const disabledArrowClass = `${styles.arrow} ${styles.disabled}`;

export function ListPagination({ page, totalPages, buildHref }: ListPaginationProps) {
  const canGoPrev = page > 1;
  const canGoNext = page < totalPages;

  return (
    <nav className={styles.pagination} aria-label="페이지 이동">
      {canGoPrev ? (
        <>
          <Link
            className={arrowClass}
            to={buildHref(clampPage(page - 10, totalPages))}
            aria-label="10페이지 앞으로"
          >
            &laquo;
          </Link>
          <Link
            className={arrowClass}
            to={buildHref(clampPage(page - 1, totalPages))}
            aria-label="이전 페이지"
          >
            &lsaquo;
          </Link>
        </>
      ) : (
        <>
          <span className={disabledArrowClass} aria-hidden="true">
            &laquo;
          </span>
          <span className={disabledArrowClass} aria-hidden="true">
            &lsaquo;
          </span>
        </>
      )}

      <span className={styles.pageInfo} aria-current="page">
        {page} / {totalPages}
      </span>

      {canGoNext ? (
        <>
          <Link
            className={arrowClass}
            to={buildHref(clampPage(page + 1, totalPages))}
            aria-label="다음 페이지"
          >
            &rsaquo;
          </Link>
          <Link
            className={arrowClass}
            to={buildHref(clampPage(page + 10, totalPages))}
            aria-label="10페이지 뒤로"
          >
            &raquo;
          </Link>
        </>
      ) : (
        <>
          <span className={disabledArrowClass} aria-hidden="true">
            &rsaquo;
          </span>
          <span className={disabledArrowClass} aria-hidden="true">
            &raquo;
          </span>
        </>
      )}
    </nav>
  );
}
