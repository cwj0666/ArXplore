import { Link } from "react-router-dom";

interface ListPaginationProps {
  page: number;
  totalPages: number;
  buildHref: (page: number) => string;
}

function clampPage(page: number, totalPages: number): number {
  if (page < 1) {
    return 1;
  }
  if (page > totalPages) {
    return totalPages;
  }
  return page;
}

export function ListPagination({ page, totalPages, buildHref }: ListPaginationProps) {
  const prevPage = clampPage(page - 1, totalPages);
  const nextPage = clampPage(page + 1, totalPages);
  const jumpPrev10 = clampPage(page - 10, totalPages);
  const jumpNext10 = clampPage(page + 10, totalPages);

  const canGoPrev = page > 1;
  const canGoNext = page < totalPages;

  return (
    <nav className="pagination" aria-label="페이지 이동">
      {canGoPrev ? (
        <>
          <Link className="arrow" to={buildHref(jumpPrev10)} aria-label="10페이지 앞으로">
            &laquo;
          </Link>
          <Link className="arrow" to={buildHref(prevPage)} aria-label="이전 페이지">
            &lsaquo;
          </Link>
        </>
      ) : (
        <>
          <span className="arrow disabled" aria-hidden="true">&laquo;</span>
          <span className="arrow disabled" aria-hidden="true">&lsaquo;</span>
        </>
      )}

      <span className="page-info" aria-current="page">
        {page} / {totalPages}
      </span>

      {canGoNext ? (
        <>
          <Link className="arrow" to={buildHref(nextPage)} aria-label="다음 페이지">
            &rsaquo;
          </Link>
          <Link className="arrow" to={buildHref(jumpNext10)} aria-label="10페이지 뒤로">
            &raquo;
          </Link>
        </>
      ) : (
        <>
          <span className="arrow disabled" aria-hidden="true">&rsaquo;</span>
          <span className="arrow disabled" aria-hidden="true">&raquo;</span>
        </>
      )}
    </nav>
  );
}
