import type { MouseEvent } from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";

import { toSafeHttpUrl } from "../../helpers/safeUrl";

interface DetailTopBarProps {
  pdfUrl: string;
  summaryLoading: boolean;
  summaryLabel: string;
  showSummaryAction: boolean;
  onViewPdf: () => void;
  onGenerateSummary: () => void;
}

export function DetailTopBar({
  pdfUrl,
  summaryLoading,
  summaryLabel,
  showSummaryAction,
  onViewPdf,
  onGenerateSummary,
}: DetailTopBarProps) {
  const location = useLocation();
  const navigate = useNavigate();

  const handleBack = (event: MouseEvent<HTMLAnchorElement>) => {
    if (location.key !== "default") {
      event.preventDefault();
      navigate(-1);
    }
  };

  return (
    <div className="topbar">
      <div className="topbar-left">
        <Link to="/" className="back-btn" onClick={handleBack}>
          뒤로가기
        </Link>
      </div>
      <div className="topbar-center">
        <Link to="/" className="topbar-logo">
          ArXplore
        </Link>
      </div>
      <div className="topbar-right">
        <button type="button" className="layout-ctrl-btn" onClick={onViewPdf}>
          <svg
            className="btn-icon"
            width="16"
            height="16"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            aria-hidden="true"
          >
            <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
            <line x1="12" y1="3" x2="12" y2="21" />
          </svg>
          PDF 분할 보기
        </button>

        <button
          type="button"
          className="layout-ctrl-btn"
          onClick={() => {
            const safePdfUrl = toSafeHttpUrl(pdfUrl);
            if (safePdfUrl) window.open(safePdfUrl, "_blank", "noopener");
          }}
        >
          <svg
            className="btn-icon"
            width="16"
            height="16"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            aria-hidden="true"
          >
            <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6" />
            <polyline points="15 3 21 3 21 9" />
            <line x1="10" y1="14" x2="21" y2="3" />
          </svg>
          PDF 전체 화면
        </button>

        {showSummaryAction && (
          <button
            type="button"
            className="pdf-btn"
            onClick={onGenerateSummary}
            disabled={summaryLoading}
            aria-haspopup="dialog"
          >
            {summaryLoading ? "생성 중..." : summaryLabel}
          </button>
        )}
      </div>
    </div>
  );
}
