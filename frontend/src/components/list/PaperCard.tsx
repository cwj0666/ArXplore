import type { KeyboardEvent, MouseEvent } from "react";
import { useNavigate } from "react-router-dom";

import { toSafeHttpUrl } from "../../helpers/safeUrl";
import type { PaperListItem } from "../../pages/list/listTypes";

function truncateText(text: string | undefined, max: number): string {
  if (!text) {
    return "";
  }
  return text.length > max ? `${text.slice(0, max)}...` : text;
}

function getPublishedDate(value: string | null | undefined): string {
  if (!value) {
    return "";
  }
  return value.slice(0, 10);
}

interface PaperCardProps {
  paper: PaperListItem;
}

export function PaperCard({ paper }: PaperCardProps) {
  const navigate = useNavigate();
  const detailLink = `/papers/${encodeURIComponent(paper.arxiv_id)}/`;
  const pdfLink = toSafeHttpUrl(paper.pdf_url, `https://arxiv.org/abs/${paper.arxiv_id}`);

  const handleTitleClick = (event: MouseEvent<HTMLAnchorElement>) => {
    event.stopPropagation();
  };

  const handleCardClick = (event: MouseEvent<HTMLElement>) => {
    if (event.metaKey || event.ctrlKey) {
      window.open(detailLink, "_blank", "noopener");
      return;
    }
    navigate(detailLink);
  };

  const handleCardKeyDown = (event: KeyboardEvent<HTMLElement>) => {
    if (event.target !== event.currentTarget) {
      return;
    }
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      navigate(detailLink);
    }
  };

  return (
    <article
      className="paper-card"
      role="link"
      tabIndex={0}
      aria-label={`${paper.title} 상세 보기`}
      onClick={handleCardClick}
      onKeyDown={handleCardKeyDown}
    >
      <div className="paper-card-top">
        <div className="paper-title">
          <a href={pdfLink} target="_blank" rel="noreferrer" onClick={handleTitleClick} data-tooltip="논문원본 바로가기">
            {truncateText(paper.title, 65)}
          </a>
        </div>
      </div>
      <div className="paper-abstract-wrapper">
        <div className="paper-abstract">{truncateText(paper.abstract, 200)}</div>
      </div>
      <div className="paper-meta paper-meta-row">
        <div className="paper-meta-item">
          <svg
            className="paper-meta-icon-muted"
            width="14"
            height="14"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            aria-hidden="true"
          >
            <rect x="3" y="4" width="18" height="18" rx="2" ry="2" />
            <line x1="16" y1="2" x2="16" y2="6" />
            <line x1="8" y1="2" x2="8" y2="6" />
            <line x1="3" y1="10" x2="21" y2="10" />
          </svg>
          <span>{getPublishedDate(paper.published_at)}</span>
        </div>
        <div className="paper-meta-item">
          <svg
            className="paper-meta-icon-heart"
            width="14"
            height="14"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            aria-hidden="true"
          >
            <path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z" />
          </svg>
          <span>{paper.upvotes ?? 0}</span>
        </div>
      </div>
    </article>
  );
}
