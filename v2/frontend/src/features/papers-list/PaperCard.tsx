import type { KeyboardEvent, MouseEvent } from "react";
import { useNavigate } from "react-router";

import type { PaperListItem } from "../../shared/api/types";
import { paperPath } from "../../shared/lib/arxiv";
import { toSafeHttpUrl } from "../../shared/lib/safeUrl";
import { datePart, truncateText } from "../../shared/lib/text";
import { CalendarIcon, HeartIcon } from "../../shared/ui/icons";
import styles from "./PaperCard.module.css";

export function PaperCard({ paper }: { paper: PaperListItem }) {
  const navigate = useNavigate();
  const detailLink = paperPath(paper.arxiv_id);
  const pdfLink = toSafeHttpUrl(paper.pdf_url, `https://arxiv.org/abs/${paper.arxiv_id}`);

  const handleCardClick = (event: MouseEvent<HTMLElement>) => {
    if (event.metaKey || event.ctrlKey) {
      window.open(detailLink, "_blank", "noopener");
      return;
    }
    void navigate(detailLink);
  };

  const handleCardKeyDown = (event: KeyboardEvent<HTMLElement>) => {
    if (event.target !== event.currentTarget) {
      return;
    }
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      void navigate(detailLink);
    }
  };

  return (
    <div
      className={styles.card}
      role="link"
      tabIndex={0}
      aria-label={`${paper.title} 상세 보기`}
      onClick={handleCardClick}
      onKeyDown={handleCardKeyDown}
    >
      <div className={styles.top}>
        <div className={styles.title}>
          <a
            href={pdfLink}
            target="_blank"
            rel="noreferrer"
            onClick={(event) => {
              event.stopPropagation();
            }}
          >
            {truncateText(paper.title, 65)}
          </a>
        </div>
      </div>
      <div className={styles.abstractWrapper}>
        <div className={styles.abstract}>{truncateText(paper.abstract, 200)}</div>
      </div>
      <div className={styles.meta}>
        <div className={styles.metaItem}>
          <CalendarIcon className={styles.iconMuted} />
          <span>{paper.published_at ? datePart(paper.published_at) : ""}</span>
        </div>
        <div className={styles.metaItem}>
          <HeartIcon className={styles.iconHeart} />
          <span>{paper.upvotes}</span>
        </div>
      </div>
    </div>
  );
}
