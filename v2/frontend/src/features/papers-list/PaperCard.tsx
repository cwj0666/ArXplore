import { Link } from "react-router";

import type { PaperListItem } from "../../shared/api/types";
import { paperPath } from "../../shared/lib/arxiv";
import { toSafeHttpUrl } from "../../shared/lib/safeUrl";
import { datePart, truncateText } from "../../shared/lib/text";
import { CalendarIcon, HeartIcon } from "../../shared/ui/icons";
import styles from "./PaperCard.module.css";

export function PaperCard({ paper }: { paper: PaperListItem }) {
  const pdfLink = toSafeHttpUrl(paper.pdf_url, `https://arxiv.org/abs/${paper.arxiv_id}`);

  return (
    <article className={styles.card}>
      <div className={styles.top}>
        <div className={styles.title}>
          <a className={styles.pdfLink} href={pdfLink} target="_blank" rel="noreferrer">
            {truncateText(paper.title, 65)}
          </a>
        </div>
      </div>
      <Link
        className={styles.detailLink}
        to={paperPath(paper.arxiv_id)}
        aria-label={`${paper.title} 상세 보기`}
      >
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
      </Link>
    </article>
  );
}
