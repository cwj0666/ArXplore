import type { ReactNode } from "react";
import { Link } from "react-router";

import type { RelatedPaper } from "../../shared/api/types";
import { arxivAbsUrl, paperPath } from "../../shared/lib/arxiv";
import { authorsToText, truncateText } from "../../shared/lib/text";
import cards from "./cards.module.css";
import styles from "./RelatedPapersCard.module.css";

function RelatedPaperLink({ paper, children }: { paper: RelatedPaper; children: ReactNode }) {
  if (paper.source === "arxiv") {
    return (
      <a className={styles.row} href={arxivAbsUrl(paper.arxiv_id)} target="_blank" rel="noreferrer">
        {children}
      </a>
    );
  }
  return (
    <Link className={styles.row} to={paperPath(paper.arxiv_id)}>
      {children}
    </Link>
  );
}

export function RelatedPapersCard({ papers }: { papers: RelatedPaper[] }) {
  if (papers.length === 0) {
    return null;
  }
  return (
    <section className={`${cards.card} ${styles.card}`}>
      <div className={cards.sectionTitle}>관련 논문</div>
      <div className={styles.list}>
        {papers.map((paper) => (
          <RelatedPaperLink key={`${paper.source}-${paper.arxiv_id}`} paper={paper}>
            <div className={styles.title}>{paper.title}</div>
            <div className={styles.meta}>
              <span>{paper.published_at?.slice(0, 10) ?? "-"}</span>
              <span>{authorsToText(paper.authors)}</span>
            </div>
            {paper.abstract ? <p className={styles.abstract}>{truncateText(paper.abstract, 180)}</p> : null}
          </RelatedPaperLink>
        ))}
      </div>
    </section>
  );
}
