import type { ReactNode } from "react";
import { Link } from "react-router-dom";

import type { RelatedPaper } from "../../pages/detail/detail-types";

interface RelatedPapersCardProps {
  papers: RelatedPaper[];
}

function truncateText(text: string | undefined | null, max: number): string {
  if (!text) {
    return "";
  }
  return text.length > max ? `${text.slice(0, max)}...` : text;
}

function authorsToText(authors: RelatedPaper["authors"]): string {
  if (Array.isArray(authors)) {
    return authors.join(", ");
  }
  return authors;
}

function RelatedPaperLink({ paper, children }: { paper: RelatedPaper; children: ReactNode }) {
  if (paper.source === "arxiv") {
    return (
      <a
        className="related-paper-row"
        href={`https://arxiv.org/abs/${encodeURIComponent(paper.arxiv_id)}`}
        target="_blank"
        rel="noreferrer"
      >
        {children}
      </a>
    );
  }
  return (
    <Link className="related-paper-row" to={`/papers/${encodeURIComponent(paper.arxiv_id)}/`}>
      {children}
    </Link>
  );
}

export function RelatedPapersCard({ papers }: RelatedPapersCardProps) {
  if (papers.length === 0) {
    return null;
  }

  return (
    <section className="card related-papers-card">
      <div className="section-title">관련 논문</div>
      <div className="related-papers-list">
        {papers.map((paper) => (
          <RelatedPaperLink key={`${paper.source ?? "local"}-${paper.arxiv_id}`} paper={paper}>
            <div className="related-paper-title">{paper.title}</div>
            <div className="related-paper-meta">
              <span>{paper.published_at?.slice(0, 10) ?? "-"}</span>
              <span>{authorsToText(paper.authors)}</span>
            </div>
            {paper.abstract ? (
              <p className="related-paper-abstract">{truncateText(paper.abstract, 180)}</p>
            ) : null}
          </RelatedPaperLink>
        ))}
      </div>
    </section>
  );
}
