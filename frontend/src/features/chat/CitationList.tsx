import type { Citation } from "../../shared/api/types";
import { paperPath, toInternalPaperHref } from "../../shared/lib/arxiv";
import { isSameOriginPathname } from "../../shared/lib/safeUrl";
import styles from "./CitationList.module.css";

function resolveCitationHref(citation: Citation): string {
  const url = citation.url.trim();
  if (/^https?:\/\//i.test(url)) {
    return toInternalPaperHref(url) ?? url;
  }
  if (url.startsWith("/") && !url.startsWith("//")) {
    try {
      const parsed = new URL(url, window.location.origin);
      if (parsed.origin === window.location.origin && isSameOriginPathname(parsed.pathname)) {
        return `${parsed.pathname}${parsed.search}${parsed.hash}`;
      }
    } catch {
      return paperPath(citation.arxiv_id);
    }
  }
  return paperPath(citation.arxiv_id);
}

function citationTitle(citation: Citation): string {
  return citation.section_title ? `${citation.title} · ${citation.section_title}` : citation.title;
}

export function CitationList({ citations }: { citations: Citation[] }) {
  if (!citations.length) {
    return null;
  }
  const ordered = citations.toSorted((a, b) => Number(b.in_answer) - Number(a.in_answer));

  return (
    <div className={styles.citations}>
      <span className={styles.label} aria-hidden="true">
        출처
      </span>
      <ul className={styles.list} aria-label="답변 출처">
        {ordered.map((citation, index) => (
          <li key={`${citation.arxiv_id}-${citation.chunk_id ?? "paper"}-${index}`}>
            <a
              className={citation.in_answer ? `${styles.citation} ${styles.cited}` : styles.citation}
              href={resolveCitationHref(citation)}
              target="_blank"
              rel="noopener noreferrer"
              title={citationTitle(citation)}
            >
              <span className={styles.title}>{citation.title}</span>
              {citation.section_title ? (
                <span className={styles.section}>{citation.section_title}</span>
              ) : null}
              {citation.in_answer ? null : <span className={styles.visuallyHidden}> (검색 참고 자료)</span>}
            </a>
          </li>
        ))}
      </ul>
    </div>
  );
}
