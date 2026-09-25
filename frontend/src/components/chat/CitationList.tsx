import { toInternalPaperHref } from "../../helpers/assistant/renderAssistantContent";
import type { Citation } from "../../types/assistant";
import "./chat-content.css";

interface CitationListProps {
  citations: Citation[];
}

function resolveCitationHref(citation: Citation): string {
  const url = citation.url.trim();
  if (/^https?:\/\//i.test(url)) {
    return toInternalPaperHref(url) ?? url;
  }
  if (url.startsWith("/") && !url.startsWith("//")) {
    return url;
  }
  return `/papers/${encodeURIComponent(citation.arxiv_id)}/`;
}

export function CitationList({ citations }: CitationListProps) {
  if (!citations.length) {
    return null;
  }

  const ordered = [...citations].sort((a, b) => Number(b.in_answer) - Number(a.in_answer));

  return (
    <div className="chat-citations">
      <span className="chat-citations-label" aria-hidden="true">
        출처
      </span>
      <ul className="chat-citation-list" aria-label="답변 출처">
        {ordered.map((citation, index) => (
          <li key={`${citation.arxiv_id}-${citation.chunk_id ?? "paper"}-${index}`}>
            <a
              className={citation.in_answer ? "chat-citation is-cited" : "chat-citation"}
              href={resolveCitationHref(citation)}
              target="_blank"
              rel="noopener noreferrer"
              title={citation.section_title ? `${citation.title} · ${citation.section_title}` : citation.title}
            >
              <span className="chat-citation-title">{citation.title}</span>
              {citation.section_title ? (
                <span className="chat-citation-section">{citation.section_title}</span>
              ) : null}
              {citation.in_answer ? null : <span className="chat-visually-hidden"> (검색 참고 자료)</span>}
            </a>
          </li>
        ))}
      </ul>
    </div>
  );
}
