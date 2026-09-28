const ARXIV_LINK_PATTERN = /^https?:\/\/(?:www\.)?arxiv\.org\/(?:abs|pdf)\/([^/?#]+?)(?:\.pdf)?(?:[?#].*)?$/i;

export function paperPath(arxivId: string): string {
  return `/papers/${encodeURIComponent(arxivId)}/`;
}

export function arxivAbsUrl(arxivId: string): string {
  return `https://arxiv.org/abs/${encodeURIComponent(arxivId)}`;
}

export function normalizeArxivId(value: string): string {
  const normalized = value.trim().replace(/\.pdf$/i, "");
  const match = /^(.*)v(\d+)$/i.exec(normalized);
  return match?.[1] ?? normalized;
}

export function toInternalPaperHref(url: string): string | null {
  const id = ARXIV_LINK_PATTERN.exec(url)?.[1];
  return id === undefined ? null : paperPath(normalizeArxivId(id));
}
