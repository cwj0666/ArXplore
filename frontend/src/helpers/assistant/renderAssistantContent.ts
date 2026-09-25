function escapeHtml(value: string): string {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function normalizeArxivId(value: string): string {
  const normalized = String(value).trim().replace(/\.pdf$/i, "");
  const match = normalized.match(/^(.*)v(\d+)$/i);
  if (!match) {
    return normalized;
  }

  return match[1];
}

export function toInternalPaperHref(url: string): string | null {
  const match = url.match(
    /^https?:\/\/(?:www\.)?arxiv\.org\/(?:abs|pdf)\/([^/?#]+?)(?:\.pdf)?(?:[?#].*)?$/i,
  );

  if (!match) {
    return null;
  }

  return `/papers/${encodeURIComponent(normalizeArxivId(match[1]))}/`;
}

function renderLink(label: string, escapedUrl: string): string {
  const url = escapedUrl.replace(/&amp;/g, "&");
  const href = toInternalPaperHref(url) ?? escapedUrl;
  return `<a href="${href}" target="_blank" rel="noopener noreferrer">${label}</a>`;
}

function renderInlineText(escaped: string): string {
  const links: string[] = [];
  const withPlaceholders = escaped.replace(/\u0000/g, "").replace(
    /\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g,
    (_, label: string, url: string) => {
      links.push(renderLink(label.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>"), url));
      return `\u0000${links.length - 1}\u0000`;
    },
  );
  return withPlaceholders
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/\u0000(\d+)\u0000/g, (_, index: string) => links[Number(index)] ?? "");
}

function renderInlineMarkdown(text: string): string {
  return text
    .split(/(`[^`\n]+`)/)
    .map((segment, index) => {
      if (index % 2 === 1) {
        return `<code>${escapeHtml(segment.slice(1, -1))}</code>`;
      }
      return renderInlineText(escapeHtml(segment));
    })
    .join("");
}

const FENCE_PATTERN = /^\s*(```|~~~)/;
const HEADING_PATTERN = /^(#{1,3})\s+(.*)$/;
const BULLET_PATTERN = /^[-*+]\s+(.*)$/;
const ORDERED_PATTERN = /^(\d{1,9})[.)]\s+(.*)$/;

export function renderAssistantContent(text: string): string {
  const lines = String(text || "").split(/\r\n|\r|\n/);
  const parts: string[] = [];
  let list: { kind: "ul" | "ol"; start: number; items: string[] } | null = null;
  let codeLines: string[] | null = null;
  let codeFence = "";

  const flushList = () => {
    if (!list) {
      return;
    }
    const startAttr = list.kind === "ol" && list.start !== 1 ? ` start="${list.start}"` : "";
    parts.push(`<${list.kind}${startAttr}>${list.items.join("")}</${list.kind}>`);
    list = null;
  };

  const pushListItem = (kind: "ul" | "ol", start: number, content: string) => {
    if (!list || list.kind !== kind) {
      flushList();
      list = { kind, start, items: [] };
    }
    list.items.push(`<li>${renderInlineMarkdown(content)}</li>`);
  };

  const flushCode = () => {
    if (codeLines === null) {
      return;
    }
    parts.push(`<pre><code>${escapeHtml(codeLines.join("\n"))}</code></pre>`);
    codeLines = null;
    codeFence = "";
  };

  for (const rawLine of lines) {
    if (codeLines !== null) {
      if (rawLine.trim().startsWith(codeFence)) {
        flushCode();
      } else {
        codeLines.push(rawLine);
      }
      continue;
    }

    const fence = rawLine.match(FENCE_PATTERN);
    if (fence) {
      flushList();
      codeLines = [];
      codeFence = fence[1];
      continue;
    }

    const line = rawLine.trim();
    if (!line) {
      flushList();
      continue;
    }

    const heading = line.match(HEADING_PATTERN);
    if (heading) {
      flushList();
      const level = heading[1].length + 2;
      parts.push(`<h${level}>${renderInlineMarkdown(heading[2])}</h${level}>`);
      continue;
    }

    const bullet = line.match(BULLET_PATTERN);
    if (bullet) {
      pushListItem("ul", 1, bullet[1]);
      continue;
    }

    const ordered = line.match(ORDERED_PATTERN);
    if (ordered) {
      pushListItem("ol", Number(ordered[1]), ordered[2]);
      continue;
    }

    flushList();
    parts.push(`<p>${renderInlineMarkdown(line)}</p>`);
  }

  flushList();
  flushCode();
  return parts.join("");
}
