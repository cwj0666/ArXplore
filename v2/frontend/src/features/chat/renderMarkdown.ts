import { toInternalPaperHref } from "../../shared/lib/arxiv";

function escapeHtml(value: string): string {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function renderLink(label: string, escapedUrl: string): string {
  const url = escapedUrl.replace(/&amp;/g, "&");
  const href = toInternalPaperHref(url) ?? escapedUrl;
  return `<a href="${href}" target="_blank" rel="noopener noreferrer">${label}</a>`;
}

const PLACEHOLDER_DELIMITER = "\u0000";

function renderInlineText(escaped: string): string {
  const links: string[] = [];
  const withPlaceholders = escaped
    .replaceAll(PLACEHOLDER_DELIMITER, "")
    .replace(/\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g, (_match, label: string, url: string) => {
      links.push(renderLink(label.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>"), url));
      return `${PLACEHOLDER_DELIMITER}${links.length - 1}${PLACEHOLDER_DELIMITER}`;
    });
  return withPlaceholders
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .split(PLACEHOLDER_DELIMITER)
    .map((segment, index) => (index % 2 === 1 ? (links[Number(segment)] ?? "") : segment))
    .join("");
}

function renderInlineMarkdown(text: string): string {
  return text
    .split(/(`[^`\n]+`)/)
    .map((segment, index) =>
      index % 2 === 1
        ? `<code>${escapeHtml(segment.slice(1, -1))}</code>`
        : renderInlineText(escapeHtml(segment)),
    )
    .join("");
}

const FENCE_PATTERN = /^\s*(```|~~~)/;
const HEADING_PATTERN = /^(#{1,3})\s+(.*)$/;
const BULLET_PATTERN = /^[-*+]\s+(.*)$/;
const ORDERED_PATTERN = /^(\d{1,9})[.)]\s+(.*)$/;

interface PendingList {
  kind: "ul" | "ol";
  start: number;
  items: string[];
}

export function renderMarkdownToHtml(text: string): string {
  const lines = text.split(/\r\n|\r|\n/);
  const parts: string[] = [];
  let list: PendingList | null = null;
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

  const pushListItem = (kind: PendingList["kind"], start: number, content: string) => {
    if (list?.kind !== kind) {
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

    const fence = FENCE_PATTERN.exec(rawLine);
    if (fence) {
      flushList();
      codeLines = [];
      codeFence = fence[1] ?? "";
      continue;
    }

    const line = rawLine.trim();
    if (!line) {
      flushList();
      continue;
    }

    const heading = HEADING_PATTERN.exec(line);
    if (heading) {
      flushList();
      const level = (heading[1] ?? "").length + 2;
      parts.push(`<h${level}>${renderInlineMarkdown(heading[2] ?? "")}</h${level}>`);
      continue;
    }

    const bullet = BULLET_PATTERN.exec(line);
    if (bullet) {
      pushListItem("ul", 1, bullet[1] ?? "");
      continue;
    }

    const ordered = ORDERED_PATTERN.exec(line);
    if (ordered) {
      pushListItem("ol", Number(ordered[1]), ordered[2] ?? "");
      continue;
    }

    flushList();
    parts.push(`<p>${renderInlineMarkdown(line)}</p>`);
  }

  flushList();
  flushCode();
  return parts.join("");
}
