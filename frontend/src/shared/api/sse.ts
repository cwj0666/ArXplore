import { getCsrfTokenFromCookie, readApiError } from "./http";
import type { Citation } from "./types";

export type SseEvent =
  | { type: "chunk"; chunk: string }
  | { type: "citations"; citations: Citation[] }
  | { type: "error"; message: string }
  | { type: "done" };

export class StreamEventError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "StreamEventError";
  }
}

export const DEFAULT_STREAM_ERROR = "답변을 생성하는 중 오류가 발생했습니다.";
export const TRUNCATED_STREAM_ERROR = "응답이 완료되기 전에 연결이 끊겼습니다.";
export const STREAMING_UNSUPPORTED_ERROR = "스트리밍을 지원하지 않습니다.";

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function normalizeCitation(value: unknown): Citation | null {
  if (!isRecord(value)) {
    return null;
  }
  const arxivId = typeof value.arxiv_id === "string" ? value.arxiv_id.trim() : "";
  if (!arxivId) {
    return null;
  }
  const title = value.title;
  const url = value.url;
  const sectionTitle = value.section_title;
  const chunkId = value.chunk_id;
  return {
    arxiv_id: arxivId,
    title: typeof title === "string" && title.trim() ? title : arxivId,
    url: typeof url === "string" ? url : "",
    section_title: typeof sectionTitle === "string" && sectionTitle.trim() ? sectionTitle : null,
    chunk_id: typeof chunkId === "number" && Number.isFinite(chunkId) ? chunkId : null,
    in_answer: value.in_answer === true,
  };
}

export function parseSseData(data: string): SseEvent | null {
  if (data.trim() === "[DONE]") {
    return { type: "done" };
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(data);
  } catch {
    return null;
  }
  if (!isRecord(parsed)) {
    return null;
  }
  if ("error" in parsed) {
    const { error } = parsed;
    return { type: "error", message: typeof error === "string" && error ? error : DEFAULT_STREAM_ERROR };
  }
  const { citations, chunk } = parsed;
  if (Array.isArray(citations)) {
    return {
      type: "citations",
      citations: citations.map(normalizeCitation).filter((citation) => citation !== null),
    };
  }
  if (typeof chunk === "string") {
    return { type: "chunk", chunk };
  }
  return null;
}

function parseSseLine(line: string): SseEvent | null {
  if (!line.startsWith("data:")) {
    return null;
  }
  const value = line.slice(5);
  return parseSseData(value.startsWith(" ") ? value.slice(1) : value);
}

export interface SseParser {
  push(text: string): SseEvent[];
  flush(): SseEvent[];
}

export function createSseParser(): SseParser {
  let buffer = "";

  const drain = (final: boolean): SseEvent[] => {
    const lines = buffer.split(/\r\n|\r|\n/);
    buffer = final ? "" : (lines.pop() ?? "");
    return lines.map(parseSseLine).filter((event) => event !== null);
  };

  return {
    push(text) {
      buffer += text;
      return drain(false);
    },
    flush() {
      return drain(true);
    },
  };
}

export interface StreamHandlers {
  onChunk: (chunk: string) => void;
  onCitations: (citations: Citation[]) => void;
}

export interface StreamChatParams extends StreamHandlers {
  endpoint: string;
  body: unknown;
  signal: AbortSignal;
}

export async function streamChat({
  endpoint,
  body,
  signal,
  onChunk,
  onCitations,
}: StreamChatParams): Promise<void> {
  const csrfToken = getCsrfTokenFromCookie();
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    Accept: "text/event-stream",
  };
  if (csrfToken) {
    headers["X-CSRFToken"] = csrfToken;
  }
  const response = await fetch(endpoint, {
    method: "POST",
    credentials: "same-origin",
    headers,
    body: JSON.stringify(body),
    signal,
  });

  if (!response.ok) {
    throw await readApiError(response);
  }
  if (!response.body) {
    throw new Error(STREAMING_UNSUPPORTED_ERROR);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  const parser = createSseParser();

  const cancelReader = async () => {
    try {
      await reader.cancel();
    } catch {
      return;
    }
  };

  const handle = async (events: SseEvent[]): Promise<boolean> => {
    for (const event of events) {
      switch (event.type) {
        case "chunk":
          if (event.chunk) {
            onChunk(event.chunk);
          }
          break;
        case "citations":
          onCitations(event.citations);
          break;
        case "error":
          await cancelReader();
          throw new StreamEventError(event.message);
        case "done":
          await cancelReader();
          return true;
      }
    }
    return false;
  };

  for (;;) {
    const { done, value } = await reader.read();
    if (done) {
      if (await handle(parser.push(decoder.decode()).concat(parser.flush()))) {
        return;
      }
      signal.throwIfAborted();
      throw new StreamEventError(TRUNCATED_STREAM_ERROR);
    }
    if (await handle(parser.push(decoder.decode(value, { stream: true })))) {
      return;
    }
  }
}
