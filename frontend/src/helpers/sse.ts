import type { Citation } from "../types/assistant";
import { getCsrfTokenFromCookie, readApiError } from "./http";

export type SseEvent =
  | { type: "chunk"; chunk: string }
  | { type: "citations"; citations: Citation[] }
  | { type: "error"; message: string }
  | { type: "done" };

/** 스트림 도중 서버가 `{"error": ...}` 이벤트를 보낸 경우. */
export class StreamEventError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "StreamEventError";
  }
}

const DEFAULT_STREAM_ERROR = "답변을 생성하는 중 오류가 발생했습니다.";

function normalizeCitation(value: unknown): Citation | null {
  if (!value || typeof value !== "object") {
    return null;
  }
  const record = value as Record<string, unknown>;
  const arxivId = typeof record.arxiv_id === "string" ? record.arxiv_id.trim() : "";
  if (!arxivId) {
    return null;
  }
  return {
    arxiv_id: arxivId,
    title: typeof record.title === "string" && record.title.trim() ? record.title : arxivId,
    url: typeof record.url === "string" ? record.url : "",
    section_title:
      typeof record.section_title === "string" && record.section_title.trim() ? record.section_title : null,
    chunk_id: typeof record.chunk_id === "number" && Number.isFinite(record.chunk_id) ? record.chunk_id : null,
    in_answer: record.in_answer === true,
  };
}

/** `data:` 필드 값 하나를 이벤트로 해석한다. 알 수 없는 형식은 null. */
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
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
    return null;
  }

  const record = parsed as Record<string, unknown>;
  if ("error" in record) {
    const message = typeof record.error === "string" && record.error ? record.error : DEFAULT_STREAM_ERROR;
    return { type: "error", message };
  }
  if (Array.isArray(record.citations)) {
    const citations = record.citations
      .map(normalizeCitation)
      .filter((citation): citation is Citation => citation !== null);
    return { type: "citations", citations };
  }
  if (typeof record.chunk === "string") {
    return { type: "chunk", chunk: record.chunk };
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

/** 임의로 잘린 텍스트 조각을 받아 완성된 줄 단위로 이벤트를 돌려준다. */
export function createSseParser() {
  let buffer = "";

  const drain = (final: boolean): SseEvent[] => {
    const lines = buffer.split(/\r\n|\r|\n/);
    buffer = final ? "" : lines.pop() ?? "";
    const events: SseEvent[] = [];
    for (const line of lines) {
      const event = parseSseLine(line);
      if (event) {
        events.push(event);
      }
    }
    return events;
  };

  return {
    push(text: string): SseEvent[] {
      buffer += text;
      return drain(false);
    },
    flush(): SseEvent[] {
      return drain(true);
    },
  };
}

export interface StreamChatParams {
  endpoint: string;
  body: unknown;
  signal: AbortSignal;
  onChunk: (chunk: string) => void;
  onCitations?: (citations: Citation[]) => void;
}

/**
 * SSE 채팅 스트림을 끝까지 읽는다.
 * 스트림 시작 전 non-2xx는 ApiError, 스트림 중 error 이벤트는 StreamEventError로 던진다.
 */
export async function streamChat({ endpoint, body, signal, onChunk, onCitations }: StreamChatParams): Promise<void> {
  const csrfToken = getCsrfTokenFromCookie();
  const response = await fetch(endpoint, {
    method: "POST",
    credentials: "same-origin",
    headers: {
      "Content-Type": "application/json",
      Accept: "text/event-stream",
      ...(csrfToken ? { "X-CSRFToken": csrfToken } : {}),
    },
    body: JSON.stringify(body),
    signal,
  });

  if (!response.ok) {
    throw await readApiError(response);
  }
  if (!response.body) {
    throw new Error("스트리밍을 지원하지 않습니다.");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  const parser = createSseParser();

  const cancelReader = async () => {
    try {
      await reader.cancel();
    } catch {
      // The stream may already be closed or errored.
    }
  };

  // Returns true once the stream is finished (DONE received).
  const handle = async (events: SseEvent[]): Promise<boolean> => {
    for (const event of events) {
      switch (event.type) {
        case "chunk":
          if (event.chunk) onChunk(event.chunk);
          break;
        case "citations":
          onCitations?.(event.citations);
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

  while (true) {
    const { done, value } = await reader.read();
    if (done) {
      await handle(parser.push(decoder.decode()).concat(parser.flush()));
      return;
    }
    if (await handle(parser.push(decoder.decode(value, { stream: true })))) {
      return;
    }
  }
}
