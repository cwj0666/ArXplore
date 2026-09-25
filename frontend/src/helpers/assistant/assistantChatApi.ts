import type {
  AssistantChatRequest,
  AssistantChatResponse,
} from "../../types/assistant";
import { fetchJsonWithBody, getCsrfTokenFromCookie } from "../http";

export interface PostAssistantChatParams extends AssistantChatRequest {
  endpoint: string;
}

export async function postAssistantChat({
  endpoint,
  message,
  history,
}: PostAssistantChatParams): Promise<AssistantChatResponse> {
  return fetchJsonWithBody<AssistantChatResponse>(endpoint, "POST", { message, history });
}

export class AssistantRequestError extends Error {
  readonly status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "AssistantRequestError";
    this.status = status;
  }
}

async function readErrorMessage(response: Response): Promise<string> {
  try {
    const payload = (await response.json()) as { error?: unknown };
    if (payload && typeof payload.error === "string" && payload.error) {
      return payload.error;
    }
  } catch {
    // Non-JSON error bodies (e.g. proxy HTML pages) fall through to the status text.
  }
  const statusLabel = response.statusText ? `${response.status} ${response.statusText}` : `${response.status}`;
  return `요청이 실패했습니다. (HTTP ${statusLabel})`;
}

export interface StreamAssistantChatParams extends AssistantChatRequest {
  endpoint: string;
  signal: AbortSignal;
  onChunk: (chunk: string) => void;
}

export async function streamAssistantChat({
  endpoint,
  message,
  history,
  signal,
  onChunk,
}: StreamAssistantChatParams): Promise<void> {
  const csrfToken = getCsrfTokenFromCookie();
  const response = await fetch(endpoint, {
    method: "POST",
    credentials: "same-origin",
    headers: {
      "Content-Type": "application/json",
      ...(csrfToken ? { "X-CSRFToken": csrfToken } : {}),
    },
    body: JSON.stringify({ message, history }),
    signal,
  });

  if (!response.ok) {
    throw new AssistantRequestError(await readErrorMessage(response), response.status);
  }

  if (!response.body) {
    throw new Error("스트리밍을 지원하지 않습니다.");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";

    for (const line of lines) {
      if (!line.startsWith("data: ")) continue;
      const data = line.slice(6);
      if (data === "[DONE]") return;
      try {
        const parsed = JSON.parse(data) as { chunk?: string; error?: string };
        if (parsed.error) throw new Error(parsed.error);
        if (parsed.chunk) onChunk(parsed.chunk);
      } catch (e) {
        if (e instanceof SyntaxError) continue;
        throw e;
      }
    }
  }
}
