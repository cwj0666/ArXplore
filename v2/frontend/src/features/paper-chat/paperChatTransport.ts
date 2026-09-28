import { ApiError, hasErrorPayload, postJson } from "../../shared/api/http";
import { streamChat } from "../../shared/api/sse";
import type { PaperChatResponse } from "../../shared/api/types";
import type { ChatTransport } from "../chat/useChatStream";

function paperChatEndpoint(arxivId: string, suffix: string): string {
  return `/papers/${encodeURIComponent(arxivId)}/${suffix}`;
}

function isStreamUnavailable(error: unknown): boolean {
  return error instanceof ApiError && error.status === 404 && !hasErrorPayload(error);
}

let streamAvailable = true;

export function createPaperChatTransport(arxivId: string): ChatTransport {
  return async (request, signal, handlers) => {
    if (streamAvailable) {
      try {
        await streamChat({
          endpoint: paperChatEndpoint(arxivId, "chat/stream/"),
          body: request,
          signal,
          ...handlers,
        });
        return;
      } catch (error) {
        if (!isStreamUnavailable(error)) {
          throw error;
        }
        streamAvailable = false;
      }
    }
    const data = await postJson<PaperChatResponse>(paperChatEndpoint(arxivId, "chat/"), request, signal);
    if (data.answer) {
      handlers.onChunk(data.answer);
    }
  };
}
