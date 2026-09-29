import { streamChat } from "../../shared/api/sse";
import type { ChatTransport } from "../chat/useChatStream";

export function createPaperChatTransport(arxivId: string): ChatTransport {
  const endpoint = `/papers/${encodeURIComponent(arxivId)}/chat/stream/`;
  return (request, signal, handlers) => streamChat({ endpoint, body: request, signal, ...handlers });
}
