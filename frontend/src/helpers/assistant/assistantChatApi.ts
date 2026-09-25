import type {
  AssistantChatRequest,
  AssistantChatResponse,
  Citation,
} from "../../types/assistant";
import { requestJsonWithBody } from "../http";
import { streamChat } from "../sse";

export interface PostAssistantChatParams extends AssistantChatRequest {
  endpoint: string;
}

export async function postAssistantChat({
  endpoint,
  message,
  history,
}: PostAssistantChatParams): Promise<AssistantChatResponse> {
  return requestJsonWithBody<AssistantChatResponse>(endpoint, "POST", { message, history });
}

export interface StreamAssistantChatParams extends AssistantChatRequest {
  endpoint: string;
  signal: AbortSignal;
  onChunk: (chunk: string) => void;
  onCitations?: (citations: Citation[]) => void;
}

export async function streamAssistantChat({
  endpoint,
  message,
  history,
  signal,
  onChunk,
  onCitations,
}: StreamAssistantChatParams): Promise<void> {
  return streamChat({ endpoint, body: { message, history }, signal, onChunk, onCitations });
}
