import { ApiError, hasErrorPayload, requestJson, requestJsonWithBody } from "../../helpers/http";
import { streamChat } from "../../helpers/sse";
import type { Citation } from "../../types/assistant";
import type {
  AnalysisResponse,
  ChatMessage,
  ChatResponse,
  DetailResponse,
  SummaryResponse,
} from "./detail-types";


function buildPaperPath(arxivId: string, suffix: string): string {
  return `/papers/${encodeURIComponent(arxivId)}/${suffix}`;
}


export async function fetchPaperDetail(arxivId: string, signal?: AbortSignal): Promise<DetailResponse> {
  return requestJson<DetailResponse>(buildPaperPath(arxivId, "detail.json"), {
    method: "GET",
    headers: {
      Accept: "application/json",
    },
    signal,
  });
}


export async function fetchPaperAnalysis(arxivId: string, signal?: AbortSignal): Promise<AnalysisResponse> {
  return requestJsonWithBody<AnalysisResponse>(buildPaperPath(arxivId, "analyze/"), "POST", undefined, signal);
}


export async function fetchPaperSummary(
  arxivId: string,
  model: string,
  signal?: AbortSignal,
): Promise<SummaryResponse> {
  return requestJsonWithBody<SummaryResponse>(buildPaperPath(arxivId, "summary/"), "POST", { model }, signal);
}


export async function postPaperChat(
  arxivId: string,
  message: string,
  history: ChatMessage[],
  signal?: AbortSignal,
): Promise<ChatResponse> {
  return requestJsonWithBody<ChatResponse>(
    buildPaperPath(arxivId, "chat/"),
    "POST",
    { message, history },
    signal,
  );
}


export interface StreamPaperChatHandlers {
  onChunk: (chunk: string) => void;
  onCitations?: (citations: Citation[]) => void;
}


export async function streamPaperChat(
  arxivId: string,
  message: string,
  history: ChatMessage[],
  signal: AbortSignal,
  { onChunk, onCitations }: StreamPaperChatHandlers,
): Promise<void> {
  return streamChat({
    endpoint: buildPaperPath(arxivId, "chat/stream/"),
    body: { message, history },
    signal,
    onChunk,
    onCitations,
  });
}


/**
 * 스트리밍 엔드포인트가 없는 구버전 백엔드인지 판별한다.
 * 논문이 없을 때의 404는 `{ error }` JSON을 싣고, 라우트 자체가 없을 때는 JSON이 아니다.
 */
export function isPaperChatStreamUnavailable(error: unknown): boolean {
  return error instanceof ApiError && error.status === 404 && !hasErrorPayload(error);
}
