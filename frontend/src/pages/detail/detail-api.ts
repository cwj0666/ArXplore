import { fetchJson, fetchJsonWithBody } from "../../helpers/http";
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


export async function fetchPaperDetail(arxivId: string): Promise<DetailResponse> {
  return fetchJson<DetailResponse>(buildPaperPath(arxivId, "detail.json"), {
    method: "GET",
    headers: {
      Accept: "application/json",
    },
  });
}


export async function fetchPaperAnalysis(arxivId: string): Promise<AnalysisResponse> {
  return fetchJsonWithBody<AnalysisResponse>(buildPaperPath(arxivId, "analyze/"), "POST");
}


export async function fetchPaperSummary(arxivId: string, model: string): Promise<SummaryResponse> {
  return fetchJsonWithBody<SummaryResponse>(buildPaperPath(arxivId, "summary/"), "POST", { model });
}


export async function postPaperChat(
  arxivId: string,
  message: string,
  history: ChatMessage[],
): Promise<ChatResponse> {
  return fetchJsonWithBody<ChatResponse>(buildPaperPath(arxivId, "chat/"), "POST", {
    message,
    history,
  });
}
