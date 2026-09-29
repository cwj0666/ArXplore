import { queryOptions } from "@tanstack/react-query";

import { getJson, postJson } from "../../shared/api/http";
import type {
  AnalysisResponse,
  PaperDetailResponse,
  SummaryRequest,
  SummaryResponse,
} from "../../shared/api/types";

function paperEndpoint(arxivId: string, suffix: string): string {
  return `/papers/${encodeURIComponent(arxivId)}/${suffix}`;
}

export function paperDetailQuery(arxivId: string) {
  return queryOptions({
    queryKey: ["papers", "detail", arxivId],
    queryFn: ({ signal }) => getJson<PaperDetailResponse>(paperEndpoint(arxivId, "detail.json"), { signal }),
  });
}

export function requestAnalysis(arxivId: string, signal: AbortSignal): Promise<AnalysisResponse> {
  return postJson<AnalysisResponse>(paperEndpoint(arxivId, "analyze/"), undefined, signal);
}

export function requestSummary(
  arxivId: string,
  model: string,
  signal: AbortSignal,
): Promise<SummaryResponse> {
  const body: SummaryRequest = { model };
  return postJson<SummaryResponse>(paperEndpoint(arxivId, "summary/"), body, signal);
}
