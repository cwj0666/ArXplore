import type { components, operations } from "./schema";

type Schemas = components["schemas"];

type JsonResponse<Operation extends keyof operations> = operations[Operation]["responses"] extends {
  200: { content: { "application/json": infer Body } };
}
  ? Body
  : never;

type JsonRequest<Operation extends keyof operations> = operations[Operation] extends {
  requestBody: { content: { "application/json": infer Body } };
}
  ? Body
  : never;

export type BootstrapPayload = JsonResponse<"bootstrap">;
export type PaperListResponse = JsonResponse<"listPapers">;
export type PaperDetailResponse = JsonResponse<"getPaperDetail">;
export type AnalysisResponse = JsonResponse<"analyzePaper">;
export type SummaryResponse = JsonResponse<"summarizePaper">;
export type SummaryRequest = JsonRequest<"summarizePaper">;
export type ChatRequest = JsonRequest<"streamPaperChat">;

export type PaperListItem = Schemas["PaperListItem"];
export type SearchMode = PaperListResponse["mode"];
export type SortOption = PaperListResponse["sort"];
export type PaperDetail = Schemas["PaperDetail"];
export type RelatedPaper = Schemas["RelatedPaperItem"];
export type ChatTurn = Schemas["ChatHistoryMessage"];
export type Citation = Schemas["CitationItem"];
