export type ChatRole = "assistant" | "user";

export interface ChatMessage {
  role: ChatRole;
  content: string;
}

export interface PaperDetail {
  arxiv_id: string;
  title: string;
  authors: string[] | string;
  abstract: string;
  published_at: string;
  upvotes?: number | null;
  pdf_url: string;
  is_favorited?: boolean;
  related_papers?: RelatedPaper[];
}

export interface RelatedPaper {
  arxiv_id: string;
  title: string;
  authors: string[] | string;
  abstract: string;
  published_at: string | null;
  upvotes?: number | null;
  pdf_url?: string | null;
  is_favorited?: boolean;
  source?: "local" | "arxiv" | string;
  relation_score?: number;
}

export interface DetailResponse {
  paper?: PaperDetail;
  error?: string;
}

export interface AnalysisResponse {
  overview?: string;
  key_findings?: string[];
  cached?: boolean;
  error?: string;
}

export interface SummaryResponse {
  summary?: string;
  cached?: boolean;
  model?: string;
  error?: string;
}

export interface ChatResponse {
  answer?: string;
  error?: string;
}

export type AiAccessReason = "login" | "api_key";

export type AiSectionState =
  | { status: "idle" }
  | { status: "loading" }
  | { status: "ready" }
  | { status: "denied"; reason: AiAccessReason }
  | { status: "error"; message: string }
  | { status: "cancelled" };
