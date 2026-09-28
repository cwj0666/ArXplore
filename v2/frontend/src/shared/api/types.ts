export interface BootstrapPayload {
  default_summary_model: string;
  available_summary_models: string[];
}

export type SearchMode = "search" | "ai";
export type SortOption = "latest" | "upvotes";

export interface PaperSummaryFields {
  arxiv_id: string;
  title: string;
  authors: string[];
  abstract: string;
  published_at: string | null;
  upvotes: number;
  pdf_url: string | null;
}

export type PaperListItem = PaperSummaryFields;

export interface PaperListResponse {
  items: PaperListItem[];
  page: number;
  page_size: number;
  total_items: number;
  total_pages: number;
  query: string;
  sort: SortOption;
  mode: SearchMode;
}

export interface RelatedPaper extends PaperSummaryFields {
  source: string;
  relation_score: number;
}

export interface PaperDetail extends PaperSummaryFields {
  related_papers: RelatedPaper[];
}

export interface PaperDetailResponse {
  paper: PaperDetail;
}

export interface AnalysisResponse {
  overview: string | null;
  key_findings: string[];
  cached: boolean;
}

export interface SummaryResponse {
  summary: string | null;
  cached: boolean;
  model: string;
}

export type ChatRole = "user" | "assistant";

export interface ChatTurn {
  role: ChatRole;
  content: string;
}

export interface ChatRequest {
  message: string;
  history: ChatTurn[];
}

export interface Citation {
  arxiv_id: string;
  title: string;
  url: string;
  section_title: string | null;
  chunk_id: number | null;
  in_answer: boolean;
}

export interface PaperChatResponse {
  answer: string;
  citations: Citation[];
  retrieval_mode: string;
}
