export type SearchMode = "search" | "ai";
export type SortOption = "latest" | "upvotes";

export interface PaperListItem {
  arxiv_id: string;
  title: string;
  abstract: string;
  authors?: string[];
  published_at?: string | null;
  upvotes?: number | null;
  pdf_url?: string | null;
}

export interface PaperListResponse {
  items: PaperListItem[];
  page: number;
  page_size: number;
  total_items: number;
  total_pages: number;
  query: string;
  sort: SortOption;
  mode?: SearchMode;
  error?: string;
}
