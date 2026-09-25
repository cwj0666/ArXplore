export type AssistantRole = "user" | "assistant";

export type AssistantRenderableRole = AssistantRole | "loading";

export interface AssistantChatMessage {
  role: AssistantRole;
  content: string;
}

export interface Citation {
  arxiv_id: string;
  title: string;
  url: string;
  section_title: string | null;
  chunk_id: number | null;
  in_answer: boolean;
}

export interface AssistantDisplayMessage extends AssistantChatMessage {
  isNotice?: boolean;
  citations?: Citation[];
}

export interface AssistantChatRequest {
  message: string;
  history: AssistantChatMessage[];
}

export interface AssistantChatResponse {
  answer?: string;
  error?: string;
}
