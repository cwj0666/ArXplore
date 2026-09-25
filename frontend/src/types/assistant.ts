export type AssistantRole = "user" | "assistant";

export type AssistantRenderableRole = AssistantRole | "loading";

export interface AssistantChatMessage {
  role: AssistantRole;
  content: string;
}

export interface AssistantDisplayMessage extends AssistantChatMessage {
  isNotice?: boolean;
}

export interface AssistantChatRequest {
  message: string;
  history: AssistantChatMessage[];
}

export interface AssistantChatResponse {
  answer?: string;
  error?: string;
}
