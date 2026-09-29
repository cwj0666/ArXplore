import type { Citation } from "../../shared/api/types";

export type ChatEntryRole = "user" | "assistant" | "notice";

export interface ChatEntry {
  id: string;
  role: ChatEntryRole;
  content: string;
  citations: Citation[];
}

const WELCOME_ENTRY_ID = "welcome";

let lastEntryNumber = 0;

export function createEntry(role: ChatEntryRole, content: string, citations: Citation[] = []): ChatEntry {
  lastEntryNumber += 1;
  return { id: `entry-${lastEntryNumber}`, role, content, citations };
}

export function welcomeEntries(welcome: string): ChatEntry[] {
  return [{ id: WELCOME_ENTRY_ID, role: "assistant", content: welcome, citations: [] }];
}
