import { useMemo } from "react";

import { renderAssistantContent } from "../../helpers/assistant/renderAssistantContent";
import "./chat-content.css";

interface MarkdownContentProps {
  content: string;
  className?: string;
}

export function MarkdownContent({ content, className }: MarkdownContentProps) {
  const html = useMemo(() => renderAssistantContent(content), [content]);
  return (
    <div
      className={className ? `chat-markdown ${className}` : "chat-markdown"}
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
}
