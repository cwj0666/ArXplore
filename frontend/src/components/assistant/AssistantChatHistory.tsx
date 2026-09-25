import { forwardRef } from "react";

import { renderAssistantContent } from "../../helpers/assistant/renderAssistantContent";
import type { AssistantDisplayMessage } from "../../types/assistant";

interface AssistantChatHistoryProps {
  messages: AssistantDisplayMessage[];
  isSending: boolean;
  streamingContent?: string;
}

function handleChatLinkClick(event: React.MouseEvent<HTMLDivElement>) {
  const target = event.target as HTMLElement;
  const anchor = target.closest("a");
  if (!anchor) return;
  const href = anchor.getAttribute("href");
  if (!href) return;
  event.preventDefault();
  window.open(href, "_blank", "noopener,noreferrer");
}

export const AssistantChatHistory = forwardRef<
  HTMLDivElement,
  AssistantChatHistoryProps
>(function AssistantChatHistory({ messages, isSending, streamingContent }, ref) {
  return (
    <div className="assistant-chat-history" id="assistant-chat-history" ref={ref} onClick={handleChatLinkClick}>
      {messages.map((message, index) => {
        if (message.isNotice) {
          return (
            <div key={`notice-${index}`} className="assistant-message assistant-message-notice" role="status">
              <p>{message.content}</p>
            </div>
          );
        }

        if (message.role === "assistant") {
          return (
            <div
              key={`assistant-${index}`}
              className="assistant-message assistant-message-assistant"
              dangerouslySetInnerHTML={{
                __html: renderAssistantContent(message.content),
              }}
            />
          );
        }

        return (
          <div key={`user-${index}`} className="assistant-message assistant-message-user">
            <p>{message.content}</p>
          </div>
        );
      })}

      {isSending && streamingContent ? (
        <div
          className="assistant-message assistant-message-assistant"
          dangerouslySetInnerHTML={{
            __html: renderAssistantContent(streamingContent),
          }}
        />
      ) : isSending ? (
        <div className="assistant-message assistant-message-loading">
          <p>답변을 생성하는 중입니다...</p>
        </div>
      ) : null}
    </div>
  );
});
