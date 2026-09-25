import type { KeyboardEvent, RefObject } from "react";

import { isImeComposing } from "../../helpers/keyboard";

interface AssistantComposerProps {
  value: string;
  isSending?: boolean;
  inputRef?: RefObject<HTMLInputElement>;
  onChange: (nextValue: string) => void;
  onSend: () => void;
  onStop?: () => void;
}

export function AssistantComposer({
  value,
  isSending,
  inputRef,
  onChange,
  onSend,
  onStop,
}: AssistantComposerProps) {
  const handleKeyDown = (event: KeyboardEvent<HTMLInputElement>) => {
    if (event.key !== "Enter" || event.shiftKey || isImeComposing(event)) return;
    event.preventDefault();
    if (!isSending) onSend();
  };

  return (
    <div className="assistant-composer">
      <input
        id="assistant-chat-input"
        type="text"
        placeholder="연구 주제, 트렌드, 논문을 질문해 보세요."
        ref={inputRef}
        value={value}
        onChange={(event) => onChange(event.target.value)}
        onKeyDown={handleKeyDown}
      />
      {isSending ? (
        <button
          id="assistant-stop-btn"
          className="assistant-stop-btn"
          onClick={onStop}
          aria-label="답변 중단"
        >
          <svg width="20" height="20" viewBox="0 0 14 14" fill="currentColor" aria-hidden="true">
            <rect x="2" y="2" width="10" height="10" rx="2" />
          </svg>
        </button>
      ) : (
        <button id="assistant-send-btn" onClick={onSend} aria-label="전송">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
            <line x1="22" y1="2" x2="11" y2="13" />
            <polygon points="22 2 15 22 11 13 2 9 22 2" />
          </svg>
        </button>
      )}
    </div>
  );
}
