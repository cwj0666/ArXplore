import { type KeyboardEvent, useCallback, useEffect, useRef, useState } from "react";

import { ApiError } from "../../shared/api/http";
import { StreamEventError, streamChat } from "../../shared/api/sse";
import { isImeComposing } from "../../shared/lib/keyboard";
import { SendIcon, StopIcon } from "../../shared/ui/icons";
import { CitationList } from "../chat/CitationList";
import { MarkdownContent } from "../chat/MarkdownContent";
import { type ChatNotices, type ChatTransport, useChatStream } from "../chat/useChatStream";
import styles from "./AssistantChat.module.css";
import { useOpenLinksInNewTab } from "./useOpenLinksInNewTab";

const WELCOME_MESSAGE = "찾고 싶은 연구 주제나 기술 키워드를 질문해 주세요.";
const STREAM_ENDPOINT = "/papers/assistant/stream/";
const MAX_HISTORY_MESSAGES = 20;

const NOTICES: ChatNotices = {
  abortedWithPartial:
    "사용자의 요청으로 답변이 중단되었습니다. 이 답변은 이후 대화 맥락에 포함되지 않습니다.",
  aborted: "사용자의 요청으로 답변이 중단되었습니다.",
  empty: "답변을 생성할 수 없습니다.",
  describeError: (error) =>
    (error instanceof ApiError || error instanceof StreamEventError) && error.message
      ? error.message
      : "답변을 불러오는 중 오류가 발생했습니다.",
};

const transport: ChatTransport = (request, signal, handlers) =>
  streamChat({ endpoint: STREAM_ENDPOINT, body: request, signal, ...handlers });

export function AssistantChat({ initialQuery }: { initialQuery: string }) {
  const historyRef = useRef<HTMLDivElement | null>(null);
  const inputRef = useRef<HTMLInputElement | null>(null);
  const initialQuerySentRef = useRef(false);
  const [inputValue, setInputValue] = useState("");
  const focusInput = useCallback(() => {
    inputRef.current?.focus();
  }, []);
  const chat = useChatStream({
    welcome: WELCOME_MESSAGE,
    transport,
    notices: NOTICES,
    historyLimit: MAX_HISTORY_MESSAGES,
    onSettled: focusInput,
  });
  const { entries, streaming, isSending, send } = chat;
  useOpenLinksInNewTab(historyRef);

  useEffect(() => {
    const box = historyRef.current;
    if (box) {
      box.scrollTop = box.scrollHeight;
    }
  }, [entries, streaming]);

  useEffect(() => {
    if (!initialQuery.trim() || initialQuerySentRef.current) {
      return;
    }
    initialQuerySentRef.current = true;
    send(initialQuery);
  }, [initialQuery, send]);

  const sendInput = () => {
    if (send(inputValue)) {
      setInputValue("");
    }
  };

  const handleKeyDown = (event: KeyboardEvent<HTMLInputElement>) => {
    if (event.key !== "Enter" || event.shiftKey || isImeComposing(event)) {
      return;
    }
    event.preventDefault();
    if (!isSending) {
      sendInput();
    }
  };

  return (
    <div className={styles.shell}>
      <div
        className={styles.history}
        id="assistant-chat-history"
        ref={historyRef}
        role="log"
        aria-live="polite"
        aria-busy={isSending}
      >
        {entries.map((entry) => {
          if (entry.role === "notice") {
            return (
              <div key={entry.id} className={`${styles.message} ${styles.notice}`} role="status">
                <p>{entry.content}</p>
              </div>
            );
          }
          if (entry.role === "assistant") {
            return (
              <div key={entry.id} className={`${styles.message} ${styles.assistant}`}>
                <MarkdownContent content={entry.content} />
                {entry.citations.length ? <CitationList citations={entry.citations} /> : null}
              </div>
            );
          }
          return (
            <div key={entry.id} className={`${styles.message} ${styles.user}`}>
              <p>{entry.content}</p>
            </div>
          );
        })}
        {streaming ? (
          <div className={`${styles.message} ${styles.assistant}`}>
            <MarkdownContent content={streaming} />
          </div>
        ) : null}
        {streaming === "" ? (
          <div className={`${styles.message} ${styles.loading}`}>
            <p>답변을 생성하는 중입니다...</p>
          </div>
        ) : null}
      </div>
      <div className={styles.composer}>
        <input
          id="assistant-chat-input"
          type="text"
          aria-label="AI 어시스턴트에게 질문하기"
          placeholder="연구 주제, 트렌드, 논문을 질문해 보세요."
          ref={inputRef}
          value={inputValue}
          onChange={(event) => {
            setInputValue(event.target.value);
          }}
          onKeyDown={handleKeyDown}
        />
        {isSending ? (
          <button
            type="button"
            id="assistant-stop-btn"
            className={styles.stopBtn}
            onClick={chat.stop}
            aria-label="답변 중단"
          >
            <StopIcon />
          </button>
        ) : (
          <button type="button" id="assistant-send-btn" onClick={sendInput} aria-label="전송">
            <SendIcon />
          </button>
        )}
      </div>
    </div>
  );
}
