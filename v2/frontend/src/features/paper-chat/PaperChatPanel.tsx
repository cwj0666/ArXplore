import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { ApiError } from "../../shared/api/http";
import { StreamEventError } from "../../shared/api/sse";
import { isImeComposing } from "../../shared/lib/keyboard";
import { ChatBubbleIcon } from "../../shared/ui/icons";
import { Spinner } from "../../shared/ui/Spinner";
import { CitationList } from "../chat/CitationList";
import { MarkdownContent } from "../chat/MarkdownContent";
import { type ChatNotices, useChatStream } from "../chat/useChatStream";
import styles from "./PaperChatPanel.module.css";
import { createPaperChatTransport } from "./paperChatTransport";
import { useFloatingPanel } from "./useFloatingPanel";

const WELCOME_MESSAGE = "이 논문에 대해 궁금한 점을 편하게 물어보세요!";

const NOTICES: ChatNotices = {
  abortedWithPartial: "답변이 중단되었습니다. 이 답변은 이후 대화 맥락에 포함되지 않습니다.",
  aborted: "답변이 중단되었습니다.",
  empty: "응답이 비어 있습니다.",
  describeError: (error) =>
    error instanceof ApiError || error instanceof StreamEventError
      ? `오류: ${error.message}`
      : "네트워크 오류가 발생했습니다.",
};

export function PaperChatPanel({ arxivId }: { arxivId: string }) {
  const panelRef = useRef<HTMLDivElement | null>(null);
  const messagesRef = useRef<HTMLDivElement | null>(null);
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [isOpen, setIsOpen] = useState(false);
  const [opacityPercent, setOpacityPercent] = useState(100);
  const [inputText, setInputText] = useState("");
  const transport = useMemo(() => createPaperChatTransport(arxivId), [arxivId]);
  const focusInput = useCallback(() => {
    inputRef.current?.focus();
  }, []);
  const chat = useChatStream({
    welcome: WELCOME_MESSAGE,
    transport,
    notices: NOTICES,
    onSettled: focusInput,
  });
  const floating = useFloatingPanel(panelRef);
  const { entries, streaming } = chat;

  useEffect(() => {
    const box = messagesRef.current;
    if (box) {
      box.scrollTop = box.scrollHeight;
    }
  }, [entries, streaming]);

  const sendMessage = () => {
    if (chat.send(inputText)) {
      setInputText("");
    }
  };

  return (
    <>
      <div
        ref={panelRef}
        id="chat-panel"
        role="region"
        aria-label="논문 AI 채팅"
        className={isOpen ? styles.panel : `${styles.panel} ${styles.hidden}`}
        style={{ opacity: opacityPercent / 100, ...floating.style }}
      >
        <div
          role="presentation"
          className={styles.resizeHandle}
          id="chat-resize-tl"
          onMouseDown={floating.startResize}
        />
        <div role="presentation" className={styles.header} id="chat-header" onMouseDown={floating.startDrag}>
          <h3>AI Chat</h3>
          <div
            role="presentation"
            className={styles.headerControls}
            onMouseDown={(event) => {
              event.stopPropagation();
            }}
          >
            <div className={styles.opacityWrapper} title="투명도 조절">
              <input
                type="range"
                min={30}
                max={100}
                value={opacityPercent}
                aria-label="채팅창 투명도"
                aria-valuetext={`${opacityPercent}%`}
                className={styles.opacitySlider}
                onChange={(event) => {
                  setOpacityPercent(Number(event.target.value));
                }}
              />
            </div>
            <button type="button" className={styles.clearBtn} onClick={chat.reset}>
              초기화
            </button>
            <button
              type="button"
              className={styles.closeBtn}
              onClick={() => {
                setIsOpen(false);
              }}
              title="채팅창 닫기"
              aria-label="채팅창 닫기"
            >
              &times;
            </button>
          </div>
        </div>
        <div
          className={styles.messages}
          id="chat-messages"
          ref={messagesRef}
          role="log"
          aria-live="polite"
          aria-busy={chat.isSending}
        >
          {entries.map((entry) => {
            if (entry.role === "assistant") {
              return (
                <div key={entry.id} className={`${styles.msg} ${styles.assistant}`}>
                  <MarkdownContent content={entry.content} />
                  {entry.citations.length ? <CitationList citations={entry.citations} /> : null}
                </div>
              );
            }
            if (entry.role === "notice") {
              return (
                <div
                  key={entry.id}
                  className={`${styles.msg} ${styles.assistant} ${styles.notice}`}
                  role="status"
                >
                  {entry.content}
                </div>
              );
            }
            return (
              <div key={entry.id} className={`${styles.msg} ${styles.user}`}>
                {entry.content}
              </div>
            );
          })}
          {streaming === "" ? (
            <div className={`${styles.msg} ${styles.loading}`}>
              <Spinner />
              답변 생성 중...
            </div>
          ) : null}
          {streaming ? (
            <div className={`${styles.msg} ${styles.assistant}`}>
              <MarkdownContent content={streaming} />
            </div>
          ) : null}
        </div>
        <div className={styles.inputArea}>
          <input
            ref={inputRef}
            type="text"
            value={inputText}
            placeholder="질문하기..."
            aria-label="논문에 대해 질문하기"
            onChange={(event) => {
              setInputText(event.target.value);
            }}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !isImeComposing(event)) {
                sendMessage();
              }
            }}
          />
          {chat.isSending ? (
            <button
              type="button"
              className={`${styles.sendBtn} ${styles.stopBtn}`}
              aria-label="답변 중단"
              onClick={chat.stop}
            >
              중지
            </button>
          ) : (
            <button type="button" className={styles.sendBtn} id="send-btn" onClick={sendMessage}>
              전송
            </button>
          )}
        </div>
      </div>

      <button
        type="button"
        className={isOpen ? `${styles.fab} ${styles.hidden}` : styles.fab}
        title="AI 채팅 열기"
        aria-label="AI 채팅 열기"
        onClick={() => {
          setIsOpen(true);
        }}
      >
        <ChatBubbleIcon className={styles.fabIcon} />
      </button>
    </>
  );
}
