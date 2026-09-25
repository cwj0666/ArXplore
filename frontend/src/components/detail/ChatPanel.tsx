import {
  type CSSProperties,
  type MouseEvent as ReactMouseEvent,
  useEffect,
  useRef,
  useState,
} from "react";

import { ApiError } from "../../helpers/http";
import { isImeComposing } from "../../helpers/keyboard";
import { postPaperChat } from "../../pages/detail/detail-api";
import type { ChatMessage } from "../../pages/detail/detail-types";

interface ChatPanelProps {
  arxivId: string;
}

type UiRole = "assistant" | "user" | "loading";

interface UiMessage {
  id: string;
  role: UiRole;
  content: string;
}

interface PanelRect {
  top: number;
  left: number;
  width: number;
  height: number;
}

const WELCOME_MESSAGE = "이 논문에 대해 궁금한 점을 편하게 물어보세요!";
const MIN_VISIBLE_HEADER = 40;

function createMessage(role: UiRole, content: string): UiMessage {
  return {
    id: `${Date.now()}-${Math.random().toString(16).slice(2)}`,
    role,
    content,
  };
}

export function ChatPanel({ arxivId }: ChatPanelProps) {
  const panelRef = useRef<HTMLDivElement | null>(null);
  const messagesRef = useRef<HTMLDivElement | null>(null);
  const inputRef = useRef<HTMLInputElement | null>(null);
  const requestControllerRef = useRef<AbortController | null>(null);

  const [isOpen, setIsOpen] = useState(false);
  const [opacityPercent, setOpacityPercent] = useState(100);
  const [panelRect, setPanelRect] = useState<PanelRect | null>(null);
  const [inputText, setInputText] = useState("");
  const [isSending, setIsSending] = useState(false);
  const [history, setHistory] = useState<ChatMessage[]>([]);
  const [messages, setMessages] = useState<UiMessage[]>([
    createMessage("assistant", WELCOME_MESSAGE),
  ]);

  useEffect(() => {
    return () => {
      requestControllerRef.current?.abort();
      requestControllerRef.current = null;
    };
  }, [arxivId]);

  useEffect(() => {
    setIsOpen(false);
    setOpacityPercent(100);
    setPanelRect(null);
    setInputText("");
    setIsSending(false);
    setHistory([]);
    setMessages([createMessage("assistant", WELCOME_MESSAGE)]);
  }, [arxivId]);

  useEffect(() => {
    const box = messagesRef.current;
    if (!box) {
      return;
    }
    box.scrollTop = box.scrollHeight;
  }, [messages]);

  const panelStyle: CSSProperties = {
    opacity: opacityPercent / 100,
  };

  if (panelRect) {
    panelStyle.top = `${panelRect.top}px`;
    panelStyle.left = `${panelRect.left}px`;
    panelStyle.width = `${panelRect.width}px`;
    panelStyle.height = `${panelRect.height}px`;
    panelStyle.bottom = "auto";
    panelStyle.right = "auto";
  }

  const clearChat = () => {
    requestControllerRef.current?.abort();
    requestControllerRef.current = null;
    setIsSending(false);
    setHistory([]);
    setMessages([createMessage("assistant", WELCOME_MESSAGE)]);
  };

  const ensureAbsoluteRect = (): PanelRect | null => {
    if (panelRect) {
      return panelRect;
    }

    const panel = panelRef.current;
    if (!panel) {
      return null;
    }

    const rect = panel.getBoundingClientRect();
    const nextRect: PanelRect = {
      top: rect.top,
      left: rect.left,
      width: rect.width,
      height: rect.height,
    };
    setPanelRect(nextRect);
    return nextRect;
  };

  const sendMessage = async () => {
    const message = inputText.trim();
    if (!message || isSending) {
      return;
    }

    setInputText("");
    setIsSending(true);

    const userMessage = createMessage("user", message);
    const loadingMessage = createMessage("loading", "답변 생성 중...");
    const userChat: ChatMessage = { role: "user", content: message };
    const nextHistory = [...history, userChat];

    setMessages((prev) => [...prev, userMessage, loadingMessage]);
    setHistory(nextHistory);

    const controller = new AbortController();
    requestControllerRef.current = controller;

    try {
      const data = await postPaperChat(arxivId, message, nextHistory, controller.signal);
      if (controller.signal.aborted) {
        return;
      }
      const assistantReply = data.error
        ? `오류: ${data.error}`
        : data.answer ?? "응답이 비어 있습니다.";

      setMessages((prev) => [
        ...prev.filter((msg) => msg.id !== loadingMessage.id),
        createMessage("assistant", assistantReply),
      ]);

      if (!data.error) {
        setHistory((prev) => [...prev, { role: "assistant", content: assistantReply }]);
      }
    } catch (error) {
      if (controller.signal.aborted) {
        return;
      }
      const errorReply =
        error instanceof ApiError ? `오류: ${error.message}` : "네트워크 오류가 발생했습니다.";
      setMessages((prev) => [
        ...prev.filter((msg) => msg.id !== loadingMessage.id),
        createMessage("assistant", errorReply),
      ]);
    } finally {
      if (requestControllerRef.current === controller) {
        requestControllerRef.current = null;
      }
      if (!controller.signal.aborted) {
        setIsSending(false);
        inputRef.current?.focus();
      }
    }
  };

  const handleDragStart = (event: ReactMouseEvent<HTMLElement>) => {
    const targetTag = (event.target as HTMLElement).tagName;
    if (targetTag === "INPUT" || targetTag === "BUTTON") {
      return;
    }

    const panel = panelRef.current;
    if (!panel) {
      return;
    }

    const baseRect = ensureAbsoluteRect();
    if (!baseRect) {
      return;
    }

    event.preventDefault();
    const startX = event.clientX;
    const startY = event.clientY;
    const startTop = baseRect.top;
    const startLeft = baseRect.left;

    const onMove = (moveEvent: MouseEvent) => {
      moveEvent.preventDefault();
      let top = startTop + (moveEvent.clientY - startY);
      let left = startLeft + (moveEvent.clientX - startX);

      const maxTop = window.innerHeight - MIN_VISIBLE_HEADER;
      const maxLeft = window.innerWidth - MIN_VISIBLE_HEADER;
      if (top < 0) top = 0;
      if (left < 0) left = 0;
      if (top > maxTop) top = maxTop;
      if (left > maxLeft) left = maxLeft;

      setPanelRect({
        top,
        left,
        width: baseRect.width,
        height: baseRect.height,
      });
    };

    const onUp = () => {
      document.removeEventListener("mousemove", onMove);
      document.removeEventListener("mouseup", onUp);
    };

    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup", onUp);
  };

  const handleResizeStart = (event: ReactMouseEvent<HTMLDivElement>) => {
    const panel = panelRef.current;
    if (!panel) {
      return;
    }

    const baseRect = ensureAbsoluteRect();
    if (!baseRect) {
      return;
    }

    event.preventDefault();
    event.stopPropagation();

    const startX = event.clientX;
    const startY = event.clientY;
    const startWidth = baseRect.width;
    const startHeight = baseRect.height;
    const startTop = baseRect.top;
    const startLeft = baseRect.left;

    const onMove = (moveEvent: MouseEvent) => {
      const diffX = startX - moveEvent.clientX;
      const diffY = startY - moveEvent.clientY;

      let newWidth = startWidth + diffX;
      let newHeight = startHeight + diffY;
      let newLeft = startLeft - diffX;
      let newTop = startTop - diffY;

      if (newTop < 0) {
        newHeight = startHeight + startTop;
        newTop = 0;
      }
      if (newLeft < 0) {
        newWidth = startWidth + startLeft;
        newLeft = 0;
      }

      let width = startWidth;
      let left = startLeft;
      let height = startHeight;
      let top = startTop;

      if (newWidth > 300 && newWidth < window.innerWidth - 40) {
        width = newWidth;
        left = newLeft;
      }
      if (newHeight > 400 && newHeight < window.innerHeight - 40) {
        height = newHeight;
        top = newTop;
      }

      setPanelRect({ width, left, height, top });
    };

    const onUp = () => {
      document.removeEventListener("mousemove", onMove);
      document.removeEventListener("mouseup", onUp);
    };

    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup", onUp);
  };

  return (
    <>
      <div
        ref={panelRef}
        id="chat-panel"
        className={`chat-panel ${isOpen ? "" : "is-hidden"}`}
        style={panelStyle}
      >
        <div
          className="chat-resize-handle"
          id="chat-resize-tl"
          onMouseDown={handleResizeStart}
        />
        <div className="chat-header" id="chat-header" onMouseDown={handleDragStart}>
          <h3>AI Chat</h3>
          <div
            className="chat-header-controls"
            onMouseDown={(event) => event.stopPropagation()}
          >
            <div className="opacity-slider-wrapper" title="투명도 조절">
              <input
                type="range"
                min={30}
                max={100}
                value={opacityPercent}
                className="opacity-slider"
                onChange={(event) => setOpacityPercent(Number(event.target.value))}
              />
            </div>
            <button
              type="button"
              className="chat-clear-btn chat-clear-btn-compact"
              onClick={clearChat}
            >
              초기화
            </button>
            <button
              type="button"
              className="pdf-control-btn chat-close-btn"
              onClick={() => setIsOpen(false)}
              title="채팅창 닫기"
            >
              &times;
            </button>
          </div>
        </div>
        <div className="chat-messages" id="chat-messages" ref={messagesRef}>
          {messages.map((msg) => {
            if (msg.role === "loading") {
              return (
                <div key={msg.id} className="msg msg-loading">
                  <span className="spinner spinner-compact" />
                  {msg.content}
                </div>
              );
            }

            return (
              <div key={msg.id} className={`msg msg-${msg.role}`}>
                {msg.content}
              </div>
            );
          })}
        </div>
        <div className="chat-input-area">
          <input
            ref={inputRef}
            type="text"
            value={inputText}
            placeholder="질문하기..."
            onChange={(event) => setInputText(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !isImeComposing(event)) {
                void sendMessage();
              }
            }}
          />
          <button
            type="button"
            className="chat-send-btn"
            id="send-btn"
            disabled={isSending}
            onClick={() => void sendMessage()}
          >
            전송
          </button>
        </div>
      </div>

      <button
        type="button"
        className={`chat-fab ${isOpen ? "is-hidden" : ""}`}
        title="AI 채팅 열기"
        aria-label="AI 채팅 열기"
        onClick={() => setIsOpen(true)}
      >
        <svg
          className="chat-fab-icon"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2.1"
          strokeLinecap="round"
          strokeLinejoin="round"
          aria-hidden="true"
        >
          <path d="M21 11.5a8.5 8.5 0 0 1-8.5 8.5H7l-4 3v-5.5A8.5 8.5 0 1 1 21 11.5z" />
          <path d="M8.5 10h7" />
          <path d="M8.5 14h4.5" />
        </svg>
      </button>
    </>
  );
}
