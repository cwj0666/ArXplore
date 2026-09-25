import { useEffect, useRef, useState } from "react";

import { AssistantChatHistory } from "../../components/assistant/AssistantChatHistory";
import { AssistantComposer } from "../../components/assistant/AssistantComposer";
import { AssistantHero } from "../../components/assistant/AssistantHero";
import { AssistantNotice } from "../../components/assistant/AssistantNotice";
import { streamAssistantChat } from "../../helpers/assistant/assistantChatApi";
import { ApiError } from "../../helpers/http";
import { StreamEventError } from "../../helpers/sse";
import type { AssistantChatMessage, AssistantDisplayMessage, Citation } from "../../types/assistant";
import type { BootstrapPayload } from "../../types/app";
import "./assistant-page.css";

const INITIAL_ASSISTANT_MESSAGE =
  "찾고 싶은 연구 주제나 기술 키워드를 질문해 주세요.";
const STREAM_ENDPOINT = "/papers/assistant/stream/";
const MAX_HISTORY_MESSAGES = 20;

export interface AssistantPageProps {
  session: BootstrapPayload;
  initialQuery?: string;
  homeHref?: string;
  onRequireLogin: () => void;
  onOpenSettings: () => void;
}

export function AssistantPage({
  session,
  initialQuery = "",
  homeHref = "/",
  onRequireLogin,
  onOpenSettings,
}: AssistantPageProps) {
  const [messages, setMessages] = useState<AssistantDisplayMessage[]>([
    { role: "assistant", content: INITIAL_ASSISTANT_MESSAGE },
  ]);
  const [chatHistory, setChatHistory] = useState<AssistantChatMessage[]>([]);
  const [inputValue, setInputValue] = useState("");
  const [isSending, setIsSending] = useState(false);
  const [streamingContent, setStreamingContent] = useState("");
  const hasSubmittedInitialQueryRef = useRef(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const chatHistoryRef = useRef<HTMLDivElement>(null);
  const abortControllerRef = useRef<AbortController | null>(null);
  const isMountedRef = useRef(false);
  const canUseAssistant = session.is_authenticated && session.has_personal_api_key;

  const scrollToBottom = () => {
    if (!chatHistoryRef.current) return;
    chatHistoryRef.current.scrollTop = chatHistoryRef.current.scrollHeight;
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isSending, streamingContent]);

  useEffect(() => {
    isMountedRef.current = true;
    return () => {
      isMountedRef.current = false;
      // Deferred so StrictMode's synchronous unmount/remount does not cancel the initial query.
      window.setTimeout(() => {
        if (!isMountedRef.current) abortControllerRef.current?.abort();
      }, 0);
    };
  }, []);

  const stopGeneration = () => {
    abortControllerRef.current?.abort();
  };

  const sendMessage = async (prefilledMessage = "") => {
    if (!canUseAssistant || isSending) return;

    const message = (prefilledMessage || inputValue).trim();
    if (!message) return;

    const requestHistory = chatHistory.slice(-MAX_HISTORY_MESSAGES);
    const userMessage: AssistantChatMessage = { role: "user", content: message };

    setInputValue("");
    setIsSending(true);
    setStreamingContent("");
    setMessages((prev) => [...prev, userMessage]);
    setChatHistory((prev) => [...prev, userMessage]);

    const controller = new AbortController();
    abortControllerRef.current = controller;
    let accumulated = "";
    let citations: Citation[] = [];
    let notice = "";

    try {
      await streamAssistantChat({
        endpoint: STREAM_ENDPOINT,
        message,
        history: requestHistory,
        signal: controller.signal,
        onChunk: (chunk) => {
          accumulated += chunk;
          setStreamingContent(accumulated);
        },
        onCitations: (received) => {
          citations = received;
        },
      });
      if (!accumulated) notice = "답변을 생성할 수 없습니다.";
    } catch (error) {
      if (controller.signal.aborted) {
        if (!accumulated) notice = "사용자의 요청으로 답변이 중단되었습니다.";
      } else {
        notice = (error instanceof ApiError || error instanceof StreamEventError) && error.message
          ? error.message
          : "답변을 불러오는 중 오류가 발생했습니다.";
      }
    } finally {
      const newDisplayMessages: AssistantDisplayMessage[] = [];
      if (accumulated) {
        const assistantMessage: AssistantChatMessage = { role: "assistant", content: accumulated };
        newDisplayMessages.push(citations.length ? { ...assistantMessage, citations } : assistantMessage);
        setChatHistory((prev) => [...prev, assistantMessage]);
      }
      if (notice) {
        newDisplayMessages.push({ role: "assistant", content: notice, isNotice: true });
      }
      setMessages((prev) => [...prev, ...newDisplayMessages]);
      setStreamingContent("");
      setIsSending(false);
      abortControllerRef.current = null;
      inputRef.current?.focus();
    }
  };

  useEffect(() => {
    if (!canUseAssistant || !initialQuery.trim() || hasSubmittedInitialQueryRef.current) return;
    hasSubmittedInitialQueryRef.current = true;
    void sendMessage(initialQuery.trim());
  }, [canUseAssistant, initialQuery]);

  return (
    <main className="assistant-page">
      <div className="assistant-page-inner">
        <AssistantHero />
        {canUseAssistant ? null : (
          <AssistantNotice
            isAuthenticated={session.is_authenticated}
            homeHref={homeHref}
            onRequireLogin={onRequireLogin}
            onOpenSettings={onOpenSettings}
          />
        )}

        <div className="assistant-chat-shell">
          <AssistantChatHistory
            messages={messages}
            isSending={isSending}
            streamingContent={streamingContent}
            ref={chatHistoryRef}
          />
          <AssistantComposer
            value={inputValue}
            disabled={!canUseAssistant}
            isSending={isSending}
            inputRef={inputRef}
            onChange={setInputValue}
            onSend={() => { void sendMessage(); }}
            onStop={stopGeneration}
          />
        </div>
      </div>
    </main>
  );
}
