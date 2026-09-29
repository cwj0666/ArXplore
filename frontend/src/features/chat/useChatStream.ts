import { useCallback, useEffect, useRef, useState } from "react";

import type { StreamHandlers } from "../../shared/api/sse";
import type { ChatRequest, ChatTurn, Citation } from "../../shared/api/types";
import { type ChatEntry, createEntry, welcomeEntries } from "./chatEntries";

export type ChatTransport = (
  request: ChatRequest,
  signal: AbortSignal,
  handlers: StreamHandlers,
) => Promise<void>;

export interface ChatNotices {
  abortedWithPartial: string;
  aborted: string;
  empty: string;
  describeError: (error: unknown) => string;
}

export interface ChatStreamOptions {
  welcome: string;
  transport: ChatTransport;
  notices: ChatNotices;
  historyLimit?: number;
  onSettled?: () => void;
}

export interface ChatStream {
  entries: ChatEntry[];
  streaming: string | null;
  isSending: boolean;
  send: (message: string) => boolean;
  stop: () => void;
  reset: () => void;
}

export function useChatStream({
  welcome,
  transport,
  notices,
  historyLimit,
  onSettled,
}: ChatStreamOptions): ChatStream {
  const [entries, setEntries] = useState<ChatEntry[]>(() => welcomeEntries(welcome));
  const [streaming, setStreaming] = useState<string | null>(null);
  const historyRef = useRef<ChatTurn[]>([]);
  const controllerRef = useRef<AbortController | null>(null);
  const mountedRef = useRef(false);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      window.setTimeout(() => {
        if (!mountedRef.current) {
          controllerRef.current?.abort();
        }
      }, 0);
    };
  }, []);

  const run = useCallback(
    async (message: string, controller: AbortController) => {
      const history = historyRef.current;
      const requestHistory = historyLimit === undefined ? history : history.slice(-historyLimit);
      const isCurrent = () => controllerRef.current === controller;

      let accumulated = "";
      let citations: Citation[] = [];
      let notice = "";
      let failed = false;

      try {
        await transport({ message, history: requestHistory }, controller.signal, {
          onChunk: (chunk) => {
            if (!isCurrent()) {
              return;
            }
            accumulated += chunk;
            setStreaming(accumulated);
          },
          onCitations: (received) => {
            citations = received;
          },
        });
        if (!accumulated) {
          notice = notices.empty;
        }
      } catch (error) {
        failed = true;
        if (controller.signal.aborted) {
          notice = accumulated ? notices.abortedWithPartial : notices.aborted;
        } else {
          notice = notices.describeError(error);
        }
      }

      if (!isCurrent()) {
        return;
      }
      controllerRef.current = null;

      if (accumulated && !failed) {
        historyRef.current = [
          ...historyRef.current,
          { role: "user", content: message },
          { role: "assistant", content: accumulated },
        ];
      }
      const settled: ChatEntry[] = [];
      if (accumulated) {
        settled.push(createEntry("assistant", accumulated, citations));
      }
      if (notice) {
        settled.push(createEntry("notice", notice));
      }
      setEntries((previous) => [...previous, ...settled]);
      setStreaming(null);
      onSettled?.();
    },
    [historyLimit, notices, onSettled, transport],
  );

  const send = useCallback(
    (rawMessage: string) => {
      const message = rawMessage.trim();
      if (!message || controllerRef.current) {
        return false;
      }
      const controller = new AbortController();
      controllerRef.current = controller;
      const userEntry = createEntry("user", message);
      setEntries((previous) => [...previous, userEntry]);
      setStreaming("");
      void run(message, controller);
      return true;
    },
    [run],
  );

  const stop = useCallback(() => {
    controllerRef.current?.abort();
  }, []);

  const reset = useCallback(() => {
    controllerRef.current?.abort();
    controllerRef.current = null;
    historyRef.current = [];
    setStreaming(null);
    setEntries(welcomeEntries(welcome));
  }, [welcome]);

  return { entries, streaming, isSending: streaming !== null, send, stop, reset };
}
