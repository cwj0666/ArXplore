import { act, renderHook } from "@testing-library/react";
import { StrictMode } from "react";
import { describe, expect, it, vi } from "vitest";

import type { StreamHandlers } from "../../shared/api/sse";
import type { ChatRequest, Citation } from "../../shared/api/types";
import { type ChatNotices, type ChatTransport, useChatStream } from "./useChatStream";

interface PendingCall {
  request: ChatRequest;
  signal: AbortSignal;
  handlers: StreamHandlers;
  resolve: () => void;
  reject: (error: unknown) => void;
}

function controllableTransport() {
  const calls: PendingCall[] = [];
  const transport: ChatTransport = (request, signal, handlers) =>
    new Promise<void>((resolve, reject) => {
      calls.push({ request, signal, handlers, resolve, reject });
      signal.addEventListener("abort", () => {
        reject(new DOMException("The operation was aborted.", "AbortError"));
      });
    });
  const last = () => {
    const call = calls.at(-1);
    if (!call) {
      throw new Error("no transport call yet");
    }
    return call;
  };
  return { transport, calls, last };
}

const NOTICES: ChatNotices = {
  abortedWithPartial: "partial aborted",
  aborted: "aborted",
  empty: "empty",
  describeError: (error) => `error: ${error instanceof Error ? error.message : "unknown"}`,
};

const CITATION: Citation = {
  arxiv_id: "2609.00001",
  title: "Paper",
  url: "",
  section_title: "Abstract",
  chunk_id: 1,
  in_answer: true,
};

function setup(options: { historyLimit?: number; strict?: boolean } = {}) {
  const fake = controllableTransport();
  const onSettled = vi.fn();
  const hook = renderHook(
    () =>
      useChatStream({
        welcome: "welcome",
        transport: fake.transport,
        notices: NOTICES,
        onSettled,
        ...(options.historyLimit === undefined ? {} : { historyLimit: options.historyLimit }),
      }),
    options.strict ? { wrapper: StrictMode } : {},
  );
  return { ...fake, onSettled, hook };
}

function contents(entries: { role: string; content: string }[]) {
  return entries.map((entry) => `${entry.role}:${entry.content}`);
}

async function succeed(call: PendingCall, ...chunks: string[]) {
  await act(async () => {
    for (const chunk of chunks) {
      call.handlers.onChunk(chunk);
    }
    call.resolve();
    await Promise.resolve();
  });
}

describe("useChatStream", () => {
  it("streams an answer, then records the turn in the history for the next request", async () => {
    const { hook, last, onSettled } = setup();
    expect(contents(hook.result.current.entries)).toEqual(["assistant:welcome"]);

    act(() => {
      expect(hook.result.current.send("  첫 질문  ")).toBe(true);
    });
    expect(hook.result.current.isSending).toBe(true);
    expect(hook.result.current.streaming).toBe("");
    expect(last().request).toEqual({ message: "첫 질문", history: [] });

    act(() => {
      last().handlers.onChunk("안녕");
    });
    expect(hook.result.current.streaming).toBe("안녕");

    await act(async () => {
      last().handlers.onChunk("하세요");
      last().handlers.onCitations([CITATION]);
      last().resolve();
      await Promise.resolve();
    });

    expect(hook.result.current.isSending).toBe(false);
    expect(hook.result.current.streaming).toBeNull();
    expect(contents(hook.result.current.entries)).toEqual([
      "assistant:welcome",
      "user:첫 질문",
      "assistant:안녕하세요",
    ]);
    expect(hook.result.current.entries[2]?.citations).toEqual([CITATION]);
    expect(onSettled).toHaveBeenCalledTimes(1);

    act(() => {
      hook.result.current.send("둘째 질문");
    });
    expect(last().request).toEqual({
      message: "둘째 질문",
      history: [
        { role: "user", content: "첫 질문" },
        { role: "assistant", content: "안녕하세요" },
      ],
    });
  });

  it("keeps a partial answer on screen after a stop but leaves it out of the history", async () => {
    const { hook, last } = setup();
    act(() => {
      hook.result.current.send("질문");
    });
    act(() => {
      last().handlers.onChunk("부분");
    });
    await act(async () => {
      hook.result.current.stop();
      await Promise.resolve();
    });

    expect(contents(hook.result.current.entries)).toEqual([
      "assistant:welcome",
      "user:질문",
      "assistant:부분",
      "notice:partial aborted",
    ]);

    act(() => {
      hook.result.current.send("다음");
    });
    expect(last().request.history).toEqual([]);
  });

  it("uses the plain stop notice when nothing arrived", async () => {
    const { hook } = setup();
    act(() => {
      hook.result.current.send("질문");
    });
    await act(async () => {
      hook.result.current.stop();
      await Promise.resolve();
    });
    expect(contents(hook.result.current.entries)).toEqual([
      "assistant:welcome",
      "user:질문",
      "notice:aborted",
    ]);
  });

  it("describes failures and empty answers without touching the history", async () => {
    const { hook, last } = setup();
    act(() => {
      hook.result.current.send("실패");
    });
    await act(async () => {
      last().handlers.onChunk("부분");
      last().reject(new Error("boom"));
      await Promise.resolve();
    });
    act(() => {
      hook.result.current.send("빈 답");
    });
    await succeed(last());

    expect(contents(hook.result.current.entries)).toEqual([
      "assistant:welcome",
      "user:실패",
      "assistant:부분",
      "notice:error: boom",
      "user:빈 답",
      "notice:empty",
    ]);
    act(() => {
      hook.result.current.send("다음");
    });
    expect(last().request.history).toEqual([]);
  });

  it("sends only the most recent history messages when a limit is set", async () => {
    const { hook, last } = setup({ historyLimit: 2 });
    for (const [question, answer] of [
      ["q1", "a1"],
      ["q2", "a2"],
    ]) {
      act(() => {
        hook.result.current.send(question ?? "");
      });
      await succeed(last(), answer ?? "");
    }
    act(() => {
      hook.result.current.send("q3");
    });
    expect(last().request.history).toEqual([
      { role: "user", content: "q2" },
      { role: "assistant", content: "a2" },
    ]);
  });

  it("ignores empty input and sends while an answer is pending", () => {
    const { hook, calls } = setup();
    act(() => {
      expect(hook.result.current.send("   ")).toBe(false);
      expect(hook.result.current.send("하나")).toBe(true);
    });
    act(() => {
      expect(hook.result.current.send("둘")).toBe(false);
    });
    expect(calls).toHaveLength(1);
    expect(contents(hook.result.current.entries)).toEqual(["assistant:welcome", "user:하나"]);
  });

  it("drops the result of a request that a reset replaced", async () => {
    const { hook, last, onSettled } = setup();
    act(() => {
      hook.result.current.send("질문");
    });
    const first = last();
    act(() => {
      first.handlers.onChunk("부분");
    });
    await act(async () => {
      hook.result.current.reset();
      await Promise.resolve();
    });

    expect(first.signal.aborted).toBe(true);
    expect(contents(hook.result.current.entries)).toEqual(["assistant:welcome"]);
    expect(hook.result.current.isSending).toBe(false);
    expect(onSettled).not.toHaveBeenCalled();

    act(() => {
      first.handlers.onChunk("늦은 조각");
    });
    expect(hook.result.current.streaming).toBeNull();

    act(() => {
      hook.result.current.send("새 질문");
    });
    expect(last().request.history).toEqual([]);
  });

  it("aborts the pending request only when the component really unmounts", () => {
    vi.useFakeTimers();
    try {
      const { hook, last } = setup({ strict: true });
      act(() => {
        hook.result.current.send("질문");
      });
      act(() => {
        vi.runAllTimers();
      });
      expect(last().signal.aborted).toBe(false);

      hook.unmount();
      expect(last().signal.aborted).toBe(false);
      vi.runAllTimers();
      expect(last().signal.aborted).toBe(true);
    } finally {
      vi.useRealTimers();
    }
  });
});
