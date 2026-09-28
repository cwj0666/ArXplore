import { describe, expect, it, vi } from "vitest";

import { ApiError } from "./http";
import {
  createSseParser,
  DEFAULT_STREAM_ERROR,
  parseSseData,
  STREAMING_UNSUPPORTED_ERROR,
  streamChat,
  StreamEventError,
  TRUNCATED_STREAM_ERROR,
} from "./sse";

describe("parseSseData", () => {
  it("recognises the done marker even with surrounding spaces", () => {
    expect(parseSseData("[DONE]")).toEqual({ type: "done" });
    expect(parseSseData(" [DONE] ")).toEqual({ type: "done" });
  });

  it("gives error events priority and falls back to the default message", () => {
    expect(parseSseData('{"error": "검색 실패", "chunk": "x"}')).toEqual({
      type: "error",
      message: "검색 실패",
    });
    expect(parseSseData('{"error": ""}')).toEqual({ type: "error", message: DEFAULT_STREAM_ERROR });
    expect(parseSseData('{"error": null}')).toEqual({ type: "error", message: DEFAULT_STREAM_ERROR });
  });

  it("normalises citations and drops entries without an arXiv id", () => {
    const event = parseSseData(
      JSON.stringify({
        citations: [
          {
            arxiv_id: " 2609.00001 ",
            title: " ",
            url: 5,
            section_title: "  ",
            chunk_id: "3",
            in_answer: "true",
          },
          { title: "no id" },
          null,
          "text",
          ["2609.00003"],
          {
            arxiv_id: "2609.00002",
            title: "Title",
            url: "/papers/2609.00002/",
            section_title: "Intro",
            chunk_id: 7,
            in_answer: true,
          },
        ],
      }),
    );
    expect(event).toEqual({
      type: "citations",
      citations: [
        {
          arxiv_id: "2609.00001",
          title: "2609.00001",
          url: "",
          section_title: null,
          chunk_id: null,
          in_answer: false,
        },
        {
          arxiv_id: "2609.00002",
          title: "Title",
          url: "/papers/2609.00002/",
          section_title: "Intro",
          chunk_id: 7,
          in_answer: true,
        },
      ],
    });
  });

  it("ignores payloads that are not events", () => {
    expect(parseSseData('{"citations": "nope", "chunk": "a"}')).toEqual({ type: "chunk", chunk: "a" });
    expect(parseSseData('{"chunk": 5}')).toBeNull();
    expect(parseSseData("[]")).toBeNull();
    expect(parseSseData('"text"')).toBeNull();
    expect(parseSseData("{broken")).toBeNull();
  });
});

describe("createSseParser", () => {
  it("splits on CRLF, CR and LF and keeps an unfinished line for the next push", () => {
    const parser = createSseParser();
    expect(parser.push('data: {"chunk":"a"}\r\ndata:{"chunk":"b"}\rdata: {"chu')).toEqual([
      { type: "chunk", chunk: "a" },
      { type: "chunk", chunk: "b" },
    ]);
    expect(parser.push('nk":"c"}\n\ndata: [DONE]')).toEqual([{ type: "chunk", chunk: "c" }]);
    expect(parser.flush()).toEqual([{ type: "done" }]);
  });

  it("reads only data lines, strips one leading space and never joins multi-line data", () => {
    const parser = createSseParser();
    expect(
      parser.push(': comment\nevent: message\nid: 7\ndata:  {"chunk":"x"}\ndata: {"chunk":\ndata: "y"}\n'),
    ).toEqual([{ type: "chunk", chunk: "x" }]);
  });

  it("holds a CR at the end of a push until the next push", () => {
    const parser = createSseParser();
    expect(parser.push('data: {"chunk":"a"}\r')).toEqual([{ type: "chunk", chunk: "a" }]);
    expect(parser.push('\ndata: {"chunk":"b"}\n')).toEqual([{ type: "chunk", chunk: "b" }]);
  });
});

function streamResponse(parts: string[], signal: AbortSignal, { close = true } = {}) {
  const encoder = new TextEncoder();
  let cancelled = false;
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const part of parts) {
        controller.enqueue(encoder.encode(part));
      }
      if (close) {
        controller.close();
      }
      signal.addEventListener("abort", () => {
        if (!close) {
          controller.error(signal.reason);
        }
      });
    },
    cancel() {
      cancelled = true;
    },
  });
  return {
    response: new Response(body, { status: 200, headers: { "Content-Type": "text/event-stream" } }),
    wasCancelled: () => cancelled,
  };
}

function runStream(parts: string[], options: { close?: boolean; controller?: AbortController } = {}) {
  const controller = options.controller ?? new AbortController();
  const stream = streamResponse(parts, controller.signal, { close: options.close ?? true });
  const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(stream.response);
  vi.stubGlobal("fetch", fetchMock);
  const chunks: string[] = [];
  const citations: unknown[] = [];
  const promise = streamChat({
    endpoint: "/papers/assistant/stream/",
    body: { message: "q", history: [] },
    signal: controller.signal,
    onChunk: (chunk) => chunks.push(chunk),
    onCitations: (received) => citations.push(received),
  });
  return { promise, chunks, citations, fetchMock, controller, wasCancelled: stream.wasCancelled };
}

describe("streamChat", () => {
  it("delivers chunks and citations until the done marker, across split network reads", async () => {
    const run = runStream([
      'data: {"chunk": "안녕"}\r\n\r\ndata: {"chu',
      'nk": "하세요"}\n\ndata: {"chunk": ""}\n\n',
      'data: {"citations": [{"arxiv_id": "2609.1", "in_answer": true}]}\n\ndata: [DONE]\n\n',
      'data: {"chunk": "after done"}\n\n',
    ]);

    await expect(run.promise).resolves.toBeUndefined();
    expect(run.chunks).toEqual(["안녕", "하세요"]);
    expect(run.citations).toEqual([
      [
        {
          arxiv_id: "2609.1",
          title: "2609.1",
          url: "",
          section_title: null,
          chunk_id: null,
          in_answer: true,
        },
      ],
    ]);
    expect(run.wasCancelled()).toBe(true);
    const init = run.fetchMock.mock.calls[0]?.[1];
    expect(init?.method).toBe("POST");
    expect(init?.headers).toEqual({ "Content-Type": "application/json", Accept: "text/event-stream" });
    expect(init?.body).toBe('{"message":"q","history":[]}');
  });

  it("accepts a done marker on the last line without a trailing newline", async () => {
    const run = runStream(['data: {"chunk": "a"}\n\ndata: [DONE]']);
    await expect(run.promise).resolves.toBeUndefined();
    expect(run.chunks).toEqual(["a"]);
  });

  it("turns an error event into StreamEventError and cancels the reader", async () => {
    const run = runStream([
      'data: {"chunk": "부분"}\n\ndata: {"error": "검색 중 오류"}\n\n',
      "data: [DONE]\n\n",
    ]);
    await expect(run.promise).rejects.toEqual(new StreamEventError("검색 중 오류"));
    expect(run.chunks).toEqual(["부분"]);
    expect(run.wasCancelled()).toBe(true);
  });

  it("reports a stream that ends without the done marker as cut short", async () => {
    const run = runStream(['data: {"chunk": "부분"}\n\n']);
    await expect(run.promise).rejects.toEqual(new StreamEventError(TRUNCATED_STREAM_ERROR));
  });

  it("reports an abort as AbortError instead of a truncated stream", async () => {
    const run = runStream(['data: {"chunk": "부분"}\n\n'], { close: false });
    await vi.waitFor(() => {
      expect(run.chunks).toEqual(["부분"]);
    });
    run.controller.abort();
    await expect(run.promise).rejects.toMatchObject({ name: "AbortError" });
  });

  it("checks the abort signal when the stream closes after an abort", async () => {
    const controller = new AbortController();
    controller.abort();
    const run = runStream(['data: {"chunk": "a"}\n\n'], { controller });
    await expect(run.promise).rejects.toMatchObject({ name: "AbortError" });
  });

  it("sends the CSRF token and turns failed responses into ApiError", async () => {
    document.cookie = "csrftoken=token-1";
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockResolvedValue(new Response(JSON.stringify({ error: "rate", retry_after: 30 }), { status: 429 }));
    vi.stubGlobal("fetch", fetchMock);

    const promise = streamChat({
      endpoint: "/papers/2609.1/chat/stream/",
      body: { message: "q", history: [] },
      signal: new AbortController().signal,
      onChunk: () => undefined,
      onCitations: () => undefined,
    });

    await expect(promise).rejects.toBeInstanceOf(ApiError);
    await expect(promise).rejects.toMatchObject({
      message: "요청이 많아 잠시 제한되었습니다. 30초 후 다시 시도해 주세요.",
      status: 429,
    });
    expect(fetchMock.mock.calls[0]?.[1]?.headers).toEqual({
      "Content-Type": "application/json",
      Accept: "text/event-stream",
      "X-CSRFToken": "token-1",
    });
    document.cookie = "csrftoken=; expires=Thu, 01 Jan 1970 00:00:00 GMT; path=/";
  });

  it("refuses responses without a body", async () => {
    vi.stubGlobal("fetch", vi.fn<typeof fetch>().mockResolvedValue(new Response(null, { status: 200 })));
    await expect(
      streamChat({
        endpoint: "/x",
        body: {},
        signal: new AbortController().signal,
        onChunk: () => undefined,
        onCitations: () => undefined,
      }),
    ).rejects.toThrow(STREAMING_UNSUPPORTED_ERROR);
  });
});
