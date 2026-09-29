import { act, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { StrictMode } from "react";
import { describe, expect, it, vi } from "vitest";

import {
  type ControlledStream,
  controlledSseResponse,
  jsonResponse,
  requestJsonBody,
} from "../../test/sseResponses";
import { AssistantChat } from "./AssistantChat";

function stubStreams() {
  const streams: { stream: ControlledStream; signal: AbortSignal | null | undefined; body: unknown }[] = [];
  const fetchMock = vi.fn<typeof fetch>().mockImplementation((_input, init) => {
    const stream = controlledSseResponse(init?.signal);
    streams.push({ stream, signal: init?.signal, body: requestJsonBody(init) });
    return Promise.resolve(stream.response);
  });
  vi.stubGlobal("fetch", fetchMock);
  return { fetchMock, streams };
}

describe("AssistantChat", () => {
  it("sends the q parameter once under StrictMode and keeps the request alive", async () => {
    const { fetchMock, streams } = stubStreams();
    render(
      <StrictMode>
        <AssistantChat initialQuery="  RAG 논문  " />
      </StrictMode>,
    );

    expect(await screen.findByText("RAG 논문")).toBeInTheDocument();
    expect(screen.getByText("답변을 생성하는 중입니다...")).toBeInTheDocument();
    await new Promise((resolve) => setTimeout(resolve, 10));
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(streams[0]?.body).toEqual({ message: "RAG 논문", history: [] });
    expect(streams[0]?.signal?.aborted).toBe(false);

    await act(async () => {
      streams[0]?.stream.push({ chunk: "답변" });
      streams[0]?.stream.finish();
      await new Promise((resolve) => setTimeout(resolve, 0));
    });
    expect(await screen.findByText("답변")).toBeInTheDocument();
    expect(screen.getByRole("textbox")).toHaveFocus();
  });

  it("shows server errors without a prefix and the rate limit text built on the client", async () => {
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockResolvedValueOnce(jsonResponse(503, { error: "AI 기능을 사용할 수 없습니다." }))
      .mockResolvedValueOnce(jsonResponse(429, { error: "server text", retry_after: 5 }))
      .mockRejectedValueOnce(new TypeError("Failed to fetch"));
    vi.stubGlobal("fetch", fetchMock);
    const user = userEvent.setup();
    render(<AssistantChat initialQuery="" />);
    const input = screen.getByRole("textbox");

    for (const question of ["하나", "둘", "셋"]) {
      await user.type(input, `${question}{Enter}`);
      await screen.findByText(question);
      await vi.waitFor(() => {
        expect(screen.getByRole("button", { name: "전송" })).toBeInTheDocument();
      });
    }

    const notices = screen.getAllByRole("status").map((element) => element.textContent);
    expect(notices).toEqual([
      "AI 기능을 사용할 수 없습니다.",
      "요청이 많아 잠시 제한되었습니다. 5초 후 다시 시도해 주세요.",
      "답변을 불러오는 중 오류가 발생했습니다.",
    ]);
  });

  it("stops the answer and ignores Enter while an IME composition is active", async () => {
    const { streams } = stubStreams();
    const user = userEvent.setup();
    render(<AssistantChat initialQuery="" />);
    const input = screen.getByRole("textbox", { name: "AI 어시스턴트에게 질문하기" });
    expect(input).toHaveAttribute("placeholder", "연구 주제, 트렌드, 논문을 질문해 보세요.");

    await user.type(input, "한글");
    act(() => {
      input.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter", bubbles: true, isComposing: true }));
    });
    expect(streams).toHaveLength(0);

    await user.keyboard("{Enter}");
    expect(streams).toHaveLength(1);
    await act(async () => {
      streams[0]?.stream.push({ chunk: "부분 답" });
      await new Promise((resolve) => setTimeout(resolve, 0));
    });
    await user.click(screen.getByRole("button", { name: "답변 중단" }));

    expect(
      await screen.findByText(
        "사용자의 요청으로 답변이 중단되었습니다. 이 답변은 이후 대화 맥락에 포함되지 않습니다.",
      ),
    ).toBeInTheDocument();
    expect(screen.getByText("부분 답")).toBeInTheDocument();
  });
});
