import { act, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import {
  type ControlledStream,
  controlledSseResponse,
  jsonResponse,
  requestJsonBody,
} from "../../test/sseResponses";
import { PaperChatPanel } from "./PaperChatPanel";

describe("PaperChatPanel", () => {
  it("prefixes errors, keeps failed turns out of the history and resets without notices", async () => {
    const streams: ControlledStream[] = [];
    const bodies: unknown[] = [];
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockResolvedValueOnce(jsonResponse(404, { error: "논문을 찾을 수 없습니다." }))
      .mockImplementation((_input, init) => {
        const stream = controlledSseResponse(init?.signal);
        streams.push(stream);
        bodies.push(requestJsonBody(init));
        return Promise.resolve(stream.response);
      });
    vi.stubGlobal("fetch", fetchMock);
    const user = userEvent.setup();
    render(<PaperChatPanel arxivId="2609.27980" />);

    await user.click(screen.getByRole("button", { name: "AI 채팅 열기" }));
    const input = screen.getByRole("textbox", { name: "논문에 대해 질문하기" });
    await user.type(input, "첫 질문{Enter}");
    expect(await screen.findByText("오류: 논문을 찾을 수 없습니다.")).toBeInTheDocument();
    expect(fetchMock.mock.calls[0]?.[0]).toBe("/papers/2609.27980/chat/stream/");

    await user.type(input, "둘째 질문{Enter}");
    const log = screen.getByRole("log");
    expect(within(log).getByText("답변 생성 중...")).toBeInTheDocument();
    expect(bodies[0]).toEqual({ message: "둘째 질문", history: [] });

    await act(async () => {
      streams[0]?.push({ chunk: "답변 일부" });
      await new Promise((resolve) => setTimeout(resolve, 0));
    });
    await user.click(screen.getByRole("button", { name: "초기화" }));

    expect(
      within(log)
        .getAllByText(/./)
        .map((element) => element.textContent),
    ).toEqual(["이 논문에 대해 궁금한 점을 편하게 물어보세요!"]);
    expect(screen.getByRole("button", { name: "전송" })).toBeInTheDocument();
  });

  it("reports a missing stream endpoint instead of falling back to the JSON chat endpoint", async () => {
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockResolvedValue(new Response("<h1>Not Found</h1>", { status: 404, statusText: "Not Found" }));
    vi.stubGlobal("fetch", fetchMock);
    const user = userEvent.setup();
    render(<PaperChatPanel arxivId="2609.27980" />);

    await user.click(screen.getByRole("button", { name: "AI 채팅 열기" }));
    await user.type(screen.getByRole("textbox", { name: "논문에 대해 질문하기" }), "질문{Enter}");

    expect(await screen.findByText("오류: 요청이 실패했습니다. (HTTP 404 Not Found)")).toBeInTheDocument();
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("moves focus into the panel when it opens and back to the button when Esc or × closes it", async () => {
    const user = userEvent.setup();
    render(<PaperChatPanel arxivId="2609.27980" />);
    const panel = screen.getByRole("region", { name: "논문 AI 채팅" });
    const fab = screen.getByRole("button", { name: "AI 채팅 열기" });

    await user.click(fab);
    const input = screen.getByRole("textbox", { name: "논문에 대해 질문하기" });
    expect(input).toHaveFocus();

    await user.keyboard("{Escape}");
    expect(panel.className).toMatch(/hidden/);
    expect(fab).toHaveFocus();

    await user.keyboard("{Enter}");
    expect(input).toHaveFocus();
    await user.click(screen.getByRole("button", { name: "채팅창 닫기" }));
    expect(fab).toHaveFocus();
  });

  it("returns focus to the input after an answer and hides the panel without cancelling", async () => {
    const streams: ControlledStream[] = [];
    vi.stubGlobal(
      "fetch",
      vi.fn<typeof fetch>().mockImplementation((_input, init) => {
        const stream = controlledSseResponse(init?.signal);
        streams.push(stream);
        return Promise.resolve(stream.response);
      }),
    );
    const user = userEvent.setup();
    render(<PaperChatPanel arxivId="2609.27980" />);
    await user.click(screen.getByRole("button", { name: "AI 채팅 열기" }));
    const input = screen.getByRole("textbox", { name: "논문에 대해 질문하기" });
    await user.type(input, "질문{Enter}");
    expect(input).toHaveValue("");

    await user.click(screen.getByRole("button", { name: "채팅창 닫기" }));
    await act(async () => {
      streams[0]?.push({ chunk: "완료된 답" });
      streams[0]?.push({
        citations: [{ arxiv_id: "2609.27980", title: "Six Layers Less", in_answer: true }],
      });
      streams[0]?.finish();
      await new Promise((resolve) => setTimeout(resolve, 0));
    });
    await user.click(screen.getByRole("button", { name: "AI 채팅 열기" }));

    expect(screen.getByText("완료된 답")).toBeInTheDocument();
    expect(screen.getByRole("list", { name: "답변 출처" })).toBeInTheDocument();
  });
});
