import { screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import type { PaperDetailResponse } from "../shared/api/types";
import { renderRoute } from "../test/renderRoute";
import { jsonResponse, requestJsonBody, requestUrl } from "../test/sseResponses";
import PaperDetailPage from "./PaperDetailPage";

const DETAIL: PaperDetailResponse = {
  paper: {
    arxiv_id: "2609.27980",
    title: "Six Layers Less",
    authors: ["Rasmus Aagaard", "Nicki Skafte Detlefsen"],
    abstract: "Pruning large models.",
    published_at: null,
    upvotes: 4,
    pdf_url: "https://arxiv.org/pdf/2609.27980.pdf",
    related_papers: [
      {
        arxiv_id: "2609.11111",
        title: "Local related",
        authors: ["A"],
        abstract: "x".repeat(200),
        published_at: "2026-09-10T00:00:00",
        upvotes: 1,
        pdf_url: null,
        source: "local",
        relation_score: 0.5,
      },
      {
        arxiv_id: "2609.22222",
        title: "External related",
        authors: [],
        abstract: "",
        published_at: null,
        upvotes: 0,
        pdf_url: null,
        source: "arxiv",
        relation_score: 0,
      },
    ],
  },
};

interface Pending {
  url: string;
  body: unknown;
  signal: AbortSignal | null | undefined;
  respond: (response: Response) => void;
}

function stubDetailApi() {
  const pending: Pending[] = [];
  const fetchMock = vi.fn<typeof fetch>().mockImplementation((input, init) => {
    const url = requestUrl(input);
    if (url.endsWith("/detail.json")) {
      return Promise.resolve(
        url.includes("2609.99999")
          ? jsonResponse(404, { error: "논문을 찾을 수 없습니다." })
          : jsonResponse(200, DETAIL),
      );
    }
    return new Promise<Response>((resolve, reject) => {
      const signal = init?.signal;
      signal?.addEventListener("abort", () => {
        reject(new DOMException("The operation was aborted.", "AbortError"));
      });
      pending.push({
        url,
        body: requestJsonBody(init),
        signal,
        respond: resolve,
      });
    });
  });
  vi.stubGlobal("fetch", fetchMock);
  const next = async (suffix: string) => {
    await vi.waitFor(() => {
      expect(pending.some((call) => call.url.endsWith(suffix))).toBe(true);
    });
    const index = pending.findIndex((call) => call.url.endsWith(suffix));
    const [call] = pending.splice(index, 1);
    if (!call) {
      throw new Error(`no pending ${suffix} request`);
    }
    return call;
  };
  return { next };
}

describe("PaperDetailPage", () => {
  it("requests the overview automatically, supports cancel and retry, and shows the findings", async () => {
    const api = stubDetailApi();
    const user = userEvent.setup();
    renderRoute("/papers/:arxivId/", PaperDetailPage, "/papers/2609.27980/");

    expect(screen.getByRole("status")).toHaveTextContent("논문을 불러오는 중입니다...");
    expect(await screen.findByRole("heading", { level: 1, name: "Six Layers Less" })).toBeInTheDocument();
    expect(within(screen.getByRole("main")).getByText("-")).toBeInTheDocument();

    const first = await api.next("/analyze/");
    expect(first.url).toBe("/papers/2609.27980/analyze/");
    expect(first.body).toBeUndefined();
    const overview = screen.getByRole("region", { name: "개요" });
    expect(overview).toHaveAttribute("aria-busy", "true");
    await user.click(within(overview).getByRole("button", { name: "취소" }));
    expect(first.signal?.aborted).toBe(true);
    expect(
      await within(overview).findByText(/요청을 취소했습니다. 서버에서는 처리가 계속될 수 있어서/),
    ).toBeInTheDocument();

    await user.click(within(overview).getByRole("button", { name: "다시 시도" }));
    const second = await api.next("/analyze/");
    second.respond(
      jsonResponse(200, { overview: "첫 문단\n\n둘째 문단", key_findings: ["하나", "둘"], cached: false }),
    );

    expect(await within(overview).findByText("둘째 문단")).toBeInTheDocument();
    expect(within(screen.getByRole("region", { name: "핵심 포인트" })).getAllByRole("listitem")).toHaveLength(
      2,
    );
  });

  it("shows analysis errors with the server text and an empty overview notice", async () => {
    const api = stubDetailApi();
    const user = userEvent.setup();
    renderRoute("/papers/:arxivId/", PaperDetailPage, "/papers/2609.27980/");

    (await api.next("/analyze/")).respond(
      jsonResponse(503, {
        error: "AI 기능을 사용할 수 없습니다. 서버에 OpenAI API 키가 설정되지 않았습니다.",
      }),
    );
    const overview = await screen.findByRole("region", { name: "개요" });
    expect(await within(overview).findByRole("alert")).toHaveTextContent("AI 기능을 사용할 수 없습니다.");

    await user.click(within(overview).getByRole("button", { name: "다시 시도" }));
    (await api.next("/analyze/")).respond(
      jsonResponse(200, { overview: null, key_findings: [], cached: false }),
    );
    expect(await within(overview).findByText("생성된 개요가 없습니다.")).toBeInTheDocument();
    expect(screen.queryByRole("region", { name: "핵심 포인트" })).not.toBeInTheDocument();
  });

  it("asks for the summary model in a dialog and keeps the label once a summary exists", async () => {
    const api = stubDetailApi();
    const user = userEvent.setup();
    renderRoute("/papers/:arxivId/", PaperDetailPage, "/papers/2609.27980/");
    (await api.next("/analyze/")).respond(
      jsonResponse(200, { overview: "개요", key_findings: [], cached: true }),
    );

    const trigger = await screen.findByRole("button", { name: "상세요약 생성하기" });
    await user.click(trigger);
    const dialog = screen.getByRole("dialog", { name: "상세요약 모델 선택" });
    const select = within(dialog).getByRole("combobox", { name: "요약 모델" });
    expect(select).toHaveFocus();
    expect(
      within(select)
        .getAllByRole("option")
        .map((option) => option.textContent),
    ).toEqual(["GPT 5 MINI", "GPT 5"]);

    await user.keyboard("{Escape}");
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    expect(trigger).toHaveFocus();

    await user.click(trigger);
    await user.selectOptions(screen.getByRole("combobox", { name: "요약 모델" }), "gpt-5");
    await user.click(screen.getByRole("button", { name: "생성" }));
    const summary = await api.next("/summary/");
    expect(summary.body).toEqual({ model: "gpt-5" });
    expect(screen.getByRole("button", { name: "생성 중..." })).toBeDisabled();

    summary.respond(
      jsonResponse(200, { summary: "## 문제\n본문 줄\n\n## 결과\n결론", cached: false, model: "gpt-5" }),
    );
    const card = await screen.findByRole("region", { name: "상세 요약" });
    expect(await within(card).findByRole("heading", { name: "결과" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "다른 모델로 상세요약" })).toBeEnabled();
  });

  it("links related papers inside the app or to arXiv", async () => {
    const api = stubDetailApi();
    renderRoute("/papers/:arxivId/", PaperDetailPage, "/papers/2609.27980/");
    await api.next("/analyze/");

    const local = await screen.findByRole("link", { name: /Local related/ });
    expect(local).toHaveAttribute("href", "/papers/2609.11111/");
    expect(local).toHaveTextContent(`${"x".repeat(180)}...`);
    const external = screen.getByRole("link", { name: /External related/ });
    expect(external).toHaveAttribute("href", "https://arxiv.org/abs/2609.22222");
    expect(external).toHaveAttribute("target", "_blank");
  });

  it("shows the server error and an arXiv fallback link for unknown papers", async () => {
    stubDetailApi();
    renderRoute("/papers/:arxivId/", PaperDetailPage, "/papers/2609.99999/");

    expect(await screen.findByRole("alert")).toHaveTextContent("논문을 찾을 수 없습니다.");
    expect(screen.getByRole("link", { name: "arXiv에서 보기 ↗" })).toHaveAttribute(
      "href",
      "https://arxiv.org/abs/2609.99999",
    );
  });
});
