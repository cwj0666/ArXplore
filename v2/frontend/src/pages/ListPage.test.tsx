import { act, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import type { PaperListResponse } from "../shared/api/types";
import { renderRoute } from "../test/renderRoute";
import { jsonResponse, requestUrl } from "../test/sseResponses";
import ListPage from "./ListPage";

function listPayload(url: URL, total: number): PaperListResponse {
  const page = Number(url.searchParams.get("page"));
  return {
    items:
      total === 0
        ? []
        : [
            {
              arxiv_id: "2609.00001",
              title: "A paper about retrieval",
              authors: ["Kim"],
              abstract: "Abstract text",
              published_at: "2026-09-23T00:00:00+00:00",
              upvotes: 3,
              pdf_url: "https://arxiv.org/pdf/2609.00001.pdf",
            },
          ],
    page,
    page_size: 21,
    total_items: total,
    total_pages: Math.max(1, Math.ceil(total / 21)),
    query: url.searchParams.get("q") ?? "",
    sort: url.searchParams.get("sort") === "upvotes" ? "upvotes" : "latest",
    mode: url.searchParams.get("mode") === "ai" ? "ai" : "search",
  };
}

function stubList(total = 60) {
  const urls: string[] = [];
  const fetchMock = vi.fn<typeof fetch>().mockImplementation((input) => {
    const url = new URL(requestUrl(input), "http://localhost");
    urls.push(`${url.pathname}${url.search}`);
    return Promise.resolve(jsonResponse(200, listPayload(url, total)));
  });
  vi.stubGlobal("fetch", fetchMock);
  return urls;
}

describe("ListPage", () => {
  it("keeps the URL as the source of truth with push and replace rules", async () => {
    const urls = stubList();
    const { router } = renderRoute("/", ListPage, "/?q=rag&page=2");
    const user = userEvent.setup();

    expect(await screen.findByText("총 60개 논문 — 2 / 3 페이지")).toBeInTheDocument();
    expect(urls[0]).toBe("/papers/list.json?q=rag&sort=latest&mode=search&page=2");
    expect(screen.getByRole("textbox")).toHaveValue("rag");

    await user.selectOptions(screen.getByRole("combobox", { name: "정렬" }), "추천순");
    expect(router.state.location.search).toBe("?q=rag&sort=upvotes");
    expect(router.state.historyAction).toBe("PUSH");

    await user.click(screen.getByRole("tab", { name: "AI 어시스턴트" }));
    expect(router.state.location.search).toBe("?q=rag&sort=upvotes&mode=ai");
    expect(router.state.historyAction).toBe("REPLACE");
    expect(screen.getByText("Search any papers with AI.")).toBeInTheDocument();

    await user.click(screen.getByRole("tab", { name: "AI 어시스턴트" }));
    await user.keyboard("{ArrowRight}");
    expect(router.state.location.search).toBe("?q=rag&sort=upvotes");
    expect(screen.getByRole("tab", { name: "키워드 검색" })).toHaveFocus();
  });

  it("submits trimmed searches on page one and skips navigation when nothing changes", async () => {
    stubList();
    const { router } = renderRoute("/", ListPage, "/?page=3");
    const user = userEvent.setup();
    await screen.findByText("총 60개 논문 — 3 / 3 페이지");

    const input = screen.getByRole("textbox");
    await user.type(input, "  diffusion  {Enter}");
    expect(router.state.location.search).toBe("?q=diffusion");
    expect(router.state.historyAction).toBe("PUSH");
    expect(input).toHaveValue("diffusion");

    const key = router.state.location.key;
    await user.click(screen.getByRole("button", { name: "Search" }));
    expect(router.state.location.key).toBe(key);

    await act(async () => {
      await router.navigate(-1);
    });
    expect(router.state.location.search).toBe("?page=3");
    expect(input).toHaveValue("");
  });

  it("opens the assistant with the query in AI mode and ignores an empty AI query", async () => {
    stubList();
    const { router } = renderRoute("/", ListPage, "/?mode=ai");
    const user = userEvent.setup();
    await screen.findByText("총 60개 논문 — 1 / 3 페이지");

    await user.click(screen.getByRole("button", { name: "Search" }));
    expect(router.state.location.pathname).toBe("/");

    await user.type(screen.getByRole("textbox"), "  그래프 RAG {Enter}");
    expect(router.state.location.pathname).toBe("/papers/assistant/");
    expect(router.state.location.search).toBe(`?q=${encodeURIComponent("그래프 RAG")}`);
  });

  it("shows the empty and error states", async () => {
    stubList(0);
    renderRoute("/", ListPage, "/?q=nothing");
    expect(await screen.findByText("수집된 논문이 없습니다.")).toBeInTheDocument();
  });

  it("uses the client rate limit text for 429 responses", async () => {
    vi.stubGlobal(
      "fetch",
      vi
        .fn<typeof fetch>()
        .mockResolvedValue(jsonResponse(429, { error: "server text" }, { "Retry-After": "42" })),
    );
    renderRoute("/", ListPage, "/");
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "요청이 많아 잠시 제한되었습니다. 42초 후 다시 시도해 주세요.",
    );
  });
});
