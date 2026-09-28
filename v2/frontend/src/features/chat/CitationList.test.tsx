import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import type { Citation } from "../../shared/api/types";
import { CitationList } from "./CitationList";

function citation(overrides: Partial<Citation>): Citation {
  return {
    arxiv_id: "2609.00001",
    title: "Paper",
    url: "",
    section_title: null,
    chunk_id: null,
    in_answer: false,
    ...overrides,
  };
}

describe("CitationList", () => {
  it("renders nothing without citations", () => {
    const { container } = render(<CitationList citations={[]} />);
    expect(container).toBeEmptyDOMElement();
  });

  it("lists cited sources first and resolves hrefs in the documented order", () => {
    render(
      <CitationList
        citations={[
          citation({ title: "외부", url: "https://example.com/paper" }),
          citation({ title: "arXiv", url: "https://arxiv.org/pdf/2609.12345v2.pdf", in_answer: true }),
          citation({ title: "상대", url: "/papers/2609.22222/?tab=1#top", section_title: "Method" }),
          citation({ title: "위험", arxiv_id: "2609.33333", url: "//evil.example/x" }),
          citation({ title: "인코딩", arxiv_id: "2609.44444", url: "/%2F%2Fevil.example" }),
          citation({ title: "스킴", arxiv_id: "2609.55555", url: "javascript:alert(1)" }),
        ]}
      />,
    );

    const links = within(screen.getByRole("list", { name: "답변 출처" })).getAllByRole("link");
    expect(links.map((link) => link.getAttribute("href"))).toEqual([
      "/papers/2609.12345/",
      "https://example.com/paper",
      "/papers/2609.22222/?tab=1#top",
      "/papers/2609.33333/",
      "/papers/2609.44444/",
      "/papers/2609.55555/",
    ]);
    expect(links[0]).toHaveTextContent(/^arXiv$/);
    expect(links[1]).toHaveTextContent(/^외부 \(검색 참고 자료\)$/);
    expect(links[2]).toHaveAttribute("title", "상대 · Method");
    for (const link of links) {
      expect(link).toHaveAttribute("target", "_blank");
      expect(link).toHaveAttribute("rel", "noopener noreferrer");
    }
  });
});
