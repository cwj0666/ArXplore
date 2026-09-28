import { render, screen, within } from "@testing-library/react";
import { MemoryRouter } from "react-router";
import { describe, expect, it } from "vitest";

import { PaperCard } from "./PaperCard";

const PAPER = {
  arxiv_id: "2609.24815",
  title: "Uranus: Building the Next-Generation Simulation Infrastructure for Embodied AI",
  authors: ["A"],
  abstract: "Scalable simulation is essential.",
  published_at: "2026-09-23T09:00:00+09:00",
  upvotes: 2,
  pdf_url: "https://arxiv.org/pdf/2609.24815.pdf",
};

describe("PaperCard", () => {
  it("offers the PDF link and the detail link side by side instead of nesting them", () => {
    render(
      <MemoryRouter>
        <PaperCard paper={PAPER} />
      </MemoryRouter>,
    );

    const card = screen.getByRole("article");
    const [pdf, detail] = within(card).getAllByRole("link");
    expect(pdf).toHaveAttribute("href", "https://arxiv.org/pdf/2609.24815.pdf");
    expect(pdf).toHaveAttribute("target", "_blank");
    expect(pdf).toHaveTextContent("Uranus: Building the Next-Generation Simulation Infrastructure fo...");
    expect(detail).toHaveAttribute("href", "/papers/2609.24815/");
    expect(detail).toHaveAccessibleName(`${PAPER.title} 상세 보기`);
    expect(detail).toHaveTextContent("2026-09-23");
    expect(pdf?.contains(detail ?? null)).toBe(false);
    expect(detail?.contains(pdf ?? null)).toBe(false);
    expect(card).not.toHaveAttribute("tabindex");
  });

  it("falls back to the arXiv abstract page when the PDF URL is not http(s)", () => {
    render(
      <MemoryRouter>
        <PaperCard paper={{ ...PAPER, pdf_url: "javascript:alert(1)", published_at: null }} />
      </MemoryRouter>,
    );
    const [pdf] = screen.getAllByRole("link");
    expect(pdf).toHaveAttribute("href", "https://arxiv.org/abs/2609.24815");
  });
});
