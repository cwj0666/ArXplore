import { render } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { MarkdownContent, resolveMarkdownHref } from "./MarkdownContent";

function renderMarkdown(content: string): HTMLElement {
  const { container } = render(<MarkdownContent content={content} />);
  const root = container.firstElementChild;
  if (!(root instanceof HTMLElement)) {
    throw new Error("markdown root missing");
  }
  return root;
}

function normalizedHtml(root: HTMLElement): string {
  return root.innerHTML.replace(/>\s+</g, "><");
}

describe("MarkdownContent", () => {
  it("renders the assistant answer shape used by the UI golden", () => {
    const root = renderMarkdown(
      "검색 결과를 정리했습니다.\n\n- **[Benchmark Radar](https://arxiv.org/abs/2609.11115v2)** — 질문과 관련된 논문입니다.",
    );
    expect(normalizedHtml(root)).toBe(
      "<p>검색 결과를 정리했습니다.</p><ul><li><strong>" +
        '<a href="/papers/2609.11115/" target="_blank" rel="noopener noreferrer">Benchmark Radar</a>' +
        "</strong> — 질문과 관련된 논문입니다.</li></ul>",
    );
  });

  it("maps headings to h3-h5, keeps single line breaks and numbered list starts", () => {
    const root = renderMarkdown("# 하나\n## 둘\n### 셋\n#### 넷\n\n첫 줄\n둘째 줄\n\n3. 셋째\n4. 넷째");
    expect(Array.from(root.querySelectorAll("h3, h4, h5"), (heading) => heading.tagName)).toEqual([
      "H3",
      "H4",
      "H5",
      "H5",
    ]);
    expect(root.querySelector("p")?.innerHTML).toBe("첫 줄<br>\n둘째 줄");
    expect(root.querySelector("ol")).toHaveAttribute("start", "3");
  });

  it("renders the GFM syntax the old renderer could not", () => {
    const root = renderMarkdown(
      [
        "*기울임* ~~취소~~ `코드`",
        "",
        "- 바깥",
        "  - 안쪽",
        "",
        "> 인용",
        "",
        "| 이름 | 값 |",
        "| --- | --- |",
        "| a | 1 |",
        "",
        "```ts",
        "const a = 1;",
        "```",
      ].join("\n"),
    );
    expect(root.querySelector("em")).toHaveTextContent("기울임");
    expect(root.querySelector("del")).toHaveTextContent("취소");
    expect(root.querySelector("ul ul li")).toHaveTextContent("안쪽");
    expect(root.querySelector("blockquote")).toHaveTextContent("인용");
    expect(Array.from(root.querySelectorAll("th"), (cell) => cell.textContent)).toEqual(["이름", "값"]);
    expect(root.querySelector("pre code")).toHaveTextContent("const a = 1;");
  });

  it("turns arXiv abs and pdf links into internal paper paths and opens every link in a new tab", () => {
    const root = renderMarkdown(
      "[a](https://www.arxiv.org/pdf/2609.12345v3.pdf) [b](http://arxiv.org/abs/2609.12345?x=1#y) " +
        "[c](https://export.arxiv.org/abs/2609.12345) [d](https://example.com/a?b=1&c=2) " +
        "https://arxiv.org/abs/2609.54321 [e](/papers/2609.1/) [f](mailto:a@example.com)",
    );
    const anchors = Array.from(root.querySelectorAll("a"));
    expect(anchors.map((anchor) => anchor.getAttribute("href"))).toEqual([
      "/papers/2609.12345/",
      "/papers/2609.12345/",
      "https://export.arxiv.org/abs/2609.12345",
      "https://example.com/a?b=1&c=2",
      "/papers/2609.54321/",
    ]);
    for (const anchor of anchors) {
      expect(anchor).toHaveAttribute("target", "_blank");
      expect(anchor).toHaveAttribute("rel", "noopener noreferrer");
    }
    expect(root).toHaveTextContent("e f");
  });

  it("never produces active markup from untrusted text", () => {
    const attacks = [
      "<script>alert(1)</script>",
      '<img src=x onerror="alert(1)">',
      "[x](javascript:alert(1))",
      "[x](JaVaScRiPt:alert(1))",
      "[x](&#106;avascript:alert(1))",
      "[x](data:text/html,<script>alert(1)</script>)",
      "<javascript:alert(1)>",
      "[ref]\n\n[ref]: javascript:alert(1)",
      '[x](https://example.com/"onmouseover="alert(1))',
      "![x](https://evil.example/pixel.png)",
      "**<svg onload=alert(1)>**",
      '<a href="javascript:alert(1)">x</a>',
      "<iframe src=https://evil.example></iframe>",
      "<style>body{display:none}</style>",
      '<div style="position:fixed">x</div>',
    ];
    for (const attack of attacks) {
      const root = renderMarkdown(attack);
      expect(
        root.querySelector(
          "script, img, svg, iframe, style, object, embed, form, input:not([type='checkbox'])",
        ),
      ).toBeNull();
      for (const element of root.querySelectorAll("*")) {
        for (const attribute of element.getAttributeNames()) {
          expect(attribute, attack).not.toMatch(/^(on|style$)/i);
        }
      }
      for (const anchor of root.querySelectorAll("a")) {
        expect(anchor.getAttribute("href"), attack).toMatch(/^(https?:\/\/|\/papers\/)/);
      }
    }
  });

  it("shows the alt text of images instead of loading them", () => {
    const root = renderMarkdown("앞 ![그림 설명](https://evil.example/pixel.png) 뒤");
    expect(root.querySelector("img")).toBeNull();
    expect(root).toHaveTextContent("앞 그림 설명 뒤");
  });
});

describe("resolveMarkdownHref", () => {
  it("accepts only absolute http(s) URLs", () => {
    expect(resolveMarkdownHref("https://arxiv.org/abs/2609.1v2")).toBe("/papers/2609.1/");
    expect(resolveMarkdownHref("HTTPS://example.com")).toBe("HTTPS://example.com");
    expect(resolveMarkdownHref("/papers/2609.1/")).toBeNull();
    expect(resolveMarkdownHref("")).toBeNull();
    expect(resolveMarkdownHref(undefined)).toBeNull();
  });
});
