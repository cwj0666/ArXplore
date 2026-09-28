import { describe, expect, it } from "vitest";

import { renderMarkdownToHtml } from "./renderMarkdown";

function parse(html: string): HTMLElement {
  const container = document.createElement("div");
  container.innerHTML = html;
  return container;
}

describe("renderMarkdownToHtml", () => {
  it("renders the assistant answer shape used by the UI golden", () => {
    const html = renderMarkdownToHtml(
      "검색 결과를 정리했습니다.\n\n- **[Benchmark Radar](https://arxiv.org/abs/2609.11115v2)** — 질문과 관련된 논문입니다.",
    );
    expect(html).toBe(
      "<p>검색 결과를 정리했습니다.</p><ul><li><strong>" +
        '<a href="/papers/2609.11115/" target="_blank" rel="noopener noreferrer">Benchmark Radar</a>' +
        "</strong> — 질문과 관련된 논문입니다.</li></ul>",
    );
  });

  it("keeps one paragraph per line, headings as h3 to h5 and numbered lists with their start", () => {
    expect(renderMarkdownToHtml("# 제목\n첫 줄\n둘째 줄\n\n3. 셋\n4) 넷\n- 가\n* 나")).toBe(
      "<h3>제목</h3><p>첫 줄</p><p>둘째 줄</p>" +
        '<ol start="3"><li>셋</li><li>넷</li></ol><ul><li>가</li><li>나</li></ul>',
    );
    expect(renderMarkdownToHtml("### 작은 제목")).toBe("<h5>작은 제목</h5>");
  });

  it("escapes code fences and inline code without interpreting markdown inside", () => {
    expect(renderMarkdownToHtml("```\n<b>**x**</b>\n  indented\n```\n`<i>` and **bold**")).toBe(
      "<pre><code>&lt;b&gt;**x**&lt;/b&gt;\n  indented</code></pre><p><code>&lt;i&gt;</code> and <strong>bold</strong></p>",
    );
  });

  it("links only http(s) URLs and turns arXiv abs and pdf links into internal paper paths", () => {
    const container = parse(
      renderMarkdownToHtml(
        "[a](https://www.arxiv.org/pdf/2609.12345v3.pdf) [b](http://arxiv.org/abs/2609.12345?x=1#y) " +
          "[c](https://export.arxiv.org/abs/2609.12345) [d](https://example.com/a?b=1&c=2) [e](javascript:alert(1)) " +
          "[f](/papers/2609.1/)",
      ),
    );
    const hrefs = Array.from(container.querySelectorAll("a"), (anchor) => anchor.getAttribute("href"));
    expect(hrefs).toEqual([
      "/papers/2609.12345/",
      "/papers/2609.12345/",
      "https://export.arxiv.org/abs/2609.12345",
      "https://example.com/a?b=1&c=2",
    ]);
    expect(container.textContent).toContain("[e](javascript:alert(1))");
    expect(container.textContent).toContain("[f](/papers/2609.1/)");
  });

  it("never produces markup from untrusted text", () => {
    const attacks = [
      "<script>alert(1)</script>",
      '<img src=x onerror="alert(1)">',
      '[x](https://example.com/"onmouseover="alert(1))',
      "[<img src=x onerror=alert(1)>](https://example.com)",
      "**<svg onload=alert(1)>**",
      "`</code><script>alert(1)</script>`",
      "\u00000\u0000 [a](https://example.com) \u00001\u0000",
      "# <iframe src=javascript:alert(1)>",
      "- <a href=javascript:alert(1)>x</a>",
    ];
    for (const attack of attacks) {
      const container = parse(renderMarkdownToHtml(attack));
      expect(container.querySelector("script, img, svg, iframe")).toBeNull();
      for (const element of container.querySelectorAll("*")) {
        for (const attribute of element.getAttributeNames()) {
          expect(attribute).not.toMatch(/^on/i);
        }
      }
      for (const anchor of container.querySelectorAll("a")) {
        expect(anchor.getAttribute("href")).toMatch(/^(https?:\/\/|\/papers\/)/);
        expect(anchor.getAttribute("rel")).toBe("noopener noreferrer");
      }
    }
  });

  it("drops NUL characters so link placeholders cannot be forged", () => {
    const html = renderMarkdownToHtml("\u00000\u0000[a](https://example.com)");
    expect(html).toBe(
      '<p>0<a href="https://example.com" target="_blank" rel="noopener noreferrer">a</a></p>',
    );
  });
});
