import { describe, expect, it } from "vitest";

import { isSameOriginPathname, toSafeHttpUrl } from "./safeUrl";

describe("toSafeHttpUrl", () => {
  it("accepts trimmed http and https URLs only", () => {
    expect(toSafeHttpUrl("  https://arxiv.org/pdf/2609.1.pdf ")).toBe("https://arxiv.org/pdf/2609.1.pdf");
    expect(toSafeHttpUrl("HTTP://example.com")).toBe("HTTP://example.com");
    expect(toSafeHttpUrl("javascript:alert(1)")).toBe("");
    expect(toSafeHttpUrl("//evil.example")).toBe("");
    expect(toSafeHttpUrl("data:text/html,x", "https://arxiv.org/abs/1")).toBe("https://arxiv.org/abs/1");
    expect(toSafeHttpUrl(null, "fallback")).toBe("fallback");
    expect(toSafeHttpUrl(undefined)).toBe("");
  });
});

describe("isSameOriginPathname", () => {
  it("accepts ordinary paths", () => {
    expect(isSameOriginPathname("/papers/2609.1/")).toBe(true);
    expect(isSameOriginPathname("/papers/%ED%95%9C/")).toBe(true);
  });

  it("rejects protocol-relative paths, backslashes and control characters, also after decoding", () => {
    for (const pathname of [
      "//evil.example/x",
      "/%2F%2Fevil.example",
      "/\\evil.example",
      "/%5Cevil",
      "/a\u0000b",
      "/a%00b",
      "/a%0Ab",
      "/a%7Fb",
      "/a\u007fb",
      "/%E0%A4%A",
    ]) {
      expect(isSameOriginPathname(pathname), pathname).toBe(false);
    }
  });
});
