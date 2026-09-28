import { describe, expect, it } from "vitest";

import { buildListHref, buildListSearchParams, normalizeSort, readListParams } from "./listParams";

describe("readListParams", () => {
  it("normalises every value while reading", () => {
    expect(readListParams(new URLSearchParams("q=diffusion&sort=upvotes&mode=ai&page=3.7"))).toEqual({
      q: "diffusion",
      sort: "upvotes",
      mode: "ai",
      page: 3,
    });
    expect(readListParams(new URLSearchParams("sort=oldest&mode=chat&page=-2"))).toEqual({
      q: "",
      sort: "latest",
      mode: "search",
      page: 1,
    });
    expect(readListParams(new URLSearchParams("page=abc")).page).toBe(1);
    expect(readListParams(new URLSearchParams("page=Infinity")).page).toBe(1);
    expect(readListParams(new URLSearchParams("q=%20spaced%20")).q).toBe(" spaced ");
  });

  it("maps anything but upvotes to latest", () => {
    expect(normalizeSort("upvotes")).toBe("upvotes");
    expect(normalizeSort("UPVOTES")).toBe("latest");
    expect(normalizeSort(null)).toBe("latest");
  });
});

describe("buildListSearchParams", () => {
  it("leaves defaults out of the URL", () => {
    expect(buildListSearchParams({ q: "", sort: "latest", mode: "search", page: 1 }).toString()).toBe("");
    expect(buildListSearchParams({ q: "a b", sort: "upvotes", mode: "ai", page: 2 }).toString()).toBe(
      "q=a+b&sort=upvotes&mode=ai&page=2",
    );
  });

  it("builds pagination hrefs from the root path", () => {
    expect(buildListHref({ q: "", sort: "latest", mode: "search", page: 1 })).toBe("/");
    expect(buildListHref({ q: "rag", sort: "latest", mode: "search", page: 4 })).toBe("/?q=rag&page=4");
  });
});
