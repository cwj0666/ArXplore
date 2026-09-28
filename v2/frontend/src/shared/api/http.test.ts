import { afterEach, describe, expect, it, vi } from "vitest";

import { ApiError, getCsrfTokenFromCookie, getJson, postJson, readApiError } from "./http";

function jsonResponse(status: number, body: unknown, headers: Record<string, string> = {}) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json", ...headers },
  });
}

function stubFetch(response: Response) {
  const fetchMock = vi.fn<typeof fetch>().mockImplementation(() => Promise.resolve(response.clone()));
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

function clearCookies() {
  for (const part of document.cookie.split(";")) {
    const name = part.split("=")[0]?.trim();
    if (name) {
      document.cookie = `${name}=; expires=Thu, 01 Jan 1970 00:00:00 GMT; path=/`;
    }
  }
}

afterEach(() => {
  clearCookies();
});

describe("readApiError", () => {
  it("builds the rate limit message from the payload retry_after first", async () => {
    const error = await readApiError(
      jsonResponse(429, { error: "server text", retry_after: 12.2 }, { "Retry-After": "40" }),
    );
    expect(error.message).toBe("요청이 많아 잠시 제한되었습니다. 13초 후 다시 시도해 주세요.");
    expect(error.status).toBe(429);
  });

  it("falls back to the Retry-After header and then to a generic rate limit message", async () => {
    const fromHeader = await readApiError(jsonResponse(429, { error: "x" }, { "Retry-After": "30" }));
    expect(fromHeader.message).toBe("요청이 많아 잠시 제한되었습니다. 30초 후 다시 시도해 주세요.");

    const withoutHint = await readApiError(new Response("not json", { status: 429 }));
    expect(withoutHint.message).toBe("요청이 많아 잠시 제한되었습니다. 잠시 후 다시 시도해 주세요.");
  });

  it("uses the server error text, otherwise the status line", async () => {
    expect((await readApiError(jsonResponse(404, { error: "논문을 찾을 수 없습니다." }))).message).toBe(
      "논문을 찾을 수 없습니다.",
    );
    const bare = await readApiError(new Response("<html>", { status: 502, statusText: "Bad Gateway" }));
    expect(bare.message).toBe("요청이 실패했습니다. (HTTP 502 Bad Gateway)");
    expect(bare.payload).toBeNull();
  });
});

describe("postJson", () => {
  it("sends the csrftoken cookie as X-CSRFToken and the body as JSON", async () => {
    document.cookie = "other=1";
    document.cookie = "csrftoken=abc%3D%3D";
    const fetchMock = stubFetch(jsonResponse(200, { ok: true }));

    await expect(postJson("/papers/x/summary/", { model: "gpt-5" })).resolves.toEqual({ ok: true });

    const [url, init] = fetchMock.mock.calls[0] ?? [];
    expect(url).toBe("/papers/x/summary/");
    expect(init?.method).toBe("POST");
    expect(init?.credentials).toBe("same-origin");
    expect(init?.headers).toEqual({ "Content-Type": "application/json", "X-CSRFToken": "abc==" });
    expect(init?.body).toBe(JSON.stringify({ model: "gpt-5" }));
  });

  it("omits the body, content type and token when there is nothing to send", async () => {
    const fetchMock = stubFetch(jsonResponse(200, {}));

    await postJson("/papers/x/analyze/");

    const init = fetchMock.mock.calls[0]?.[1];
    expect(init?.headers).toEqual({});
    expect(init).not.toHaveProperty("body");
  });

  it("rejects with ApiError when the response is not JSON", async () => {
    stubFetch(new Response("oops", { status: 200 }));

    await expect(postJson("/x")).rejects.toEqual(new ApiError("응답을 해석하지 못했습니다.", 200));
  });
});

describe("getJson", () => {
  it("asks for JSON unless told otherwise and passes the abort signal", async () => {
    const fetchMock = stubFetch(jsonResponse(200, { a: 1 }));
    const controller = new AbortController();

    await getJson("/bootstrap.json", { signal: controller.signal });
    await getJson("/papers/list.json", { accept: false });

    expect(fetchMock.mock.calls[0]?.[1]).toEqual({
      credentials: "same-origin",
      method: "GET",
      headers: { Accept: "application/json" },
      signal: controller.signal,
    });
    expect(fetchMock.mock.calls[1]?.[1]).toEqual({ credentials: "same-origin", method: "GET" });
  });

  it("throws the API error message for failed responses", async () => {
    stubFetch(jsonResponse(500, { error: "논문 데이터를 불러오지 못했습니다. 잠시 후 다시 시도해 주세요." }));

    await expect(getJson("/papers/list.json")).rejects.toThrow("논문 데이터를 불러오지 못했습니다.");
  });
});

describe("getCsrfTokenFromCookie", () => {
  it("returns an empty string without the cookie and the raw value when decoding fails", () => {
    expect(getCsrfTokenFromCookie()).toBe("");
    document.cookie = "csrftoken=%E0%A4%A";
    expect(getCsrfTokenFromCookie()).toBe("%E0%A4%A");
  });
});
