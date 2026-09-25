function getCsrfTokenFromCookie(): string {
  if (typeof document === "undefined") {
    return "";
  }

  const token = document.cookie
    .split(";")
    .map((part) => part.trim())
    .find((part) => part.startsWith("csrftoken="));

  return token ? decodeURIComponent(token.split("=")[1] ?? "") : "";
}


export class ApiError extends Error {
  readonly status: number;
  readonly payload: unknown;

  constructor(message: string, status: number, payload: unknown = null) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.payload = payload;
  }
}


function extractErrorMessage(payload: unknown): string | null {
  if (payload && typeof payload === "object" && "error" in payload) {
    const error = (payload as { error?: unknown }).error;
    if (typeof error === "string" && error) {
      return error;
    }
  }
  return null;
}


function describeStatus(response: Response): string {
  const statusLabel = response.statusText ? `${response.status} ${response.statusText}` : `${response.status}`;
  return `요청이 실패했습니다. (HTTP ${statusLabel})`;
}


async function readJsonBody(response: Response): Promise<{ ok: true; value: unknown } | { ok: false }> {
  try {
    return { ok: true, value: await response.json() };
  } catch {
    return { ok: false };
  }
}


interface JsonRequestOptions {
  // Non-2xx responses whose JSON body carries an `error` string resolve instead of throwing,
  // for callers that still branch on `payload.error`.
  resolveErrorPayload?: boolean;
}


async function performJsonRequest<T>(
  input: RequestInfo | URL,
  init: RequestInit | undefined,
  { resolveErrorPayload = false }: JsonRequestOptions,
): Promise<T> {
  const response = await fetch(input, {
    credentials: "same-origin",
    ...init,
  });
  const body = await readJsonBody(response);

  if (response.ok) {
    if (!body.ok) {
      throw new ApiError("응답을 해석하지 못했습니다.", response.status);
    }
    return body.value as T;
  }

  const payload = body.ok ? body.value : null;
  const message = extractErrorMessage(payload);
  if (message && resolveErrorPayload) {
    return payload as T;
  }
  throw new ApiError(message ?? describeStatus(response), response.status, payload);
}


function buildBodyInit(method: "POST" | "DELETE", body: unknown, signal?: AbortSignal): RequestInit {
  const csrfToken = getCsrfTokenFromCookie();
  const headers: Record<string, string> = {};
  if (body !== undefined) {
    headers["Content-Type"] = "application/json";
  }
  if (csrfToken) {
    headers["X-CSRFToken"] = csrfToken;
  }
  return {
    method,
    headers,
    body: body === undefined ? undefined : JSON.stringify(body),
    signal,
  };
}


/** 2xx가 아니면 ApiError를 던진다. */
export async function requestJson<T>(input: RequestInfo | URL, init?: RequestInit): Promise<T> {
  return performJsonRequest<T>(input, init, {});
}


/** 2xx가 아니면 ApiError를 던진다. */
export async function requestJsonWithBody<T>(
  input: RequestInfo | URL,
  method: "POST" | "DELETE",
  body?: unknown,
  signal?: AbortSignal,
): Promise<T> {
  return performJsonRequest<T>(input, buildBodyInit(method, body, signal), {});
}


/**
 * 2xx가 아니면 ApiError를 던지되, `{ error }` JSON 본문은 그대로 반환한다.
 * 기존 호출부(`payload.error` 분기)와의 호환용이며 새 코드는 requestJson을 쓴다.
 */
export async function fetchJson<T>(input: RequestInfo | URL, init?: RequestInit): Promise<T> {
  return performJsonRequest<T>(input, init, { resolveErrorPayload: true });
}


/** fetchJson과 같은 규칙의 POST/DELETE 버전. */
export async function fetchJsonWithBody<T>(
  input: RequestInfo | URL,
  method: "POST" | "DELETE",
  body?: unknown,
): Promise<T> {
  return performJsonRequest<T>(input, buildBodyInit(method, body), { resolveErrorPayload: true });
}


export function getErrorMessage(error: unknown, fallback: string): string {
  if (error instanceof Error && error.message) {
    return error.message;
  }
  return fallback;
}


export { getCsrfTokenFromCookie };
