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


/** Non-2xx 응답을 ApiError로 변환한다. 본문의 `error` 문자열이 있으면 메시지로 쓴다. */
export async function readApiError(response: Response): Promise<ApiError> {
  const body = await readJsonBody(response);
  const payload = body.ok ? body.value : null;
  return new ApiError(extractErrorMessage(payload) ?? describeStatus(response), response.status, payload);
}


async function performJsonRequest<T>(input: RequestInfo | URL, init: RequestInit | undefined): Promise<T> {
  const response = await fetch(input, {
    credentials: "same-origin",
    ...init,
  });

  if (!response.ok) {
    throw await readApiError(response);
  }

  const body = await readJsonBody(response);
  if (!body.ok) {
    throw new ApiError("응답을 해석하지 못했습니다.", response.status);
  }
  return body.value as T;
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
  return performJsonRequest<T>(input, init);
}


/** 2xx가 아니면 ApiError를 던진다. */
export async function requestJsonWithBody<T>(
  input: RequestInfo | URL,
  method: "POST" | "DELETE",
  body?: unknown,
  signal?: AbortSignal,
): Promise<T> {
  return performJsonRequest<T>(input, buildBodyInit(method, body, signal));
}


export function getErrorMessage(error: unknown, fallback: string): string {
  if (error instanceof Error && error.message) {
    return error.message;
  }
  return fallback;
}


export function getApiErrorMessage(error: unknown, fallback: string): string {
  if (error instanceof ApiError && error.message) {
    return error.message;
  }
  return fallback;
}


export function hasErrorPayload(error: ApiError): boolean {
  return extractErrorMessage(error.payload) !== null;
}


export { getCsrfTokenFromCookie };
