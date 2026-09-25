function getCsrfTokenFromCookie(): string {
  if (typeof document === "undefined") {
    return "";
  }

  const token = document.cookie
    .split(";")
    .map((part) => part.trim())
    .find((part) => part.startsWith("csrftoken="));

  if (!token) {
    return "";
  }
  const value = token.split("=")[1] ?? "";
  try {
    return decodeURIComponent(value);
  } catch {
    return value;
  }
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


function parseRetryAfterSeconds(payload: unknown, response: Response): number | null {
  const fromPayload =
    payload && typeof payload === "object" ? Number((payload as { retry_after?: unknown }).retry_after) : NaN;
  const fromHeader = Number(response.headers.get("Retry-After"));
  const seconds = Number.isFinite(fromPayload) && fromPayload > 0 ? fromPayload : fromHeader;
  return Number.isFinite(seconds) && seconds > 0 ? Math.ceil(seconds) : null;
}


function describeRateLimit(retryAfterSeconds: number | null): string {
  return retryAfterSeconds
    ? `요청이 많아 잠시 제한되었습니다. ${retryAfterSeconds}초 후 다시 시도해 주세요.`
    : "요청이 많아 잠시 제한되었습니다. 잠시 후 다시 시도해 주세요.";
}


/** Non-2xx 응답을 ApiError로 변환한다. 본문의 `error` 문자열이 있으면 메시지로 쓰고, 429는 대기 시간을 안내한다. */
export async function readApiError(response: Response): Promise<ApiError> {
  const body = await readJsonBody(response);
  const payload = body.ok ? body.value : null;
  if (response.status === 429) {
    return new ApiError(describeRateLimit(parseRetryAfterSeconds(payload, response)), response.status, payload);
  }
  return new ApiError(extractErrorMessage(payload) ?? describeStatus(response), response.status, payload);
}


/** 회원가입 400 응답의 `password_errors` 목록. 없으면 빈 배열. */
export function getPasswordErrors(error: unknown): string[] {
  if (!(error instanceof ApiError) || !error.payload || typeof error.payload !== "object") {
    return [];
  }
  const raw = (error.payload as { password_errors?: unknown }).password_errors;
  return Array.isArray(raw) ? raw.filter((item): item is string => typeof item === "string" && item.length > 0) : [];
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
