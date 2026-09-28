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

export function getCsrfTokenFromCookie(): string {
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

function extractErrorMessage(payload: unknown): string | null {
  if (payload && typeof payload === "object" && "error" in payload) {
    const { error } = payload;
    if (typeof error === "string" && error) {
      return error;
    }
  }
  return null;
}

function describeStatus(response: Response): string {
  const statusLabel = response.statusText
    ? `${response.status} ${response.statusText}`
    : `${response.status}`;
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
    payload && typeof payload === "object" && "retry_after" in payload ? Number(payload.retry_after) : NaN;
  const fromHeader = Number(response.headers.get("Retry-After"));
  const seconds = Number.isFinite(fromPayload) && fromPayload > 0 ? fromPayload : fromHeader;
  return Number.isFinite(seconds) && seconds > 0 ? Math.ceil(seconds) : null;
}

function describeRateLimit(retryAfterSeconds: number | null): string {
  return retryAfterSeconds
    ? `요청이 많아 잠시 제한되었습니다. ${retryAfterSeconds}초 후 다시 시도해 주세요.`
    : "요청이 많아 잠시 제한되었습니다. 잠시 후 다시 시도해 주세요.";
}

export async function readApiError(response: Response): Promise<ApiError> {
  const body = await readJsonBody(response);
  const payload = body.ok ? body.value : null;
  if (response.status === 429) {
    return new ApiError(
      describeRateLimit(parseRetryAfterSeconds(payload, response)),
      response.status,
      payload,
    );
  }
  return new ApiError(extractErrorMessage(payload) ?? describeStatus(response), response.status, payload);
}

async function performJsonRequest<T>(input: string, init: RequestInit): Promise<T> {
  const response = await fetch(input, { credentials: "same-origin", ...init });
  if (!response.ok) {
    throw await readApiError(response);
  }
  const body = await readJsonBody(response);
  if (!body.ok) {
    throw new ApiError("응답을 해석하지 못했습니다.", response.status);
  }
  return body.value as T;
}

export function getJson<T>(
  input: string,
  { signal, accept = true }: { signal?: AbortSignal | undefined; accept?: boolean } = {},
): Promise<T> {
  const init: RequestInit = { method: "GET" };
  if (accept) {
    init.headers = { Accept: "application/json" };
  }
  if (signal) {
    init.signal = signal;
  }
  return performJsonRequest<T>(input, init);
}

export function postJson<T>(input: string, body?: unknown, signal?: AbortSignal): Promise<T> {
  const csrfToken = getCsrfTokenFromCookie();
  const headers: Record<string, string> = {};
  if (body !== undefined) {
    headers["Content-Type"] = "application/json";
  }
  if (csrfToken) {
    headers["X-CSRFToken"] = csrfToken;
  }
  const init: RequestInit = { method: "POST", headers };
  if (body !== undefined) {
    init.body = JSON.stringify(body);
  }
  if (signal) {
    init.signal = signal;
  }
  return performJsonRequest<T>(input, init);
}

export function getErrorMessage(error: unknown, fallback: string): string {
  if (error instanceof Error && error.message) {
    return error.message;
  }
  return fallback;
}
