const UNSAFE_CHARACTERS = /[\\\u0000-\u001F\u007F]/;

export function sanitizeNextPath(raw: string | null | undefined, fallback = "/"): string {
  if (!raw) {
    return fallback;
  }

  const value = raw.trim();
  if (!value || value.startsWith("//") || UNSAFE_CHARACTERS.test(raw)) {
    return fallback;
  }

  const origin = window.location.origin;
  let parsed: URL;
  try {
    parsed = new URL(value, origin);
  } catch {
    return fallback;
  }

  if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
    return fallback;
  }
  if (parsed.origin !== origin) {
    return fallback;
  }

  return `${parsed.pathname}${parsed.search}${parsed.hash}` || fallback;
}
