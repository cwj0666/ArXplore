const UNSAFE_CHARACTERS = /[\\\u0000-\u001F\u007F]/;

export function toSafeHttpUrl(value: string | null | undefined, fallback = ""): string {
  const trimmed = value?.trim() ?? "";
  return /^https?:\/\//i.test(trimmed) ? trimmed : fallback;
}

export function isSameOriginPathname(pathname: string): boolean {
  if (pathname.startsWith("//") || UNSAFE_CHARACTERS.test(pathname)) {
    return false;
  }
  let decoded: string;
  try {
    decoded = decodeURIComponent(pathname);
  } catch {
    return false;
  }
  return !decoded.startsWith("//") && !UNSAFE_CHARACTERS.test(decoded);
}
