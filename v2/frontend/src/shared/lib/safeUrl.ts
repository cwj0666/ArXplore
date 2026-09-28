const BACKSLASH = 0x5c;
const LAST_C0_CONTROL = 0x1f;
const DELETE = 0x7f;

function hasUnsafeCharacter(value: string): boolean {
  for (let index = 0; index < value.length; index += 1) {
    const code = value.charCodeAt(index);
    if (code <= LAST_C0_CONTROL || code === DELETE || code === BACKSLASH) {
      return true;
    }
  }
  return false;
}

export function toSafeHttpUrl(value: string | null | undefined, fallback = ""): string {
  const trimmed = value?.trim() ?? "";
  return /^https?:\/\//i.test(trimmed) ? trimmed : fallback;
}

export function isSameOriginPathname(pathname: string): boolean {
  if (pathname.startsWith("//") || hasUnsafeCharacter(pathname)) {
    return false;
  }
  let decoded: string;
  try {
    decoded = decodeURIComponent(pathname);
  } catch {
    return false;
  }
  return !decoded.startsWith("//") && !hasUnsafeCharacter(decoded);
}
