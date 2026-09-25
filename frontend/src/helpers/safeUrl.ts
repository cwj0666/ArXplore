/** Returns the trimmed value only when it is an absolute http(s) URL; anything else (javascript:, data:, relative) yields the fallback. */
export function toSafeHttpUrl(value: string | null | undefined, fallback = ""): string {
  const trimmed = value?.trim() ?? "";
  return /^https?:\/\//i.test(trimmed) ? trimmed : fallback;
}
