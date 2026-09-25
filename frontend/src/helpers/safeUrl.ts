export function toSafeHttpUrl(value: string | null | undefined, fallback = ""): string {
  const trimmed = value?.trim() ?? "";
  return /^https?:\/\//i.test(trimmed) ? trimmed : fallback;
}
