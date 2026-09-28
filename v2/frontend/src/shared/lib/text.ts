export function truncateText(text: string | null | undefined, max: number): string {
  if (!text) {
    return "";
  }
  return text.length > max ? `${text.slice(0, max)}...` : text;
}

export function authorsToText(authors: readonly string[] | string): string {
  return typeof authors === "string" ? authors : authors.join(", ");
}

export function datePart(value: string): string {
  return value.slice(0, 10);
}
