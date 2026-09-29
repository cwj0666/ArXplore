export interface SummaryBlock {
  type: "heading" | "paragraph";
  text: string;
}

export function formatSummaryBlocks(text: string): SummaryBlock[] {
  if (!text) {
    return [];
  }
  return text
    .split("\n")
    .filter((line) => line.trim().length > 0)
    .map((line) =>
      line.startsWith("## ")
        ? { type: "heading", text: line.replace("## ", "").trim() }
        : { type: "paragraph", text: line },
    );
}
