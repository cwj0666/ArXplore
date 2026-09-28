import { useMemo } from "react";

import styles from "./MarkdownContent.module.css";
import { renderMarkdownToHtml } from "./renderMarkdown";

export function MarkdownContent({ content }: { content: string }) {
  const html = useMemo(() => renderMarkdownToHtml(content), [content]);
  return <div className={styles.markdown} dangerouslySetInnerHTML={{ __html: html }} />;
}
