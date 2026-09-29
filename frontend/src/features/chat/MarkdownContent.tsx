import Markdown, { type Components, type Options } from "react-markdown";
import rehypeSanitize, { defaultSchema } from "rehype-sanitize";
import remarkBreaks from "remark-breaks";
import remarkGfm from "remark-gfm";

import { toInternalPaperHref } from "../../shared/lib/arxiv";
import styles from "./MarkdownContent.module.css";

const HTTP_URL = /^https?:\/\//i;

const sanitizeSchema = {
  ...defaultSchema,
  protocols: { ...defaultSchema.protocols, href: ["http", "https"], src: ["http", "https"] },
};

export function resolveMarkdownHref(href: string | undefined): string | null {
  if (!href || !HTTP_URL.test(href)) {
    return null;
  }
  return toInternalPaperHref(href) ?? href;
}

const components: Components = {
  a: ({ href, children }) => {
    const target = resolveMarkdownHref(href);
    return target ? (
      <a href={target} target="_blank" rel="noopener noreferrer">
        {children}
      </a>
    ) : (
      <>{children}</>
    );
  },
  img: ({ alt }) => (alt ? <>{alt}</> : null),
  h1: ({ children }) => <h3>{children}</h3>,
  h2: ({ children }) => <h4>{children}</h4>,
  h3: ({ children }) => <h5>{children}</h5>,
  h4: ({ children }) => <h5>{children}</h5>,
  h5: ({ children }) => <h5>{children}</h5>,
  h6: ({ children }) => <h5>{children}</h5>,
};

const remarkPlugins: Options["remarkPlugins"] = [remarkGfm, remarkBreaks];
const rehypePlugins: Options["rehypePlugins"] = [[rehypeSanitize, sanitizeSchema]];
const remarkRehypeOptions: Options["remarkRehypeOptions"] = { footnoteLabel: "각주" };

export function MarkdownContent({ content }: { content: string }) {
  return (
    <div className={styles.markdown}>
      <Markdown
        remarkPlugins={remarkPlugins}
        rehypePlugins={rehypePlugins}
        remarkRehypeOptions={remarkRehypeOptions}
        components={components}
      >
        {content}
      </Markdown>
    </div>
  );
}
