import { type RefObject, useEffect } from "react";

export function useOpenLinksInNewTab(containerRef: RefObject<HTMLElement | null>) {
  useEffect(() => {
    const container = containerRef.current;
    if (!container) {
      return;
    }
    const handleClick = (event: MouseEvent) => {
      const target = event.target;
      const anchor = target instanceof Element ? target.closest("a") : null;
      const href = anchor?.getAttribute("href");
      if (!href) {
        return;
      }
      event.preventDefault();
      window.open(href, "_blank", "noopener,noreferrer");
    };
    container.addEventListener("click", handleClick);
    return () => {
      container.removeEventListener("click", handleClick);
    };
  }, [containerRef]);
}
