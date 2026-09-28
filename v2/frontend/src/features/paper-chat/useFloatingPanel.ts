import { type CSSProperties, type MouseEvent as ReactMouseEvent, type RefObject, useState } from "react";

interface PanelRect {
  top: number;
  left: number;
  width: number;
  height: number;
}

const MIN_VISIBLE_HEADER = 40;
const MIN_WIDTH = 300;
const MIN_HEIGHT = 400;
const VIEWPORT_MARGIN = 40;

function trackPointer(onMove: (event: MouseEvent) => void) {
  const onUp = () => {
    document.removeEventListener("mousemove", onMove);
    document.removeEventListener("mouseup", onUp);
  };
  document.addEventListener("mousemove", onMove);
  document.addEventListener("mouseup", onUp);
}

export function useFloatingPanel(panelRef: RefObject<HTMLElement | null>) {
  const [rect, setRect] = useState<PanelRect | null>(null);

  const ensureAbsoluteRect = (): PanelRect | null => {
    if (rect) {
      return rect;
    }
    const panel = panelRef.current;
    if (!panel) {
      return null;
    }
    const bounds = panel.getBoundingClientRect();
    const next = { top: bounds.top, left: bounds.left, width: bounds.width, height: bounds.height };
    setRect(next);
    return next;
  };

  const startDrag = (event: ReactMouseEvent<HTMLElement>) => {
    const target = event.target;
    if (target instanceof HTMLElement && (target.tagName === "INPUT" || target.tagName === "BUTTON")) {
      return;
    }
    const base = ensureAbsoluteRect();
    if (!base) {
      return;
    }
    event.preventDefault();
    const startX = event.clientX;
    const startY = event.clientY;

    trackPointer((moveEvent) => {
      moveEvent.preventDefault();
      const maxTop = window.innerHeight - MIN_VISIBLE_HEADER;
      const maxLeft = window.innerWidth - MIN_VISIBLE_HEADER;
      const top = Math.min(Math.max(base.top + (moveEvent.clientY - startY), 0), maxTop);
      const left = Math.min(Math.max(base.left + (moveEvent.clientX - startX), 0), maxLeft);
      setRect({ top, left, width: base.width, height: base.height });
    });
  };

  const startResize = (event: ReactMouseEvent<HTMLElement>) => {
    const base = ensureAbsoluteRect();
    if (!base) {
      return;
    }
    event.preventDefault();
    event.stopPropagation();
    const startX = event.clientX;
    const startY = event.clientY;

    trackPointer((moveEvent) => {
      const diffX = startX - moveEvent.clientX;
      const diffY = startY - moveEvent.clientY;
      let newWidth = base.width + diffX;
      let newHeight = base.height + diffY;
      let newLeft = base.left - diffX;
      let newTop = base.top - diffY;
      if (newTop < 0) {
        newHeight = base.height + base.top;
        newTop = 0;
      }
      if (newLeft < 0) {
        newWidth = base.width + base.left;
        newLeft = 0;
      }
      const widthFits = newWidth > MIN_WIDTH && newWidth < window.innerWidth - VIEWPORT_MARGIN;
      const heightFits = newHeight > MIN_HEIGHT && newHeight < window.innerHeight - VIEWPORT_MARGIN;
      setRect({
        width: widthFits ? newWidth : base.width,
        left: widthFits ? newLeft : base.left,
        height: heightFits ? newHeight : base.height,
        top: heightFits ? newTop : base.top,
      });
    });
  };

  const style: CSSProperties = rect
    ? {
        top: `${rect.top}px`,
        left: `${rect.left}px`,
        width: `${rect.width}px`,
        height: `${rect.height}px`,
        bottom: "auto",
        right: "auto",
      }
    : {};

  return { style, startDrag, startResize };
}
