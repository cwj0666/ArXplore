import type { KeyboardEvent } from "react";

const IME_PROCESS_KEY_CODE = 229;

export function isImeComposing(event: KeyboardEvent<HTMLElement>): boolean {
  return event.nativeEvent.isComposing || event.keyCode === IME_PROCESS_KEY_CODE;
}
