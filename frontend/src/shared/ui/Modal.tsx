import { type MouseEvent, type ReactNode, type RefObject, useRef } from "react";

import styles from "./Modal.module.css";
import { useModalDialog } from "./useModalDialog";

interface ModalProps {
  open: boolean;
  onClose: () => void;
  labelledBy: string;
  initialFocusRef?: RefObject<HTMLElement | null>;
  children: ReactNode;
}

export function Modal({ open, onClose, labelledBy, initialFocusRef, children }: ModalProps) {
  const dialogRef = useRef<HTMLDivElement | null>(null);
  useModalDialog(open, dialogRef, onClose, initialFocusRef);

  if (!open) {
    return null;
  }

  const handleBackdropClick = (event: MouseEvent<HTMLDivElement>) => {
    if (event.target === event.currentTarget) {
      onClose();
    }
  };

  return (
    <div className={styles.overlay} role="presentation" onClick={handleBackdropClick}>
      <div
        ref={dialogRef}
        className={styles.dialog}
        role="dialog"
        aria-modal="true"
        aria-labelledby={labelledBy}
      >
        {children}
      </div>
    </div>
  );
}
