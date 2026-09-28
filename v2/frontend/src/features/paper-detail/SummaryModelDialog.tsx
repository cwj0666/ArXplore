import { useRef } from "react";

import { Modal } from "../../shared/ui/Modal";
import styles from "./SummaryModelDialog.module.css";

interface SummaryModelDialogProps {
  open: boolean;
  models: string[];
  selectedModel: string;
  onSelectModel: (model: string) => void;
  onConfirm: () => void;
  onClose: () => void;
}

function formatModelName(model: string): string {
  return model.replace(/-/g, " ").toUpperCase();
}

export function SummaryModelDialog({
  open,
  models,
  selectedModel,
  onSelectModel,
  onConfirm,
  onClose,
}: SummaryModelDialogProps) {
  const selectRef = useRef<HTMLSelectElement | null>(null);

  return (
    <Modal open={open} onClose={onClose} labelledBy="summary-model-title" initialFocusRef={selectRef}>
      <h2 className={styles.title} id="summary-model-title">
        상세요약 모델 선택
      </h2>
      <select
        ref={selectRef}
        className={styles.select}
        aria-label="요약 모델"
        value={selectedModel}
        onChange={(event) => {
          onSelectModel(event.target.value);
        }}
      >
        {models.map((model) => (
          <option key={model} value={model}>
            {formatModelName(model)}
          </option>
        ))}
      </select>
      <div className={styles.actions}>
        <button type="button" onClick={onClose}>
          취소
        </button>
        <button type="button" className={styles.primary} onClick={onConfirm}>
          생성
        </button>
      </div>
    </Modal>
  );
}
