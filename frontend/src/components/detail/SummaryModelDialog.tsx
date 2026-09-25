import { useRef } from "react";

import { useModalDialog } from "../../helpers/useModalDialog";

interface SummaryModelDialogProps {
  open: boolean;
  models: string[];
  selectedModel: string;
  confirmLabel: string;
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
  confirmLabel,
  onSelectModel,
  onConfirm,
  onClose,
}: SummaryModelDialogProps) {
  const dialogRef = useRef<HTMLDivElement | null>(null);
  const selectRef = useRef<HTMLSelectElement | null>(null);

  useModalDialog(open, dialogRef, onClose, selectRef);

  if (!open) {
    return null;
  }

  return (
    <div className="summary-model-overlay" onClick={onClose}>
      <div
        ref={dialogRef}
        className="summary-model-dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby="summary-model-title"
        onClick={(event) => event.stopPropagation()}
      >
        <h2 id="summary-model-title">상세요약 모델 선택</h2>
        <select
          ref={selectRef}
          aria-label="요약 모델"
          value={selectedModel}
          onChange={(event) => onSelectModel(event.target.value)}
        >
          {models.map((model) => (
            <option key={model} value={model}>
              {formatModelName(model)}
            </option>
          ))}
        </select>
        <div className="summary-model-actions">
          <button type="button" onClick={onClose}>
            취소
          </button>
          <button type="button" className="primary" onClick={onConfirm}>
            {confirmLabel}
          </button>
        </div>
      </div>
    </div>
  );
}
