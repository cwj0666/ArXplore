import { toSafeHttpUrl } from "../../shared/lib/safeUrl";
import buttons from "./buttons.module.css";
import styles from "./PdfPanel.module.css";

interface PdfPanelProps {
  visible: boolean;
  pdfUrl: string | null;
  onClose: () => void;
}

export function PdfPanel({ visible, pdfUrl, onClose }: PdfPanelProps) {
  const src = visible ? toSafeHttpUrl(pdfUrl) : "";
  return (
    <div className={visible ? styles.panel : `${styles.panel} ${styles.hidden}`} id="pdf-panel">
      <div className={styles.header}>
        <button
          type="button"
          className={`${buttons.layoutCtrlBtn} ${buttons.pdfCloseBtn}`}
          title="분할창 닫기"
          aria-label="분할창 닫기"
          onClick={onClose}
        >
          &times;
        </button>
      </div>
      <iframe id="pdf-frame" src={src || undefined} title="Paper PDF" loading="lazy" />
    </div>
  );
}
