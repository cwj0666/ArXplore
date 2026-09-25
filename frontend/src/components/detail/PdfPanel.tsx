import { toSafeHttpUrl } from "../../helpers/safeUrl";

interface PdfPanelProps {
  visible: boolean;
  pdfUrl: string;
  onClose: () => void;
}

export function PdfPanel({ visible, pdfUrl, onClose }: PdfPanelProps) {
  return (
    <div className={`pdf-panel ${visible ? "" : "pdf-panel-hidden"}`} id="pdf-panel">
      <div className="pdf-header">
        <button
          type="button"
          className="layout-ctrl-btn pdf-close-btn"
          title="분할창 닫기"
          onClick={onClose}
        >
          &times;
        </button>
      </div>
      <iframe
        id="pdf-frame"
        src={visible ? toSafeHttpUrl(pdfUrl) : ""}
        title="Paper PDF"
        loading="lazy"
      />
    </div>
  );
}
