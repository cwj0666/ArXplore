import type { MouseEvent } from "react";
import { Link, useLocation, useNavigate } from "react-router";

import { toSafeHttpUrl } from "../../shared/lib/safeUrl";
import { ExternalLinkIcon, SplitViewIcon } from "../../shared/ui/icons";
import buttons from "./buttons.module.css";
import styles from "./DetailTopBar.module.css";

interface DetailTopBarProps {
  pdfUrl: string | null;
  summaryLoading: boolean;
  summaryLabel: string;
  onViewPdf: () => void;
  onGenerateSummary: () => void;
}

export function DetailTopBar({
  pdfUrl,
  summaryLoading,
  summaryLabel,
  onViewPdf,
  onGenerateSummary,
}: DetailTopBarProps) {
  const location = useLocation();
  const navigate = useNavigate();

  const handleBack = (event: MouseEvent<HTMLAnchorElement>) => {
    if (location.key !== "default") {
      event.preventDefault();
      void navigate(-1);
    }
  };

  const openFullPdf = () => {
    const safePdfUrl = toSafeHttpUrl(pdfUrl);
    if (safePdfUrl) {
      window.open(safePdfUrl, "_blank", "noopener");
    }
  };

  return (
    <div className={styles.topbar}>
      <div className={styles.left}>
        <Link to="/" className={styles.backBtn} onClick={handleBack}>
          뒤로가기
        </Link>
      </div>
      <div className={styles.center}>
        <Link to="/" className={styles.logo}>
          ArXplore
        </Link>
      </div>
      <div className={styles.right}>
        <button type="button" className={buttons.layoutCtrlBtn} onClick={onViewPdf}>
          <SplitViewIcon className={styles.btnIcon} />
          PDF 분할 보기
        </button>
        <button type="button" className={buttons.layoutCtrlBtn} onClick={openFullPdf}>
          <ExternalLinkIcon className={styles.btnIcon} />
          PDF 전체 화면
        </button>
        <button
          type="button"
          className={styles.summaryBtn}
          onClick={onGenerateSummary}
          disabled={summaryLoading}
          aria-haspopup="dialog"
        >
          {summaryLoading ? "생성 중..." : summaryLabel}
        </button>
      </div>
    </div>
  );
}
