import { arxivAbsUrl } from "../../shared/lib/arxiv";
import styles from "./DetailStatus.module.css";
import notices from "./notices.module.css";

export function DetailError({ message, arxivId }: { message: string; arxivId: string }) {
  return (
    <div className={styles.wrap}>
      <div className={notices.errorBox} role="alert">
        {message}
      </div>
      {arxivId ? (
        <a href={arxivAbsUrl(arxivId)} target="_blank" rel="noreferrer" className={styles.arxivFallbackBtn}>
          arXiv에서 보기 ↗
        </a>
      ) : null}
    </div>
  );
}

export function DetailLoading() {
  return (
    <div className={styles.wrap}>
      <div className={notices.infoBox} role="status">
        논문을 불러오는 중입니다...
      </div>
    </div>
  );
}
