import { Spinner } from "../../shared/ui/Spinner";
import styles from "./AnalysisCards.module.css";
import cards from "./cards.module.css";
import notices from "./notices.module.css";
import type { SummaryBlock } from "./summaryBlocks";
import type { AiSectionState } from "./useAbortableMutation";

const CANCELLED_TEXT =
  "요청을 취소했습니다. 서버에서는 처리가 계속될 수 있어서, 잠시 후 다시 시도하면 저장된 결과가 바로 표시될 수 있습니다.";

export function AbstractCard({ abstractText }: { abstractText: string }) {
  return (
    <section className={cards.card} aria-labelledby="abstract-title">
      <h2 className={cards.sectionTitle} id="abstract-title">
        초록
      </h2>
      <div className={cards.overviewText}>{abstractText}</div>
    </section>
  );
}

function AiProgress({ message, onCancel }: { message: string; onCancel: () => void }) {
  return (
    <div className={styles.progress} role="status" aria-live="polite">
      <div className={styles.skeleton} aria-hidden="true">
        <span />
        <span />
        <span />
        <span className={styles.short} />
      </div>
      <div className={styles.progressRow}>
        <Spinner />
        <span className={styles.progressText}>{message}</span>
        <button type="button" className={styles.secondaryBtn} onClick={onCancel}>
          취소
        </button>
      </div>
      <p className={styles.progressHint}>
        취소하면 화면에서만 기다림을 멈춥니다. 서버에서는 요청이 계속 처리될 수 있습니다.
      </p>
    </div>
  );
}

interface AiStatusBodyProps {
  state: AiSectionState;
  loadingMessage: string;
  onCancel: () => void;
  onRetry: () => void;
}

function AiStatusBody({ state, loadingMessage, onCancel, onRetry }: AiStatusBodyProps) {
  switch (state.status) {
    case "loading":
      return <AiProgress message={loadingMessage} onCancel={onCancel} />;
    case "error":
      return (
        <div className={styles.statusBlock}>
          <div className={`${notices.errorBox} ${styles.fullWidth}`} role="alert">
            {state.message}
          </div>
          <button type="button" className={styles.secondaryBtn} onClick={onRetry}>
            다시 시도
          </button>
        </div>
      );
    case "cancelled":
      return (
        <div className={styles.statusBlock}>
          <div className={`${notices.infoBox} ${styles.fullWidth}`}>{CANCELLED_TEXT}</div>
          <button type="button" className={styles.secondaryBtn} onClick={onRetry}>
            다시 시도
          </button>
        </div>
      );
    default:
      return null;
  }
}

function splitParagraphs(text: string): string[] {
  return text
    .replace(/\r\n/g, "\n")
    .split(/\n\s*\n/)
    .map((paragraph) => paragraph.trim())
    .filter(Boolean);
}

interface OverviewCardProps {
  state: AiSectionState;
  overviewText: string;
  onCancel: () => void;
  onRetry: () => void;
}

export function OverviewCard({ state, overviewText, onCancel, onRetry }: OverviewCardProps) {
  if (state.status === "idle") {
    return null;
  }
  const paragraphs = state.status === "ready" && overviewText ? splitParagraphs(overviewText) : [];

  return (
    <section
      className={cards.card}
      id="overview-card"
      aria-labelledby="overview-title"
      aria-busy={state.status === "loading"}
    >
      <h2 className={cards.sectionTitle} id="overview-title">
        개요
      </h2>
      {paragraphs.length > 0 ? (
        <div id="overview-content" className={cards.overviewText}>
          {paragraphs.map((text, index) => (
            <p key={`p-${index}`}>{text}</p>
          ))}
        </div>
      ) : null}
      {state.status === "ready" && paragraphs.length === 0 ? (
        <div className={notices.infoBox}>생성된 개요가 없습니다.</div>
      ) : null}
      <AiStatusBody
        state={state}
        loadingMessage="AI가 논문을 분석하고 있습니다. 처음 분석하는 논문은 시간이 조금 걸립니다."
        onCancel={onCancel}
        onRetry={onRetry}
      />
    </section>
  );
}

export function FindingsCard({ findings }: { findings: string[] }) {
  if (!findings.length) {
    return null;
  }
  return (
    <section className={cards.card} id="findings-card" aria-labelledby="findings-title">
      <h2 className={cards.sectionTitle} id="findings-title">
        핵심 포인트
      </h2>
      <ul className={styles.keyFindings} id="findings-list">
        {findings.map((finding, index) => (
          <li key={`${index}-${finding.slice(0, 24)}`}>{finding}</li>
        ))}
      </ul>
    </section>
  );
}

interface SummaryCardProps {
  state: AiSectionState;
  blocks: SummaryBlock[];
  onCancel: () => void;
  onRetry: () => void;
}

export function SummaryCard({ state, blocks, onCancel, onRetry }: SummaryCardProps) {
  if (state.status === "idle") {
    return null;
  }
  const showBlocks = state.status === "ready" && blocks.length > 0;

  return (
    <section
      id="summary-card"
      className={cards.card}
      aria-label="상세 요약"
      aria-busy={state.status === "loading"}
    >
      {state.status !== "ready" ? <h2 className={cards.sectionTitle}>상세 요약</h2> : null}
      {showBlocks ? (
        <div id="summary-content" className={styles.summaryText}>
          {blocks.map((block, index) =>
            block.type === "heading" ? (
              <h2 key={`${index}-${block.text}`}>{block.text}</h2>
            ) : (
              <p key={`${index}-${block.text}`}>{block.text}</p>
            ),
          )}
        </div>
      ) : null}
      {state.status === "ready" && !showBlocks ? (
        <div className={notices.infoBox}>생성된 요약이 없습니다.</div>
      ) : null}
      <AiStatusBody
        state={state}
        loadingMessage="AI가 상세 요약을 만들고 있습니다. 섹션별로 요약하므로 시간이 걸릴 수 있습니다."
        onCancel={onCancel}
        onRetry={onRetry}
      />
    </section>
  );
}
