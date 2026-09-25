import type { AiSectionState } from "../../pages/detail/detail-types";
import type { SummaryBlock } from "../../pages/detail/detail-summary";
import { AiAccessNotice } from "./AiAccessNotice";

const CANCELLED_TEXT =
  "요청을 취소했습니다. 서버에서는 처리가 계속될 수 있어서, 잠시 후 다시 시도하면 저장된 결과가 바로 표시될 수 있습니다.";

interface AbstractCardProps {
  abstractText: string;
}

export function AbstractCard({ abstractText }: AbstractCardProps) {
  return (
    <section className="card" aria-labelledby="abstract-title">
      <h2 className="section-title" id="abstract-title">초록</h2>
      <div className="overview-text">{abstractText}</div>
    </section>
  );
}

interface AiProgressProps {
  message: string;
  onCancel: () => void;
}

function AiProgress({ message, onCancel }: AiProgressProps) {
  return (
    <div className="ai-progress" role="status" aria-live="polite">
      <div className="ai-skeleton" aria-hidden="true">
        <span />
        <span />
        <span />
        <span className="ai-skeleton-short" />
      </div>
      <div className="ai-progress-row">
        <span className="spinner spinner-compact" aria-hidden="true" />
        <span className="ai-progress-text">{message}</span>
        <button type="button" className="ai-secondary-btn" onClick={onCancel}>
          취소
        </button>
      </div>
      <p className="ai-progress-hint">취소하면 화면에서만 기다림을 멈춥니다. 서버에서는 요청이 계속 처리될 수 있습니다.</p>
    </div>
  );
}

interface AiStatusBodyProps {
  state: AiSectionState;
  feature: "overview" | "summary";
  loadingMessage: string;
  onCancel: () => void;
  onRetry: () => void;
  onOpenSettings: () => void;
}

function AiStatusBody({ state, feature, loadingMessage, onCancel, onRetry, onOpenSettings }: AiStatusBodyProps) {
  switch (state.status) {
    case "loading":
      return <AiProgress message={loadingMessage} onCancel={onCancel} />;
    case "denied":
      return <AiAccessNotice feature={feature} reason={state.reason} onOpenSettings={onOpenSettings} />;
    case "error":
      return (
        <div className="ai-status-block">
          <div className="error-box" role="alert">{state.message}</div>
          <button type="button" className="ai-secondary-btn" onClick={onRetry}>
            다시 시도
          </button>
        </div>
      );
    case "cancelled":
      return (
        <div className="ai-status-block">
          <div className="info-box">{CANCELLED_TEXT}</div>
          <button type="button" className="ai-secondary-btn" onClick={onRetry}>
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
    .map((p) => p.trim())
    .filter(Boolean);
}

interface OverviewCardProps {
  state: AiSectionState;
  overviewText: string;
  onCancel: () => void;
  onRetry: () => void;
  onOpenSettings: () => void;
}

export function OverviewCard({ state, overviewText, onCancel, onRetry, onOpenSettings }: OverviewCardProps) {
  if (state.status === "idle") {
    return null;
  }

  const paragraphs = state.status === "ready" && overviewText ? splitParagraphs(overviewText) : [];

  return (
    <section className="card" id="overview-card" aria-labelledby="overview-title" aria-busy={state.status === "loading"}>
      <h2 className="section-title" id="overview-title">개요</h2>
      {paragraphs.length > 0 ? (
        <div id="overview-content" className="overview-text">
          {paragraphs.map((text, idx) => (
            <p key={`p-${idx}`}>{text}</p>
          ))}
        </div>
      ) : null}
      {state.status === "ready" && paragraphs.length === 0 ? (
        <div className="info-box">생성된 개요가 없습니다.</div>
      ) : null}
      <AiStatusBody
        state={state}
        feature="overview"
        loadingMessage="AI가 논문을 분석하고 있습니다. 처음 분석하는 논문은 시간이 조금 걸립니다."
        onCancel={onCancel}
        onRetry={onRetry}
        onOpenSettings={onOpenSettings}
      />
    </section>
  );
}

interface FindingsCardProps {
  findings: string[];
}

export function FindingsCard({ findings }: FindingsCardProps) {
  if (!findings.length) {
    return null;
  }

  return (
    <section className="card" id="findings-card" aria-labelledby="findings-title">
      <h2 className="section-title" id="findings-title">핵심 포인트</h2>
      <ul className="key-findings" id="findings-list">
        {findings.map((finding, idx) => (
          <li key={`${idx}-${finding.slice(0, 24)}`}>{finding}</li>
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
  onOpenSettings: () => void;
}

export function SummaryCard({ state, blocks, onCancel, onRetry, onOpenSettings }: SummaryCardProps) {
  if (state.status === "idle") {
    return null;
  }

  const showBlocks = state.status === "ready" && blocks.length > 0;

  return (
    <section id="summary-card" className="card" aria-label="상세 요약" aria-busy={state.status === "loading"}>
      {state.status !== "ready" ? <h2 className="section-title">상세 요약</h2> : null}
      {showBlocks ? (
        <div id="summary-content" className="summary-text">
          {blocks.map((block, idx) =>
            block.type === "heading" ? (
              <h2 key={`${idx}-${block.text}`}>{block.text}</h2>
            ) : (
              <p key={`${idx}-${block.text}`}>{block.text}</p>
            ),
          )}
        </div>
      ) : null}
      {state.status === "ready" && !showBlocks ? <div className="info-box">생성된 요약이 없습니다.</div> : null}
      <AiStatusBody
        state={state}
        feature="summary"
        loadingMessage="AI가 상세 요약을 만들고 있습니다. 섹션별로 요약하므로 시간이 걸릴 수 있습니다."
        onCancel={onCancel}
        onRetry={onRetry}
        onOpenSettings={onOpenSettings}
      />
    </section>
  );
}
