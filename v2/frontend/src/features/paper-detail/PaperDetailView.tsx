import { type ReactNode, useCallback, useEffect, useState } from "react";

import type { PaperDetail } from "../../shared/api/types";
import { AbstractCard, FindingsCard, OverviewCard, SummaryCard } from "./AnalysisCards";
import { requestAnalysis, requestSummary } from "./api";
import { DetailTopBar } from "./DetailTopBar";
import { PaperHeroCard } from "./PaperHeroCard";
import styles from "./PaperDetailView.module.css";
import { PdfPanel } from "./PdfPanel";
import { RelatedPapersCard } from "./RelatedPapersCard";
import { formatSummaryBlocks, type SummaryBlock } from "./summaryBlocks";
import { SummaryModelDialog } from "./SummaryModelDialog";
import { useAbortableMutation } from "./useAbortableMutation";

interface SummaryVariables {
  arxivId: string;
  model: string;
}

function analyze(arxivId: string, signal: AbortSignal) {
  return requestAnalysis(arxivId, signal);
}

function summarize({ arxivId, model }: SummaryVariables, signal: AbortSignal) {
  return requestSummary(arxivId, model, signal);
}

interface PaperDetailViewProps {
  paper: PaperDetail;
  summaryModels: string[];
  selectedModel: string;
  onSelectModel: (model: string) => void;
  chat: ReactNode;
}

export function PaperDetailView({
  paper,
  summaryModels,
  selectedModel,
  onSelectModel,
  chat,
}: PaperDetailViewProps) {
  const [pdfVisible, setPdfVisible] = useState(false);
  const [modelPickerOpen, setModelPickerOpen] = useState(false);
  const [summaryBlocks, setSummaryBlocks] = useState<SummaryBlock[]>([]);
  const analysis = useAbortableMutation(analyze, "분석 요청 실패");
  const summary = useAbortableMutation(summarize, "요약 생성 실패");
  const arxivId = paper.arxiv_id;
  const startAnalysis = analysis.start;
  const startSummary = summary.start;

  useEffect(() => {
    startAnalysis(arxivId);
  }, [arxivId, startAnalysis]);

  const runSummary = useCallback(
    (model: string) => {
      startSummary({ arxivId, model }, (data) => {
        setSummaryBlocks(formatSummaryBlocks(data.summary ?? ""));
      });
    },
    [arxivId, startSummary],
  );

  const closeModelPicker = useCallback(() => {
    setModelPickerOpen(false);
  }, []);

  const confirmSummary = () => {
    setModelPickerOpen(false);
    runSummary(selectedModel);
  };

  const analysisReady = analysis.state.status === "ready";

  return (
    <>
      <DetailTopBar
        pdfUrl={paper.pdf_url}
        summaryLoading={summary.state.status === "loading"}
        summaryLabel={summaryBlocks.length > 0 ? "다른 모델로 상세요약" : "상세요약 생성하기"}
        onViewPdf={() => {
          setPdfVisible(true);
        }}
        onGenerateSummary={() => {
          setModelPickerOpen(true);
        }}
      />

      <SummaryModelDialog
        open={modelPickerOpen}
        models={summaryModels}
        selectedModel={selectedModel}
        onSelectModel={onSelectModel}
        onConfirm={confirmSummary}
        onClose={closeModelPicker}
      />

      <div className={pdfVisible ? `${styles.layout} ${styles.showPdf}` : styles.layout} id="main-layout">
        <PdfPanel
          visible={pdfVisible}
          pdfUrl={paper.pdf_url}
          onClose={() => {
            setPdfVisible(false);
          }}
        />

        <main className={styles.mainPanel}>
          <PaperHeroCard paper={paper} />
          <OverviewCard
            state={analysis.state}
            overviewText={analysis.data?.overview ?? ""}
            onCancel={analysis.cancel}
            onRetry={() => {
              startAnalysis(arxivId);
            }}
          />
          {analysisReady ? <FindingsCard findings={analysis.data?.key_findings ?? []} /> : null}
          <SummaryCard
            state={summary.state}
            blocks={summaryBlocks}
            onCancel={summary.cancel}
            onRetry={() => {
              runSummary(selectedModel);
            }}
          />
          <AbstractCard abstractText={paper.abstract} />
        </main>
      </div>

      {paper.related_papers.length > 0 ? (
        <div className={styles.bottomSection}>
          <RelatedPapersCard papers={paper.related_papers} />
        </div>
      ) : null}

      {chat}
    </>
  );
}
