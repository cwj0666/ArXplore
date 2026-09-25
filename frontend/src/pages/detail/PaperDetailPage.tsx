import { useCallback, useEffect, useRef, useState } from "react";
import { useParams } from "react-router-dom";

import {
  AbstractCard,
  FindingsCard,
  OverviewCard,
  SummaryCard,
} from "../../components/detail/AnalysisCards";
import { ChatPanel } from "../../components/detail/ChatPanel";
import { DetailTopBar } from "../../components/detail/DetailTopBar";
import { PaperHeroCard } from "../../components/detail/PaperHeroCard";
import { PdfPanel } from "../../components/detail/PdfPanel";
import { RelatedPapersCard } from "../../components/detail/RelatedPapersCard";
import { SummaryModelDialog } from "../../components/detail/SummaryModelDialog";
import { ApiError, getErrorMessage } from "../../helpers/http";
import type { BootstrapPayload } from "../../types/app";
import {
  fetchPaperAnalysis,
  fetchPaperDetail,
  fetchPaperSummary,
} from "./detail-api";
import { formatSummaryBlocks, type SummaryBlock } from "./detail-summary";
import type { AiSectionState, PaperDetail } from "./detail-types";
import "./detail-page.css";


interface PaperDetailPageProps {
  bootstrap: BootstrapPayload;
}


const IDLE: AiSectionState = { status: "idle" };


function describeRequestError(error: unknown, prefix: string): string {
  if (error instanceof ApiError) {
    return error.message;
  }
  return `${prefix}: ${getErrorMessage(error, "알 수 없는 오류")}`;
}


export function PaperDetailPage({ bootstrap }: PaperDetailPageProps) {
  const { arxivId = "" } = useParams<{ arxivId: string }>();

  const [paper, setPaper] = useState<PaperDetail | null>(null);
  const [pageError, setPageError] = useState("");
  const [pageLoading, setPageLoading] = useState(true);
  const [pdfVisible, setPdfVisible] = useState(false);

  const [analysisState, setAnalysisState] = useState<AiSectionState>(IDLE);
  const [overview, setOverview] = useState("");
  const [findings, setFindings] = useState<string[]>([]);

  const [summaryState, setSummaryState] = useState<AiSectionState>(IDLE);
  const [summaryBlocks, setSummaryBlocks] = useState<SummaryBlock[]>([]);
  const [selectedSummaryModel, setSelectedSummaryModel] = useState(bootstrap.default_summary_model);
  const [summaryModelPickerOpen, setSummaryModelPickerOpen] = useState(false);

  const analysisControllerRef = useRef<AbortController | null>(null);
  const summaryControllerRef = useRef<AbortController | null>(null);

  const abortPending = useCallback(() => {
    const analysisController = analysisControllerRef.current;
    const summaryController = summaryControllerRef.current;
    analysisControllerRef.current = null;
    summaryControllerRef.current = null;
    analysisController?.abort();
    summaryController?.abort();
  }, []);

  useEffect(() => {
    const controller = new AbortController();

    abortPending();
    setPageError("");
    setPaper(null);
    setPdfVisible(false);
    setAnalysisState(IDLE);
    setOverview("");
    setFindings([]);
    setSummaryState(IDLE);
    setSummaryBlocks([]);
    setSummaryModelPickerOpen(false);

    if (!arxivId) {
      setPageError("잘못된 경로입니다.");
      setPageLoading(false);
      return () => controller.abort();
    }

    setPageLoading(true);
    fetchPaperDetail(arxivId, controller.signal)
      .then((data) => {
        if (data.error || !data.paper) {
          setPageError(data.error ?? "논문을 찾을 수 없습니다.");
          return;
        }
        setPaper(data.paper);
      })
      .catch((error: unknown) => {
        if (controller.signal.aborted) {
          return;
        }
        setPageError(describeRequestError(error, "데이터 로드 실패"));
      })
      .finally(() => {
        if (!controller.signal.aborted) {
          setPageLoading(false);
        }
      });

    return () => {
      controller.abort();
      abortPending();
    };
  }, [abortPending, arxivId]);

  const paperId = paper?.arxiv_id ?? "";

  const runAnalysis = useCallback(async (targetId: string) => {
    analysisControllerRef.current?.abort();
    const controller = new AbortController();
    analysisControllerRef.current = controller;
    const isCurrent = () => analysisControllerRef.current === controller;

    setAnalysisState({ status: "loading" });
    try {
      const data = await fetchPaperAnalysis(targetId, controller.signal);
      if (!isCurrent()) {
        return;
      }
      if (data.error) {
        setAnalysisState({ status: "error", message: data.error });
        return;
      }
      setOverview(data.overview ?? "");
      setFindings(Array.isArray(data.key_findings) ? data.key_findings : []);
      setAnalysisState({ status: "ready" });
    } catch (error) {
      if (!isCurrent()) {
        return;
      }
      if (controller.signal.aborted) {
        setAnalysisState({ status: "cancelled" });
        return;
      }
      setAnalysisState({ status: "error", message: describeRequestError(error, "분석 요청 실패") });
    } finally {
      if (isCurrent()) {
        analysisControllerRef.current = null;
      }
    }
  }, []);

  const analysisReady = analysisState.status === "ready";

  useEffect(() => {
    if (!paperId || analysisReady) {
      return;
    }
    void runAnalysis(paperId);
  }, [analysisReady, paperId, runAnalysis]);

  const runSummary = useCallback(async (targetId: string, model: string) => {
    summaryControllerRef.current?.abort();
    const controller = new AbortController();
    summaryControllerRef.current = controller;
    const isCurrent = () => summaryControllerRef.current === controller;

    setSummaryState({ status: "loading" });
    try {
      const data = await fetchPaperSummary(targetId, model, controller.signal);
      if (!isCurrent()) {
        return;
      }
      if (data.error) {
        setSummaryState({ status: "error", message: data.error });
        return;
      }
      setSummaryBlocks(formatSummaryBlocks(data.summary ?? ""));
      setSummaryState({ status: "ready" });
    } catch (error) {
      if (!isCurrent()) {
        return;
      }
      if (controller.signal.aborted) {
        setSummaryState({ status: "cancelled" });
        return;
      }
      setSummaryState({ status: "error", message: describeRequestError(error, "요약 생성 실패") });
    } finally {
      if (isCurrent()) {
        summaryControllerRef.current = null;
      }
    }
  }, []);

  const summaryLoading = summaryState.status === "loading";
  const summaryButtonLabel = summaryBlocks.length > 0 ? "다른 모델로 상세요약" : "상세요약 생성하기";

  const handleConfirmSummary = () => {
    if (!paperId) {
      return;
    }
    setSummaryModelPickerOpen(false);
    void runSummary(paperId, selectedSummaryModel);
  };

  if (pageError) {
    return (
      <div className="detail-page">
        <div className="page-error-wrap">
          <div className="error-box" role="alert">{pageError}</div>
          {arxivId ? (
            <a
              href={`https://arxiv.org/abs/${encodeURIComponent(arxivId)}`}
              target="_blank"
              rel="noreferrer"
              className="arxiv-fallback-btn"
            >
              arXiv에서 보기 ↗
            </a>
          ) : null}
        </div>
      </div>
    );
  }

  if (pageLoading || !paper) {
    return (
      <div className="detail-page">
        <div className="page-error-wrap">
          <div className="info-box" role="status">논문을 불러오는 중입니다...</div>
        </div>
      </div>
    );
  }

  return (
    <div className="detail-page">
      <DetailTopBar
        pdfUrl={paper.pdf_url}
        summaryLoading={summaryLoading}
        summaryLabel={summaryButtonLabel}
        showSummaryAction
        onViewPdf={() => setPdfVisible(true)}
        onGenerateSummary={() => setSummaryModelPickerOpen(true)}
      />

      <SummaryModelDialog
        open={summaryModelPickerOpen}
        models={bootstrap.available_summary_models}
        selectedModel={selectedSummaryModel}
        onSelectModel={setSelectedSummaryModel}
        onConfirm={handleConfirmSummary}
        onClose={() => setSummaryModelPickerOpen(false)}
      />

      <div className={`layout ${pdfVisible ? "show-pdf" : ""}`} id="main-layout">
        <PdfPanel
          visible={pdfVisible}
          pdfUrl={paper.pdf_url}
          onClose={() => setPdfVisible(false)}
        />

        <main className="main-panel">
          <PaperHeroCard paper={paper} />

          <OverviewCard
            state={analysisState}
            overviewText={overview}
            onCancel={() => analysisControllerRef.current?.abort()}
            onRetry={() => void runAnalysis(paper.arxiv_id)}
          />
          {analysisReady ? <FindingsCard findings={findings} /> : null}
          <SummaryCard
            state={summaryState}
            blocks={summaryBlocks}
            onCancel={() => summaryControllerRef.current?.abort()}
            onRetry={() => void runSummary(paper.arxiv_id, selectedSummaryModel)}
          />
          <AbstractCard abstractText={paper.abstract} />
        </main>
      </div>

      {paper.related_papers && paper.related_papers.length > 0 ? (
        <div className="detail-bottom-section">
          <RelatedPapersCard papers={paper.related_papers} />
        </div>
      ) : null}

      <ChatPanel arxivId={paper.arxiv_id} />
    </div>
  );
}
