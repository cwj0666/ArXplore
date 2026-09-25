import { useEffect, useMemo, useRef, useState } from "react";
import { Navigate, useLocation, useParams } from "react-router-dom";

import { toggleFavorite } from "../../helpers/accountApi";
import { ApiError, getErrorMessage } from "../../helpers/http";
import { AnalyzeOverlay } from "../../components/detail/AnalyzeOverlay";
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
import {
  fetchPaperAnalysis,
  fetchPaperDetail,
  fetchPaperSummary,
} from "./detail-api";
import { formatSummaryBlocks, type SummaryBlock } from "./detail-summary";
import type { PaperDetail } from "./detail-types";
import type { BootstrapPayload, FavoriteTogglePayload } from "../../types/app";
import "./detail-page.css";


interface PaperDetailPageProps {
  session: BootstrapPayload;
  onRequireLogin: () => void;
  onOpenSettings: (tab?: "settings" | "favorites") => void;
  onLogout: () => void;
}


const ANALYZE_OVERLAY_TEXT = "AI가 실시간으로 논문을 분석하고 있습니다...";
const SUMMARY_OVERLAY_TEXT = "AI가 상세 요약을 생성하고 있습니다...";


function describeRequestError(error: unknown, prefix: string): string {
  if (error instanceof ApiError) {
    return error.message;
  }
  return `${prefix}: ${getErrorMessage(error, "알 수 없는 오류")}`;
}


export function PaperDetailPage({
  session,
  onRequireLogin,
  onOpenSettings,
  onLogout,
}: PaperDetailPageProps) {
  const { arxivId = "" } = useParams<{ arxivId: string }>();
  const location = useLocation();
  const canUseAi = session.is_authenticated && session.has_personal_api_key;

  const [paper, setPaper] = useState<PaperDetail | null>(null);
  const [pageError, setPageError] = useState("");
  const [pageLoading, setPageLoading] = useState(true);
  const [pdfVisible, setPdfVisible] = useState(false);

  const [analysisLoading, setAnalysisLoading] = useState(false);
  const [overview, setOverview] = useState("");
  const [overviewError, setOverviewError] = useState("");
  const [findings, setFindings] = useState<string[]>([]);

  const [summaryLoading, setSummaryLoading] = useState(false);
  const [summaryBlocks, setSummaryBlocks] = useState<SummaryBlock[]>([]);
  const [summaryError, setSummaryError] = useState("");
  const [selectedSummaryModel, setSelectedSummaryModel] = useState(session.preferred_summary_model);
  const [summaryModelPickerOpen, setSummaryModelPickerOpen] = useState(false);

  const [overlayVisible, setOverlayVisible] = useState(false);
  const [overlayMessage, setOverlayMessage] = useState(ANALYZE_OVERLAY_TEXT);
  const summaryRequestIdRef = useRef(0);

  useEffect(() => {
    setSelectedSummaryModel(session.preferred_summary_model);
  }, [session.preferred_summary_model]);

  useEffect(() => {
    let active = true;

    async function loadDetail() {
      if (!session.is_authenticated) {
        setPageLoading(false);
        return;
      }

      if (!arxivId) {
        setPageError("잘못된 경로입니다.");
        setPageLoading(false);
        return;
      }

      setPageLoading(true);
      setPageError("");
      setPaper(null);
      setPdfVisible(false);
      setAnalysisLoading(false);
      setOverview("");
      setOverviewError("");
      setFindings([]);
      setSummaryLoading(false);
      setSummaryBlocks([]);
      setSummaryError("");
      setOverlayVisible(false);
      setSummaryModelPickerOpen(false);
      setOverlayMessage(ANALYZE_OVERLAY_TEXT);

      try {
        const data = await fetchPaperDetail(arxivId);
        if (!active) {
          return;
        }

        if (data.error || !data.paper) {
          setPageError(data.error ?? "논문을 찾을 수 없습니다.");
          setPageLoading(false);
          return;
        }

        setPaper(data.paper);
        setPageLoading(false);
      } catch (error) {
        if (!active) {
          return;
        }
        setPageError(describeRequestError(error, "데이터 로드 실패"));
        setPageLoading(false);
      }
    }

    void loadDetail();

    return () => {
      active = false;
      summaryRequestIdRef.current += 1;
    };
  }, [arxivId, session.is_authenticated]);

  useEffect(() => {
    let active = true;

    async function loadAnalysis() {
      if (!paper || !canUseAi) {
        return;
      }

      setAnalysisLoading(true);
      setOverviewError("");
      setOverlayMessage(ANALYZE_OVERLAY_TEXT);
      setOverlayVisible(true);

      try {
        const data = await fetchPaperAnalysis(paper.arxiv_id);
        if (!active) {
          return;
        }

        setOverlayVisible(false);
        setAnalysisLoading(false);

        if (data.error) {
          setOverviewError(data.error);
          return;
        }

        if (data.overview) {
          setOverview(data.overview);
        }

        if (Array.isArray(data.key_findings) && data.key_findings.length > 0) {
          setFindings(data.key_findings);
        }
      } catch (error) {
        if (!active) {
          return;
        }
        setOverlayVisible(false);
        setAnalysisLoading(false);
        setOverviewError(describeRequestError(error, "분석 요청 실패"));
      }
    }

    void loadAnalysis();

    return () => {
      active = false;
    };
  }, [canUseAi, paper]);

  const lockedNotice = useMemo(() => {
    return "개인 API 키를 등록하면 개요, 핵심 포인트, 논문 채팅을 사용할 수 있습니다.";
  }, []);

  const summaryButtonLabel = useMemo(() => {
    if (!session.has_personal_api_key) {
      return "API 키 등록하기";
    }
    if (summaryBlocks.length > 0) {
      return "다른 모델로 상세요약";
    }
    return "상세요약 생성하기";
  }, [session.has_personal_api_key, session.is_authenticated, summaryBlocks.length]);

  const handleViewPdf = () => {
    setPdfVisible(true);
    setOverlayVisible(false);
  };

  const handleSummaryAction = () => {
    if (!session.has_personal_api_key) {
      onOpenSettings();
      return;
    }
    setSummaryModelPickerOpen(true);
  };

  const handleConfirmSummary = async () => {
    if (!paper || !canUseAi) {
      return;
    }

    const requestId = ++summaryRequestIdRef.current;
    const isStale = () => requestId !== summaryRequestIdRef.current;

    setSummaryModelPickerOpen(false);
    setSummaryError("");
    setSummaryLoading(true);
    setOverlayMessage(SUMMARY_OVERLAY_TEXT);
    setOverlayVisible(true);

    try {
      const data = await fetchPaperSummary(paper.arxiv_id, selectedSummaryModel);
      if (isStale()) {
        return;
      }
      setOverlayVisible(false);
      setSummaryLoading(false);

      if (data.error) {
        setSummaryError(data.error);
        return;
      }

      setSummaryBlocks(formatSummaryBlocks(data.summary ?? ""));
    } catch (error) {
      if (isStale()) {
        return;
      }
      setOverlayVisible(false);
      setSummaryLoading(false);
      setSummaryError(describeRequestError(error, "요약 생성 실패"));
    }
  };

  const handleFavoriteToggle = async (paperId: string) => {
    let payload: FavoriteTogglePayload;
    try {
      payload = await toggleFavorite(paperId);
    } catch (error) {
      if (error instanceof ApiError && error.status === 401) {
        onRequireLogin();
      }
      return;
    }
    setPaper((previous) =>
      previous ? { ...previous, is_favorited: payload.is_favorited ?? false } : previous,
    );
  };

  if (pageError) {
    return (
      <div className="detail-page">
        <div className="page-error-wrap">
          <div className="error-box">{pageError}</div>
          {arxivId ? (
            <a
              href={`https://arxiv.org/abs/${arxivId}`}
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

  if (!session.is_authenticated) {
    return <Navigate replace to={`/login/?next=${encodeURIComponent(`${location.pathname}${location.search}`)}`} />;
  }

  if (pageLoading || !paper) {
    return (
      <div className="detail-page">
        <div className="page-error-wrap">
          <div className="info-box">논문을 불러오는 중입니다...</div>
        </div>
      </div>
    );
  }

  return (
    <div className="detail-page">
      <AnalyzeOverlay
        title={paper.title}
        message={overlayMessage}
        visible={canUseAi && overlayVisible}
      />

      <DetailTopBar
        pdfUrl={paper.pdf_url}
        summaryLoading={summaryLoading}
        summaryLabel={summaryButtonLabel}
        showSummaryAction
        session={session}
        onViewPdf={handleViewPdf}
        onGenerateSummary={handleSummaryAction}
        onOpenSettings={onOpenSettings}
        onLogout={onLogout}
      />

      {summaryModelPickerOpen ? (
        <div className="summary-model-overlay" onClick={() => setSummaryModelPickerOpen(false)}>
          <div className="summary-model-dialog" onClick={(event) => event.stopPropagation()}>
            <h2>상세요약 모델 선택</h2>
            <select value={selectedSummaryModel} onChange={(event) => setSelectedSummaryModel(event.target.value)}>
              {session.available_summary_models.map((model) => (
                <option key={model} value={model}>
                  {model.replace(/-/g, " ").toUpperCase()}
                </option>
              ))}
            </select>
            <div className="summary-model-actions">
              <button type="button" onClick={() => setSummaryModelPickerOpen(false)}>
                취소
              </button>
              <button type="button" className="primary" onClick={() => void handleConfirmSummary()}>
                생성
              </button>
            </div>
          </div>
        </div>
      ) : null}

      <div className={`layout ${pdfVisible ? "show-pdf" : ""}`} id="main-layout">
        <PdfPanel
          visible={pdfVisible}
          pdfUrl={paper.pdf_url}
          onClose={() => setPdfVisible(false)}
        />

        <div className="main-panel">
          <PaperHeroCard
            paper={paper}
            isFavorited={Boolean(paper.is_favorited)}
            canFavorite={session.is_authenticated}
            onToggleFavorite={() => { void handleFavoriteToggle(paper.arxiv_id); }}
            onRequireLogin={onRequireLogin}
          />

          {!canUseAi ? (
            <>
              <AbstractCard abstractText={paper.abstract} noticeText={lockedNotice} />
              <div className="info-box">상단 설정에서 API 키를 등록하면 AI 기능을 사용할 수 있습니다.</div>
            </>
          ) : (
            <>
              <OverviewCard
                loading={analysisLoading}
                overviewText={overview}
                errorText={overviewError}
              />
              <FindingsCard findings={findings} />
              <SummaryCard blocks={summaryBlocks} errorText={summaryError} />
            </>
          )}
        </div>
      </div>

      {paper.related_papers && paper.related_papers.length > 0 ? (
        <div className="detail-bottom-section">
          <RelatedPapersCard papers={paper.related_papers} />
        </div>
      ) : null}

      {canUseAi ? <ChatPanel arxivId={paper.arxiv_id} /> : null}
    </div>
  );
}
