import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { useParams } from "react-router";

import { PaperChatPanel } from "../features/paper-chat/PaperChatPanel";
import { paperDetailQuery } from "../features/paper-detail/api";
import { DetailError, DetailLoading } from "../features/paper-detail/DetailStatus";
import { PaperDetailView } from "../features/paper-detail/PaperDetailView";
import { describeRequestError } from "../features/paper-detail/requestErrors";
import { useBootstrap } from "../shared/api/bootstrap";
import styles from "./PaperDetailPage.module.css";

function PaperDetailContent({ arxivId }: { arxivId: string }) {
  const bootstrap = useBootstrap();
  const [selectedModel, setSelectedModel] = useState(bootstrap.default_summary_model);
  const { data, error } = useQuery(paperDetailQuery(arxivId));

  if (!data) {
    return error ? (
      <DetailError message={describeRequestError(error, "데이터 로드 실패")} arxivId={arxivId} />
    ) : (
      <DetailLoading />
    );
  }

  return (
    <PaperDetailView
      key={arxivId}
      paper={data.paper}
      summaryModels={bootstrap.available_summary_models}
      selectedModel={selectedModel}
      onSelectModel={setSelectedModel}
      chat={<PaperChatPanel arxivId={data.paper.arxiv_id} />}
    />
  );
}

export default function PaperDetailPage() {
  const { arxivId = "" } = useParams();

  return (
    <div className={styles.page}>
      {arxivId ? (
        <PaperDetailContent arxivId={arxivId} />
      ) : (
        <DetailError message="잘못된 경로입니다." arxivId="" />
      )}
    </div>
  );
}
