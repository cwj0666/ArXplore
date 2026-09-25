import { Link, useLocation } from "react-router-dom";

import { buildLoginPath } from "../../helpers/loginPath";
import type { AiAccessReason } from "../../pages/detail/detail-types";

export type AiFeature = "overview" | "summary" | "chat";

const MESSAGES: Record<AiFeature, Record<AiAccessReason, string>> = {
  overview: {
    login: "AI 개요를 생성하려면 로그인과 OpenAI 키가 필요합니다.",
    api_key: "AI 개요를 생성하려면 설정에서 개인 OpenAI 키를 등록해 주세요.",
  },
  summary: {
    login: "상세 요약을 생성하려면 로그인과 OpenAI 키가 필요합니다.",
    api_key: "상세 요약을 생성하려면 설정에서 개인 OpenAI 키를 등록해 주세요.",
  },
  chat: {
    login: "논문 채팅을 사용하려면 로그인과 OpenAI 키가 필요합니다.",
    api_key: "논문 채팅을 사용하려면 설정에서 개인 OpenAI 키를 등록해 주세요.",
  },
};

interface AiAccessNoticeProps {
  feature: AiFeature;
  reason: AiAccessReason;
  onOpenSettings: () => void;
  compact?: boolean;
}

export function AiAccessNotice({ feature, reason, onOpenSettings, compact = false }: AiAccessNoticeProps) {
  const location = useLocation();
  const loginPath = buildLoginPath(`${location.pathname}${location.search}`);

  return (
    <div className={`ai-access-notice${compact ? " ai-access-notice-compact" : ""}`} role="note">
      <p>{MESSAGES[feature][reason]}</p>
      {reason === "login" ? (
        <Link className="ai-access-action" to={loginPath}>
          로그인
        </Link>
      ) : (
        <button type="button" className="ai-access-action" onClick={onOpenSettings}>
          설정 열기
        </button>
      )}
    </div>
  );
}
