import { Link } from "react-router-dom";


interface AssistantNoticeProps {
  isAuthenticated: boolean;
  homeHref?: string;
  onRequireLogin: () => void;
  onOpenSettings: () => void;
}


export function AssistantNotice({
  isAuthenticated,
  homeHref = "/",
  onRequireLogin,
  onOpenSettings,
}: AssistantNoticeProps) {
  return (
    <div className="assistant-notice">
      <p>
        {isAuthenticated
          ? "개인 API 키를 등록하면 AI 어시스턴트를 사용할 수 있습니다."
          : "AI 어시스턴트는 로그인 후 개인 API 키를 등록해야 사용할 수 있습니다."}
      </p>
      <div className="assistant-notice-actions">
        {isAuthenticated ? (
          <button type="button" onClick={onOpenSettings}>
            API 키 등록하기
          </button>
        ) : (
          <button type="button" onClick={onRequireLogin}>
            로그인
          </button>
        )}
        <Link to={homeHref}>뒤로가기</Link>
      </div>
    </div>
  );
}
