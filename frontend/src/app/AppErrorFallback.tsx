import { FallbackScreen } from "../shared/ui/FallbackScreen";

export function AppErrorFallback() {
  return (
    <FallbackScreen
      alert
      title="화면을 표시하지 못했습니다"
      message="예상하지 못한 오류가 발생했습니다. 새로고침하거나 첫 화면으로 돌아가 주세요."
    >
      <button
        type="button"
        onClick={() => {
          window.location.reload();
        }}
      >
        새로고침
      </button>
      <a href="/">첫 화면으로</a>
    </FallbackScreen>
  );
}
