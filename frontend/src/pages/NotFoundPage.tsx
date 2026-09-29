import { Link } from "react-router";

import { FallbackScreen } from "../shared/ui/FallbackScreen";

export default function NotFoundPage() {
  return (
    <FallbackScreen title="페이지를 찾을 수 없습니다" message="주소가 바뀌었거나 없는 페이지입니다.">
      <Link to="/">논문 목록으로</Link>
    </FallbackScreen>
  );
}
