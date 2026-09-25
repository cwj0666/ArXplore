import { Link } from "react-router-dom";


export function NotFoundPage() {
  return (
    <main className="app-fallback">
      <h1>페이지를 찾을 수 없습니다</h1>
      <p>주소가 바뀌었거나 없는 페이지입니다.</p>
      <div className="app-fallback-actions">
        <Link to="/">논문 목록으로</Link>
      </div>
    </main>
  );
}
