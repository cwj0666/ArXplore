import { Component, type ErrorInfo, type ReactNode } from "react";


interface ErrorBoundaryProps {
  children: ReactNode;
}


interface ErrorBoundaryState {
  error: Error | null;
}


export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  state: ErrorBoundaryState = { error: null };

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    console.error("Unhandled UI error", error, info.componentStack);
  }

  render() {
    if (!this.state.error) {
      return this.props.children;
    }

    return (
      <main className="app-fallback" role="alert">
        <h1>화면을 표시하지 못했습니다</h1>
        <p>예상하지 못한 오류가 발생했습니다. 새로고침하거나 첫 화면으로 돌아가 주세요.</p>
        <div className="app-fallback-actions">
          <button type="button" onClick={() => window.location.reload()}>
            새로고침
          </button>
          <a href="/">첫 화면으로</a>
        </div>
      </main>
    );
  }
}
