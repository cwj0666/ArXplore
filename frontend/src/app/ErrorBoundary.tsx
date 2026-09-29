import { Component, type ErrorInfo, type ReactNode } from "react";

import { AppErrorFallback } from "./AppErrorFallback";

interface ErrorBoundaryState {
  error: Error | null;
}

export class ErrorBoundary extends Component<{ children: ReactNode }, ErrorBoundaryState> {
  override state: ErrorBoundaryState = { error: null };

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { error };
  }

  override componentDidCatch(error: Error, info: ErrorInfo): void {
    console.error("Unhandled UI error", error, info.componentStack);
  }

  override render() {
    return this.state.error ? <AppErrorFallback /> : this.props.children;
  }
}
