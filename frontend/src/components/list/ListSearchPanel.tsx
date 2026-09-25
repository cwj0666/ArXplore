import { type FormEvent, type KeyboardEvent, useRef } from "react";
import type { SearchMode } from "../../pages/list/listTypes";

const MODE_TABS: { mode: SearchMode; label: string }[] = [
  { mode: "search", label: "키워드 검색" },
  { mode: "ai", label: "AI 어시스턴트" },
];

const SEARCH_MODE_CONFIG: Record<
  SearchMode,
  { helper: string; placeholder: string }
> = {
  search: {
    helper: "Search papers by title or abstract.",
    placeholder: "Search papers by title or abstract",
  },
  ai: {
    helper: "Search any papers with AI.",
    placeholder: "Search any papers with AI",
  },
};

interface ListSearchPanelProps {
  mode: SearchMode;
  queryInput: string;
  onModeChange: (mode: SearchMode) => void;
  onQueryInputChange: (value: string) => void;
  onSubmit: () => void;
  busy: boolean;
}

export function ListSearchPanel({
  mode,
  queryInput,
  onModeChange,
  onQueryInputChange,
  onSubmit,
  busy,
}: ListSearchPanelProps) {
  const config = SEARCH_MODE_CONFIG[mode];
  const tabRefs = useRef<Record<SearchMode, HTMLButtonElement | null>>({ search: null, ai: null });

  const handleTabKeyDown = (event: KeyboardEvent<HTMLButtonElement>) => {
    if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") {
      return;
    }
    event.preventDefault();
    const currentIndex = MODE_TABS.findIndex((tab) => tab.mode === mode);
    const offset = event.key === "ArrowRight" ? 1 : -1;
    const nextMode = MODE_TABS[(currentIndex + offset + MODE_TABS.length) % MODE_TABS.length].mode;
    onModeChange(nextMode);
    tabRefs.current[nextMode]?.focus();
  };

  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    onSubmit();
  };

  return (
    <div className="smart-search-container">
      <div className="search-shell">
        <div className="search-mode-row">
          <div className="search-mode-segment" role="tablist" aria-label="검색 모드 선택">
            {MODE_TABS.map((tab) => {
              const selected = mode === tab.mode;
              return (
                <button
                  key={tab.mode}
                  ref={(element) => {
                    tabRefs.current[tab.mode] = element;
                  }}
                  type="button"
                  role="tab"
                  id={`search-mode-tab-${tab.mode}`}
                  aria-selected={selected}
                  aria-controls="search-mode-panel"
                  tabIndex={selected ? 0 : -1}
                  className={`mode-chip${selected ? " active" : ""}`}
                  data-mode={tab.mode}
                  onClick={() => onModeChange(tab.mode)}
                  onKeyDown={handleTabKeyDown}
                >
                  {tab.label}
                </button>
              );
            })}
          </div>
          <div className="search-mode-helper" id="search-mode-helper">{config.helper}</div>
        </div>

        <form
          className="pill-search-bar"
          id="search-mode-panel"
          role="tabpanel"
          aria-labelledby={`search-mode-tab-${mode}`}
          onSubmit={handleSubmit}
        >
          <input
            type="text"
            name="q"
            aria-label={config.placeholder}
            aria-describedby="search-mode-helper"
            value={queryInput}
            placeholder={config.placeholder}
            autoComplete="off"
            onChange={(event) => onQueryInputChange(event.target.value)}
          />
          <div className="search-actions-right">
            <button type="submit" className="submit-btn-premium" disabled={busy}>
              <svg
                width="18"
                height="18"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2.5"
                strokeLinecap="round"
                strokeLinejoin="round"
                aria-hidden="true"
              >
                <circle cx="11" cy="11" r="8" />
                <line x1="21" y1="21" x2="16.65" y2="16.65" />
              </svg>
              Search
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
