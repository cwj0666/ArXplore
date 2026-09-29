import { type KeyboardEvent, type SubmitEvent, useRef } from "react";

import type { SearchMode } from "../../shared/api/types";
import { SearchIcon } from "../../shared/ui/icons";
import styles from "./ListSearchPanel.module.css";

const MODE_TABS: { mode: SearchMode; label: string }[] = [
  { mode: "search", label: "키워드 검색" },
  { mode: "ai", label: "AI 어시스턴트" },
];

const SEARCH_MODE_CONFIG: Record<SearchMode, { helper: string; placeholder: string }> = {
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
}

export function ListSearchPanel({
  mode,
  queryInput,
  onModeChange,
  onQueryInputChange,
  onSubmit,
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
    const nextTab = MODE_TABS[(currentIndex + offset + MODE_TABS.length) % MODE_TABS.length];
    if (!nextTab) {
      return;
    }
    onModeChange(nextTab.mode);
    tabRefs.current[nextTab.mode]?.focus();
  };

  const handleSubmit = (event: SubmitEvent<HTMLFormElement>) => {
    event.preventDefault();
    onSubmit();
  };

  return (
    <div className={styles.container}>
      <div className={styles.shell}>
        <div className={styles.modeRow}>
          <div className={styles.modeSegment} role="tablist" aria-label="검색 모드 선택">
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
                  className={selected ? `${styles.modeChip} ${styles.active}` : styles.modeChip}
                  data-mode={tab.mode}
                  onClick={() => {
                    onModeChange(tab.mode);
                  }}
                  onKeyDown={handleTabKeyDown}
                >
                  {tab.label}
                </button>
              );
            })}
          </div>
          <div className={styles.helper} id="search-mode-helper">
            {config.helper}
          </div>
        </div>

        <form
          className={styles.searchBar}
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
            onChange={(event) => {
              onQueryInputChange(event.target.value);
            }}
          />
          <div className={styles.actions}>
            <button type="submit" className={styles.submit}>
              <SearchIcon />
              Search
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
