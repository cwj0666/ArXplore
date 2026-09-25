import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";

import {
  clearPersonalApiKey,
  fetchFavorites,
  savePersonalApiKey,
} from "../../helpers/accountApi";
import { getApiErrorMessage } from "../../helpers/http";
import { useModalDialog } from "../../helpers/useModalDialog";
import type { BootstrapPayload, FavoriteListPayload } from "../../types/app";
import type { SettingsTab } from "./AccountMenu";


interface SettingsPanelProps {
  open: boolean;
  initialTab: SettingsTab;
  session: BootstrapPayload;
  onClose: () => void;
  onSessionChanged: () => Promise<void>;
}


export function SettingsPanel({
  open,
  initialTab,
  session,
  onClose,
  onSessionChanged,
}: SettingsPanelProps) {
  const [activeTab, setActiveTab] = useState<SettingsTab>(initialTab);
  const [apiKeyInput, setApiKeyInput] = useState("");

  const [statusMessage, setStatusMessage] = useState("");
  const [favorites, setFavorites] = useState<FavoriteListPayload["items"]>([]);
  const [favoritesError, setFavoritesError] = useState("");
  const [isBusy, setIsBusy] = useState(false);
  const panelRef = useRef<HTMLElement | null>(null);

  useModalDialog(open, panelRef, onClose);

  useEffect(() => {
    if (!open) {
      return;
    }
    setActiveTab(initialTab);
    setStatusMessage("");
  }, [initialTab, open]);

  useEffect(() => {
    if (!open || activeTab !== "favorites" || !session.is_authenticated) {
      return;
    }

    let active = true;
    fetchFavorites()
      .then((payload) => {
        if (!active) {
          return;
        }
        setFavorites(payload.items);
        setFavoritesError("");
      })
      .catch((error) => {
        if (!active) {
          return;
        }
        setFavorites([]);
        setFavoritesError(error instanceof Error ? error.message : "즐겨찾기를 불러오지 못했습니다.");
      });

    return () => {
      active = false;
    };
  }, [activeTab, open, session.is_authenticated]);

  if (!open) {
    return null;
  }

  const handleSaveApiKey = async () => {
    setIsBusy(true);
    setStatusMessage("");
    try {
      await savePersonalApiKey(apiKeyInput);
      setApiKeyInput("");
      await onSessionChanged();
      setStatusMessage("API 키를 저장했습니다.");
    } catch (error) {
      setStatusMessage(getApiErrorMessage(error, "API 키를 저장하지 못했습니다."));
    } finally {
      setIsBusy(false);
    }
  };

  const handleClearApiKey = async () => {
    setIsBusy(true);
    setStatusMessage("");
    try {
      await clearPersonalApiKey();
      await onSessionChanged();
      setStatusMessage("API 키를 삭제했습니다.");
    } catch (error) {
      setStatusMessage(getApiErrorMessage(error, "API 키를 삭제하지 못했습니다."));
    } finally {
      setIsBusy(false);
    }
  };

  const tabs: { id: SettingsTab; label: string }[] = [
    { id: "settings", label: "설정" },
    { id: "favorites", label: "즐겨찾기" },
  ];

  return (
    <div className="settings-panel-overlay" onClick={onClose}>
      <aside
        ref={panelRef}
        className="settings-panel"
        role="dialog"
        aria-modal="true"
        aria-labelledby="settings-panel-title"
        tabIndex={-1}
        onClick={(event) => event.stopPropagation()}
      >
        <div className="settings-panel-header">
          <h2 id="settings-panel-title">내 설정</h2>
          <button type="button" onClick={onClose}>
            닫기
          </button>
        </div>

        <div className="settings-tabs" role="tablist" aria-label="설정 탭">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              type="button"
              role="tab"
              id={`settings-tab-${tab.id}`}
              aria-selected={activeTab === tab.id}
              aria-controls="settings-tabpanel"
              className={activeTab === tab.id ? "active" : ""}
              onClick={() => setActiveTab(tab.id)}
            >
              {tab.label}
            </button>
          ))}
        </div>

        <div id="settings-tabpanel" role="tabpanel" aria-labelledby={`settings-tab-${activeTab}`}>
          {activeTab === "settings" ? (
            <div className="settings-section-stack">
              <section className="settings-section">
                <div className="settings-section-title-row">
                  <h3>세션 API 키</h3>
                  <span className={session.has_personal_api_key ? "status-chip active" : "status-chip"}>
                    {session.has_personal_api_key ? "등록됨" : "미등록"}
                  </span>
                </div>
                <p className="settings-help" id="settings-api-key-help">
                  API 키는 현재 로그인 세션 동안에만 서버에 저장됩니다.
                </p>
                <div className="settings-field-row">
                  <input
                    type="password"
                    placeholder="sk-..."
                    aria-label="OpenAI API 키"
                    aria-describedby="settings-api-key-help"
                    autoComplete="off"
                    value={apiKeyInput}
                    onChange={(event) => setApiKeyInput(event.target.value)}
                  />
                  <button type="button" onClick={() => void handleSaveApiKey()} disabled={isBusy}>
                    저장
                  </button>
                  <button type="button" onClick={() => void handleClearApiKey()} disabled={isBusy}>
                    삭제
                  </button>
                </div>
              </section>

              {statusMessage ? (
                <div className="settings-status" role="status">
                  {statusMessage}
                </div>
              ) : null}
            </div>
          ) : (
            <section className="settings-section">
              <h3>즐겨찾기</h3>
              {favoritesError ? <div className="settings-status error">{favoritesError}</div> : null}
              {!favoritesError && favorites.length === 0 ? (
                <div className="settings-help">즐겨찾기한 논문이 없습니다.</div>
              ) : (
                <div className="favorites-list">
                  {favorites.map((paper) => (
                    <Link
                      key={paper.arxiv_id}
                      className="favorite-row"
                      to={`/papers/${encodeURIComponent(paper.arxiv_id)}/`}
                      onClick={onClose}
                    >
                      <strong>{paper.title}</strong>
                      <span>{paper.published_at?.slice(0, 10) ?? ""}</span>
                    </Link>
                  ))}
                </div>
              )}
            </section>
          )}
        </div>
      </aside>
    </div>
  );
}
