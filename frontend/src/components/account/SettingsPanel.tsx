import { useEffect, useState } from "react";

import {
  clearPersonalApiKey,
  fetchFavorites,
  savePersonalApiKey,
} from "../../helpers/accountApi";
import { getApiErrorMessage } from "../../helpers/http";
import type { BootstrapPayload, FavoriteListPayload } from "../../types/app";


type SettingsTab = "settings" | "favorites";


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

  return (
    <div className="settings-panel-overlay" onClick={onClose}>
      <aside className="settings-panel" onClick={(event) => event.stopPropagation()}>
        <div className="settings-panel-header">
          <h2>내 설정</h2>
          <button type="button" onClick={onClose}>
            닫기
          </button>
        </div>

        <div className="settings-tabs">
          <button
            type="button"
            className={activeTab === "settings" ? "active" : ""}
            onClick={() => setActiveTab("settings")}
          >
            설정
          </button>
          <button
            type="button"
            className={activeTab === "favorites" ? "active" : ""}
            onClick={() => setActiveTab("favorites")}
          >
            즐겨찾기
          </button>
        </div>

        {activeTab === "settings" ? (
          <div className="settings-section-stack">
            <section className="settings-section">
              <div className="settings-section-title-row">
                <h3>세션 API 키</h3>
                <span className={session.has_personal_api_key ? "status-chip active" : "status-chip"}>
                  {session.has_personal_api_key ? "등록됨" : "미등록"}
                </span>
              </div>
              <p className="settings-help">
                API 키는 현재 로그인 세션 동안에만 서버에 저장됩니다.
              </p>
              <div className="settings-field-row">
                <input
                  type="password"
                  placeholder="sk-..."
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

            {statusMessage ? <div className="settings-status">{statusMessage}</div> : null}
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
                  <a key={paper.arxiv_id} className="favorite-row" href={`/papers/${paper.arxiv_id}/`}>
                    <strong>{paper.title}</strong>
                    <span>{paper.published_at?.slice(0, 10) ?? ""}</span>
                  </a>
                ))}
              </div>
            )}
          </section>
        )}
      </aside>
    </div>
  );
}
