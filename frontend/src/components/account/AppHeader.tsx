import { useEffect, useRef, useState } from "react";
import { useLocation } from "react-router-dom";

import type { BootstrapPayload } from "../../types/app";


interface AppHeaderProps {
  session: BootstrapPayload;
  onOpenSettings: (tab?: "settings" | "favorites") => void;
  onLogout: () => void;
}


export function AppHeader({ session, onOpenSettings, onLogout }: AppHeaderProps) {
  const [menuOpen, setMenuOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement | null>(null);
  const location = useLocation();
  const isLoginPage = location.pathname === "/login/";

  useEffect(() => {
    setMenuOpen(false);
  }, [location.pathname, location.search]);

  useEffect(() => {
    if (!menuOpen) {
      return;
    }

    const handleClick = (event: MouseEvent) => {
      if (!menuRef.current?.contains(event.target as Node)) {
        setMenuOpen(false);
      }
    };

    document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, [menuOpen]);

  if (location.pathname === "/" || location.pathname.startsWith("/papers/") || location.pathname === "/login/") {
    return null;
  }

  const initial = session.username ? session.username.slice(0, 1).toUpperCase() : "?";

  return (
    <header className="app-header">
      <div className="app-header-inner">
        <div className="app-header-spacer" aria-hidden="true" />
        <a href="/" className="app-header-logo">
          ArXplore
        </a>

        <div className="app-header-actions" ref={menuRef}>
          {isLoginPage ? null : !session.is_authenticated ? (
            <a className="app-header-login" href={`/login/?next=${encodeURIComponent(`${location.pathname}${location.search}`)}`}>
              로그인
            </a>
          ) : (
            <>
              <button type="button" className="account-trigger" onClick={() => setMenuOpen((value) => !value)}>
                <span className="account-trigger-badge">{initial}</span>
                <span>{session.username}</span>
              </button>

              {menuOpen ? (
                <div className="account-menu">
                  <button type="button" onClick={() => onOpenSettings("settings")}>
                    내 설정
                  </button>
                  <button type="button" onClick={() => onOpenSettings("favorites")}>
                    즐겨찾기
                  </button>
                  <button type="button" onClick={onLogout}>
                    로그아웃
                  </button>
                </div>
              ) : null}
            </>
          )}
        </div>
      </div>
    </header>
  );
}
