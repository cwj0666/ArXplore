import { useEffect, useRef, useState } from "react";
import { Link, useLocation } from "react-router-dom";

import { buildLoginPath } from "../../helpers/loginPath";
import type { BootstrapPayload } from "../../types/app";


export type SettingsTab = "settings" | "favorites";


interface AccountMenuProps {
  session: BootstrapPayload;
  onOpenSettings: (tab?: SettingsTab) => void;
  onLogout: () => void;
  className?: string;
}


export function AccountMenu({ session, onOpenSettings, onLogout, className }: AccountMenuProps) {
  const [menuOpen, setMenuOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement | null>(null);
  const triggerRef = useRef<HTMLButtonElement | null>(null);
  const location = useLocation();

  useEffect(() => {
    setMenuOpen(false);
  }, [location.pathname, location.search]);

  useEffect(() => {
    if (!menuOpen) {
      return;
    }

    const handlePointerDown = (event: MouseEvent) => {
      if (!rootRef.current?.contains(event.target as Node)) {
        setMenuOpen(false);
      }
    };
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setMenuOpen(false);
        triggerRef.current?.focus();
      }
    };

    document.addEventListener("mousedown", handlePointerDown);
    document.addEventListener("keydown", handleKeyDown);
    return () => {
      document.removeEventListener("mousedown", handlePointerDown);
      document.removeEventListener("keydown", handleKeyDown);
    };
  }, [menuOpen]);

  const rootClassName = className ? `account-menu-root ${className}` : "account-menu-root";

  if (!session.is_authenticated) {
    return (
      <div className={rootClassName}>
        <Link className="app-header-login" to={buildLoginPath(`${location.pathname}${location.search}`)}>
          로그인
        </Link>
      </div>
    );
  }

  const initial = session.username ? session.username.slice(0, 1).toUpperCase() : "?";
  const choose = (action: () => void) => () => {
    setMenuOpen(false);
    action();
  };

  return (
    <div className={rootClassName} ref={rootRef}>
      <button
        ref={triggerRef}
        type="button"
        className="account-trigger"
        aria-haspopup="menu"
        aria-expanded={menuOpen}
        onClick={() => setMenuOpen((value) => !value)}
      >
        <span className="account-trigger-badge" aria-hidden="true">{initial}</span>
        <span>{session.username}</span>
      </button>

      {menuOpen ? (
        <div className="account-menu" role="menu">
          <button type="button" role="menuitem" onClick={choose(() => onOpenSettings("settings"))}>
            내 설정
          </button>
          <button type="button" role="menuitem" onClick={choose(() => onOpenSettings("favorites"))}>
            즐겨찾기
          </button>
          <button type="button" role="menuitem" onClick={choose(onLogout)}>
            로그아웃
          </button>
        </div>
      ) : null}
    </div>
  );
}
