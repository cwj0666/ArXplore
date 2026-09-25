import { useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";

import { postLogin, postSignup } from "../../helpers/accountApi";
import { getApiErrorMessage } from "../../helpers/http";
import { sanitizeNextPath } from "../../helpers/safeRedirect";
import "./login-page.css";


const REQUEST_FAILED_MESSAGE = "요청을 처리하지 못했습니다. 잠시 후 다시 시도해 주세요.";


interface LoginPageProps {
  onAuthSuccess: () => Promise<void>;
}


export function LoginPage({ onAuthSuccess }: LoginPageProps) {
  const navigate = useNavigate();
  const location = useLocation();
  const [mode, setMode] = useState<"login" | "signup">("login");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [errorMessage, setErrorMessage] = useState("");
  const [successMessage, setSuccessMessage] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);

  const next = sanitizeNextPath(new URLSearchParams(location.search).get("next"));

  const handleSubmit = async () => {
    setErrorMessage("");
    setSuccessMessage("");
    setIsSubmitting(true);
    try {
      if (mode === "signup") {
        try {
          await postSignup(username, password);
        } catch (error) {
          setErrorMessage(getApiErrorMessage(error, REQUEST_FAILED_MESSAGE));
          return;
        }

        setSuccessMessage("회원가입이 완료되었습니다. 로그인 중입니다...");
        await new Promise((resolve) => window.setTimeout(resolve, 1200));
      }

      try {
        await postLogin(username, password);
      } catch (error) {
        const message = getApiErrorMessage(error, REQUEST_FAILED_MESSAGE);
        setSuccessMessage("");
        setErrorMessage(
          mode === "signup"
            ? `회원가입은 완료되었지만 자동 로그인에 실패했습니다. ${message}`
            : message,
        );
        return;
      }

      await onAuthSuccess();
      navigate(next, { replace: true });
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <>
    <div className="login-topbar">
      <div className="login-topbar-left">
        <a href={next} className="back-btn">
          뒤로가기
        </a>
      </div>
      <div className="login-topbar-center">
        <a href="/" className="login-topbar-logo">
          ArXplore
        </a>
      </div>
      <div className="login-topbar-right" />
    </div>
    <main className="login-page">
      <div className="login-card">
        <div className="login-card-header">
          <h1>{mode === "login" ? "로그인" : "회원가입"}</h1>
        </div>

        <div className="login-mode-tabs">
          <button
            type="button"
            className={mode === "login" ? "active" : ""}
            onClick={() => setMode("login")}
          >
            로그인
          </button>
          <button
            type="button"
            className={mode === "signup" ? "active" : ""}
            onClick={() => setMode("signup")}
          >
            회원가입
          </button>
        </div>

        <label className="login-field">
          <span>사용자 이름</span>
          <input value={username} onChange={(event) => setUsername(event.target.value)} />
        </label>

        <label className="login-field">
          <span>비밀번호</span>
          <input
            type="password"
            value={password}
            onChange={(event) => setPassword(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter") {
                void handleSubmit();
              }
            }}
          />
        </label>

        {errorMessage ? <div className="login-feedback login-feedback-error">{errorMessage}</div> : null}
        {successMessage ? <div className="login-feedback login-feedback-success">{successMessage}</div> : null}

        <button type="button" className="login-submit" onClick={() => void handleSubmit()} disabled={isSubmitting}>
          {isSubmitting ? "처리 중..." : mode === "login" ? "로그인" : "회원가입 후 로그인"}
        </button>
      </div>
    </main>
    </>
  );
}
