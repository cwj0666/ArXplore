import { useState } from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";

import { postLogin, postSignup } from "../../helpers/accountApi";
import { getApiErrorMessage, getPasswordErrors } from "../../helpers/http";
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
  const [passwordErrors, setPasswordErrors] = useState<string[]>([]);
  const [successMessage, setSuccessMessage] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);

  const next = sanitizeNextPath(new URLSearchParams(location.search).get("next"));

  const handleSubmit = async () => {
    setErrorMessage("");
    setPasswordErrors([]);
    setSuccessMessage("");
    setIsSubmitting(true);
    try {
      if (mode === "signup") {
        try {
          await postSignup(username, password);
        } catch (error) {
          const passwordProblems = getPasswordErrors(error);
          setPasswordErrors(passwordProblems);
          setErrorMessage(
            passwordProblems.length > 0
              ? "비밀번호가 아래 조건을 만족하지 않습니다."
              : getApiErrorMessage(error, REQUEST_FAILED_MESSAGE),
          );
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
        <Link to={next} className="back-btn">
          뒤로가기
        </Link>
      </div>
      <div className="login-topbar-center">
        <Link to="/" className="login-topbar-logo">
          ArXplore
        </Link>
      </div>
      <div className="login-topbar-right" />
    </div>
    <main className="login-page">
      <div className="login-card">
        <div className="login-card-header">
          <h1>{mode === "login" ? "로그인" : "회원가입"}</h1>
        </div>

        <div className="login-mode-tabs" role="group" aria-label="로그인 또는 회원가입">
          <button
            type="button"
            aria-pressed={mode === "login"}
            className={mode === "login" ? "active" : ""}
            onClick={() => setMode("login")}
          >
            로그인
          </button>
          <button
            type="button"
            aria-pressed={mode === "signup"}
            className={mode === "signup" ? "active" : ""}
            onClick={() => setMode("signup")}
          >
            회원가입
          </button>
        </div>

        <label className="login-field">
          <span>사용자 이름</span>
          <input
            value={username}
            autoComplete="username"
            onChange={(event) => setUsername(event.target.value)}
          />
        </label>

        <label className="login-field">
          <span>비밀번호</span>
          <input
            type="password"
            value={password}
            autoComplete={mode === "login" ? "current-password" : "new-password"}
            onChange={(event) => setPassword(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter") {
                void handleSubmit();
              }
            }}
          />
        </label>

        {errorMessage ? (
          <div className="login-feedback login-feedback-error" role="alert">
            {errorMessage}
            {passwordErrors.length > 0 ? (
              <ul className="login-feedback-list">
                {passwordErrors.map((item) => (
                  <li key={item}>{item}</li>
                ))}
              </ul>
            ) : null}
          </div>
        ) : null}
        {successMessage ? <div className="login-feedback login-feedback-success" role="status">{successMessage}</div> : null}

        <button type="button" className="login-submit" onClick={() => void handleSubmit()} disabled={isSubmitting}>
          {isSubmitting ? "처리 중..." : mode === "login" ? "로그인" : "회원가입 후 로그인"}
        </button>
      </div>
    </main>
    </>
  );
}
