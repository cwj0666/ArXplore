---
title: 커밋된 Tailscale 키 폐기와 서버 compose 노출 축소
date: 2026-09-25
area: [infra]
decision: 문서에 커밋된 Tailscale 인증 키를 폐기하고 키·서버 IP를 플레이스홀더로 바꾸며, Airflow 전원 관리자 모드와 전 인터페이스 포트 공개를 사용자 계정·Fernet 필수·Tailscale IP 바인딩으로 바꾼다
---

**결정과 근거** — 팀 시절 커밋 c7b2c34가 `docs/management/TEAM_SETUP.md`에 Tailscale 인증 키와 서버 IP를 평문으로 담고
있었다. 키는 폐기(revoke)했고 문서의 값은 플레이스홀더로 바꿨다(dc0981d). 서버 compose는 Airflow를
`SIMPLE_AUTH_MANAGER_ALL_ADMINS`(모든 접속자가 관리자)로 띄우고 포트를 모든 인터페이스에 공개했다. 관리자 사용자
(`AIRFLOW_ADMIN_USER`)와 `AIRFLOW_FERNET_KEY`를 필수로 하고(비어 있으면 compose가 실행을 거부), 포트를
`TAILSCALE_SERVER_IP`에만 바인딩했다. 리뷰에서 Airflow API 시크릿 필수화(f229c59)와 레이아웃 파서의 loopback 바인딩
(515e211)을 더했다.

**트레이드오프** — git 히스토리를 다시 쓰는 대안은 팀 저장소의 모든 클론과 fork를 깨뜨리므로 택하지 않았다. 키 폐기로
위험을 없애고 히스토리는 그대로 두었다. 대신 CI 시크릿 스캔이 과거 커밋의 폐기된 키를 계속 찾으므로 예외 처리가 필요하다.
Tailscale IP 바인딩은 Tailscale 없이 서버에 붙는 경로를 없앤다.

**eval 영향** — CI security 잡이 `gitleaks git`으로 전체 히스토리를 스캔한다. 기본 규칙에는 Tailscale 키 규칙이 없어
`.gitleaks.toml`에 사용자 규칙 `tailscale-auth-key`를 더했고, 폐기된 키는 `.gitleaksignore`의 fingerprint와 키 ID 허용
목록으로만 예외 처리한다. infra 잡이 더미 `.env`로 두 compose 파일의 `docker compose config`를 검사한다.

**알려진 한계** — 폐기된 키 문자열은 히스토리에 남아 있다. 허용 목록은 그 키 하나만 예외로 두므로, 같은 형식의 새 키가
들어오면 CI가 실패한다.

**트러블슈팅** — 저장소 문서에 실제 Tailscale 인증 키가 커밋되어 있음

- 증상: `docs/management/TEAM_SETUP.md`에 Tailscale 인증 키와 서버 IP가 평문으로 들어 있었다.
- 원인: 팀 설정 절차 문서에 실제 값을 붙여 넣은 채 커밋했고(c7b2c34), 시크릿 스캔 장치가 없었다.
- 확인 방법: 감사 중 문서에서 발견했고, gitleaks 기본 규칙으로는 이 형식이 잡히지 않는 것을 확인해 사용자 규칙을 만들었다.
- 해결: 키를 폐기하고 문서 값을 플레이스홀더로 바꿨다(dc0981d). 사용자 규칙과 폐기 키 예외를 두어 security 잡이 녹색이다.
- 재발 방지: CI security 잡(전체 히스토리 스캔)과 pre-commit gitleaks 훅이 `tskey-` 형식을 잡는다.
