---
title: IP당 rate limit, DEBUG 기본 off, 플레이스홀더 시크릿 거부
date: 2026-09-25
area: [backend]
decision: LLM 엔드포인트는 IP당 분당 30회를 공유하고 detail.json은 60회로 제한하며, DEBUG는 명시적으로 true일 때만 켜고, change-me로 시작하는 DJANGO_SECRET_KEY로는 시작하지 않는다
---

**결정과 근거** — 공개 접근을 전제로 운영 기본값을 바꿨다(dd3b013, 515e211). `backend/papers/ratelimit.py`가 Django 캐시로
클라이언트 IP(`RATE_LIMIT_IP_HEADER`, 기본 `X-Real-IP`)당 60초 고정 윈도를 센다. LLM을 부르는 4개 엔드포인트가
`RATE_LIMIT_LLM_PER_MINUTE`(기본 30)을 공유하고 `detail.json`은 `RATE_LIMIT_DETAIL_PER_MINUTE`(기본 60)을 쓴다. 넘으면 429와
`Retry-After`를 돌려준다. `DJANGO_DEBUG`는 문자열 "true"일 때만 켜지고, 테스트 설정 밖에서 `change-me`로 시작하는
`DJANGO_SECRET_KEY`는 거부한다. 당시에는 로그인 사용자가 입력한 OpenAI 키를 세션에 Fernet(시크릿에서 HKDF로 유도한 키)으로
암호화해 저장했는데, 이 부분은 phase-4에서 로그인과 함께 없앴다.

**트레이드오프** — 고정 윈도는 윈도 경계에서 최대 두 배까지 허용하지만 구현과 테스트가 단순하다. `REDIS_URL`이 없으면
프로세스별 LocMem 캐시라 gunicorn 워커 4개가 따로 세어 실제 허용량이 워커 수만큼 늘어난다. 로그인을 없앤 뒤에는 모든
방문자의 LLM 호출이 서버 키 비용이 되는데, IP당 한도 외에 전체 사용량 상한은 없다.

**eval 영향** — `tests/unit/test_security.py`(23개): `test_llm_limit_is_shared_between_llm_endpoints`,
`test_429_payload_has_error_and_retry_after`, `test_get_is_rejected_before_counting`, `test_debug_only_when_explicitly_true`,
`test_production_settings_refuse_change_me_secret`, `test_redis_url_selects_redis_cache`.

**알려진 한계** — IP 기준이라 NAT 뒤 여러 사용자가 한도를 나눠 쓰고, 헤더를 믿는 구성이므로 앞단 프록시가 헤더를 덮어써야
한다(nginx와 dev 프록시는 덮어쓴다). nginx는 HTTP만 제공하므로 외부 공개 전에 TLS와 `DJANGO_SECURE_COOKIES=true`가 필요하다.

**트러블슈팅** — 해당 없음
