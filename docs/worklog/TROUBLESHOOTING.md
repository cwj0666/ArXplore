# 트러블슈팅 모음

`python scripts/worklog.py index`가 생성한다. 손으로 고치지 않는다.
각 항목의 **트러블슈팅** 절을 그대로 옮긴다. 고칠 때는 항목 파일을 고치고 `index`를 다시 실행한다.

항목 17개, 사례 20건.

| phase | 날짜 | 제목 | 영역 |
|---|---|---|---|
| 4 | 2026-09-25 | [행동 지표 보정, 거절 문구 목록 확대와 인용 제목 인정](phase-4/2026-09-25_09_행동-지표-보정,-거절-문구-목록-확대와-인용-제목-인정.md) | eval |
| 4 | 2026-09-25 | [에이전트 답변 범위를 코퍼스로 제한하고 메시지 속 지시를 따르지 않게](phase-4/2026-09-25_08_에이전트-답변-범위를-코퍼스로-제한하고-메시지-속-지시를-따르지-않게.md) | agent |
| 4 | 2026-09-25 | [lexical 두 단계 점수와 논문당 후보 상한](phase-4/2026-09-25_06_lexical-두-단계-점수와-논문당-후보-상한.md) | retrieval |
| 4 | 2026-09-25 | [로컬 코퍼스 구축과 입력 문자 정규화](phase-4/2026-09-25_05_로컬-코퍼스-구축과-입력-문자-정규화.md) | eval, pipeline |
| 3 | 2026-09-25 | [스트림 종료 판정과 인용 링크 호스트 검증](phase-3/2026-09-25_06_스트림-종료-판정과-인용-링크-호스트-검증.md) | frontend, agent |
| 3 | 2026-09-25 | [재처리 멱등성 보강, 해시는 청크 저장 뒤에 기록하고 부분 실패 잡은 재시도](phase-3/2026-09-25_05_재처리-멱등성-보강,-해시는-청크-저장-뒤에-기록하고-부분-실패-잡은-재시도.md) | pipeline |
| 3 | 2026-09-25 | [인프라와 프론트엔드 정리, nginx 스트림 경로 수정](phase-3/2026-09-25_03_인프라와-프론트엔드-정리,-nginx-스트림-경로-수정.md) | infra, frontend |
| 2 | 2026-09-25 | [에이전트 가드레일, 인용 사후 검증, 도구 호출 턴 스트림 버퍼](phase-2/2026-09-25_03_에이전트-가드레일,-인용-사후-검증,-도구-호출-턴-스트림-버퍼.md) | agent |
| 0 | 2026-09-25 | [프론트엔드 입력·리다이렉트·스트림 오류 처리](phase-0/2026-09-25_09_프론트엔드-입력·리다이렉트·스트림-오류-처리.md) | frontend |
| 0 | 2026-09-25 | [커밋된 Tailscale 키 폐기와 서버 compose 노출 축소](phase-0/2026-09-25_08_커밋된-Tailscale-키-폐기와-서버-compose-노출-축소.md) | infra |
| 0 | 2026-09-25 | [논문 단위 실패 격리와 보강 필드 보존, 임베딩 backlog 독립 실행](phase-0/2026-09-25_07_논문-단위-실패-격리와-보강-필드-보존,-임베딩-backlog-독립-실행.md) | pipeline |
| 0 | 2026-09-25 | [리포지토리 생성자에서 DDL을 빼고 스키마 생성을 명시 호출로](phase-0/2026-09-25_06_리포지토리-생성자에서-DDL을-빼고-스키마-생성을-명시-호출로.md) | pipeline, backend |
| 0 | 2026-09-25 | [SSE 요청 검증을 스트림 시작 전에 하고 분석 엔드포인트를 POST로](phase-0/2026-09-25_05_SSE-요청-검증을-스트림-시작-전에-하고-분석-엔드포인트를-POST로.md) | backend |
| 0 | 2026-09-25 | [에이전트 검색 도구의 필드 계약과 설명 복구](phase-0/2026-09-25_04_에이전트-검색-도구의-필드-계약과-설명-복구.md) | agent |
| 0 | 2026-09-25 | [참고문헌 섹션 판정을 제목 전체 일치 규칙 하나로](phase-0/2026-09-25_03_참고문헌-섹션-판정을-제목-전체-일치-규칙-하나로.md) | pipeline, retrieval |
| 0 | 2026-09-25 | [초록 폴백이 기존 본문과 임베딩을 지우지 않게 한다](phase-0/2026-09-25_02_초록-폴백이-기존-본문과-임베딩을-지우지-않게-한다.md) | pipeline |
| 0 | 2026-09-25 | [PDF 파서 믹스인 참조 복구](phase-0/2026-09-25_01_PDF-파서-믹스인-참조-복구.md) | pipeline |

## Phase 4

### [행동 지표 보정, 거절 문구 목록 확대와 인용 제목 인정](phase-4/2026-09-25_09_행동-지표-보정,-거절-문구-목록-확대와-인용-제목-인정.md)

phase 4 · 2026-09-25 · eval

**트러블슈팅** — 올바른 거절이 거절로 판정되지 않음

- 증상: run1 에이전트 `refusal_correct`가 초기 문구 목록 기준 0.400(15건)이었는데, 답변을 읽어 보면 거절한 답변이 더 많았다.
- 원인: 초기 거절 문구 목록이 상세 챗 프롬프트의 문구만 담아, 에이전트가 쓰는 "죄송하지만", "할 수 없습니다", "unable to"
  같은 표현을 잡지 못했다.
- 확인 방법: refuse 케이스 중 refused로 분류되지 않은 레코드의 답변을 읽어 실제 거절 표현을 모았다.
- 해결: 표현을 기본 목록에 더했다(ebf4d90). 같은 run1 답변을 넓힌 목록으로 다시 분류하면 0.733이다. run2의 0.933은
  여기에 에이전트 범위 규칙(에이전트 범위 항목)을 더한 뒤의 값이다.
- 재발 방지: 거절 문구 목록을 파일로 바꿔 끼울 수 있고(`--refusal-phrases`), 답변 파일은 그대로 두고 다시 채점할 수 있다.
  `eval/README.md`에 문자열 포함 검사의 한계를 적었다.

### [에이전트 답변 범위를 코퍼스로 제한하고 메시지 속 지시를 따르지 않게](phase-4/2026-09-25_08_에이전트-답변-범위를-코퍼스로-제한하고-메시지-속-지시를-따르지-않게.md)

phase 4 · 2026-09-25 · agent

**트러블슈팅** — 코퍼스 밖 질문에 일반 지식으로 답하고 기억 속 arXiv 링크를 씀

- 증상: run1에서 에이전트 `refusal_correct`가 0.400(15건), `no_fabricated_links`가 0.993(134건 중 1건 실패),
  `must_not_contain_ok`가 0.929였다. 실패한 1건은 검색 결과에 없는 arXiv 1706.03762 링크를 답변에 넣었다.
- 원인: 시스템 프롬프트가 "도구 결과만 인용한다"만 요구하고 답변 범위를 정하지 않아, 모델이 일반 지식과 기억 속 논문으로
  답했다.
- 확인 방법: `answers_run1.jsonl`에서 실패 레코드의 답변을 읽어 링크와 검색 hit을 대조했다.
- 해결: 범위 규칙과 기억 속 링크 금지(ebf4d90), 메시지 속 지시 무시 규칙(94a0a0d)을 더했다. run2에서 `refusal_correct` 0.933,
  `no_fabricated_links` 1.000, `must_not_contain_ok` 1.000이다.
- 재발 방지: 코퍼스 밖·안전 관점 케이스가 생성 평가에 상주해 행동 지표로 다시 잰다.
  `test_agent_guardrails.py`의 `test_system_prompt_contains_guardrail_rules`는 phase-2 규칙(데이터-지시 분리, 날조 금지,
  도구 결과만 인용, 도구 턴 무텍스트)만 확인하고 규칙 4~6 문구는 단위 테스트로 고정하지 않았다.

### [lexical 두 단계 점수와 논문당 후보 상한](phase-4/2026-09-25_06_lexical-두-단계-점수와-논문당-후보-상한.md)

phase 4 · 2026-09-25 · retrieval

**트러블슈팅** — 긴 서술형 질의에서 lexical 결과가 0행

- 증상: lexical 평가에서 hit@1·hit@5·hit@10이 모두 0.211이고 한국어 질의는 hit@10 0.000이었다. 긴 서술형 질의를 lexical
  검색에 넣으면 결과가 비었다.
- 원인: `websearch_to_tsquery`는 모든 lexeme을 AND로 요구한다. 질의가 길어질수록 모든 단어를 가진 청크가 없어진다.
- 확인 방법: hit@1과 hit@10이 같다는 것은 맞힌 질의는 1위로 맞히고 나머지는 상위 10위 안에 정답이 없다는 뜻이다. 놓친 긴
  질의를 lexical에 직접 넣어 0행을 확인했다.
- 해결: strict 일치와 부분 일치를 나눈 두 단계 점수로 바꿨다(5f9c4cd). 통합 테스트 `test_long_query_returns_partial_matches`,
  `test_strict_match_ranks_above_partial_match`가 통과하고, lexical hit@10이 0.211에서 0.643으로 올랐다.
- 재발 방지: 긴 질의·strict 우선 순위를 통합 테스트가 실제 PostgreSQL에서 고정하고, 질의 형태 관점 케이스(query_form 12건)가
  평가에 상주한다.

사례 2 — 상위 10개가 한 논문의 청크로 채워짐

- 증상: 전체 검색의 상위 10개가 한 논문의 청크로만 채워지는 질의가 있었다.
- 원인: 논문당 2청크 다양성 필터는 후보 풀 안에서만 작동하는데, 후보 풀 `max(limit×3, 10)`개가 모두 1위 논문의 청크여서
  다른 논문으로 채울 후보가 없었다.
- 확인 방법: 평가 질의별 결과에서 상위 hit의 논문 id가 하나로 몰린 질의를 찾고, 그 질의의 후보 풀 구성을 확인했다.
- 해결: SQL에서 논문당 3청크로 자른 뒤 limit을 적용하고 후보 풀을 넓혔다(5f9c4cd).
  `test_per_paper_cap_spreads_top_k_across_papers`, `test_vector_per_paper_cap_spreads_top_k_across_papers`가 통과한다.
  vector hit@10은 0.800에서 0.826이 됐다.
- 재발 방지: `*_nodiv` ablation이 다양성 규칙의 효과를 계속 잰다. 최종 실행의 hit@10은 lexical 0.643 / nodiv 0.640, vector
  0.826 / 0.809, hybrid 0.817 / 0.807이다.

### [로컬 코퍼스 구축과 입력 문자 정규화](phase-4/2026-09-25_05_로컬-코퍼스-구축과-입력-문자-정규화.md)

phase 4 · 2026-09-25 · eval, pipeline

**트러블슈팅** — 서로게이트 문자로 본문 해시 계산 실패

- 증상: 코퍼스 구축 중 4개 날짜의 prepare가 `compute_fulltext_content_hash`에서 `UnicodeEncodeError`로 실패했다.
- 원인: pypdf가 추출한 텍스트에 짝이 없는 서로게이트 코드포인트(U+D800~U+DFFF)가 섞여 있었고, 해시를 위해 UTF-8로
  인코딩하는 순간 실패했다.
- 확인 방법: 실패한 날짜들의 예외 위치가 모두 해시 계산의 인코딩이었고, 해당 본문에서 서로게이트 문자를 확인했다.
- 해결: 파싱 직후 본문 텍스트와 섹션 제목·본문에서 서로게이트를 지운다(`strip_surrogates`, 24c067f).
  `test_surrogate_characters_are_stripped_before_hashing_and_saving`이 통과하고, 재처리 뒤 275편 모두 청크를 가진다(결과
  파일의 "청크 보유 275").
- 재발 방지: 해시와 저장 전에 같은 정규화를 거치는 것을 단위 테스트가 확인한다. raw 저장 쪽 NUL은
  `test_nul_characters_are_stored_without_error`가 따로 지킨다.

사례 2 — NUL 문자가 든 질의로 검색이 ValueError

- 증상: 제어 문자가 섞인 안전 케이스(`sf-control-chars-mixed`)가 검색에서
  `ValueError: A string literal cannot contain NUL (0x00) characters.`로 실패했다.
- 원인: psycopg2는 NUL이 든 문자열 파라미터를 거부하는데, API와 retriever 진입점이 제어 문자를 지우지 않았다.
- 확인 방법: 검색 평가 질의별 CSV의 `error` 열. 24c067f 시점 실행(`20260925-222321.csv`)의 lexical 1건, 5f9c4cd 시점
  실행(`20260925-225658.csv`)의 ablation 3개가 같은 케이스·같은 오류였다.
- 해결: API 입력에서 줄바꿈·탭을 뺀 제어 문자를 지우고(`strip_control_characters`, 24c067f), retriever 진입점의
  `normalize_search_query`가 제어 문자를 지우고 빈 질의면 DB를 부르지 않게 했으며(5f9c4cd), 평가 ablation 경로도 같은
  정규화를 거치게 했다(0217493). 위 테스트들이 통과하고, 제품 경로(lexical·vector·hybrid)는 최종 실행에서 오류 0건이다.
- 재발 방지: 제어 문자 케이스가 평가 카탈로그 safety 관점에 상주하고, 오류는 결과 표의 `errors` 열에 드러난다.

## Phase 3

### [스트림 종료 판정과 인용 링크 호스트 검증](phase-3/2026-09-25_06_스트림-종료-판정과-인용-링크-호스트-검증.md)

phase 3 · 2026-09-25 · frontend, agent

**트러블슈팅** — arXiv 유사 도메인 링크가 인용으로 인정됨

- 증상: 답변 속 링크가 `arxiv.org`가 아닌 호스트를 가리켜도 경로에 `arxiv.org/abs/<id>` 모양이 있으면 해당 arXiv hit의 인용으로
  `in_answer` 표시를 받을 수 있었다.
- 원인: `_ARXIV_URL_PATTERN`이 URL 문자열 어디에서든 `arxiv\.org/(abs|pdf)/`를 검색했고 호스트를 확인하지 않았다.
- 확인 방법: 3차 리뷰에서 지적됐고, 다른 호스트에 같은 경로를 붙인 링크로 재현하는 테스트를 만들었다
  (`test_lookalike_domain_is_not_attributed_to_an_arxiv_hit`).
- 해결: `urlsplit`으로 scheme과 hostname을 확인한 뒤 경로에서만 ID를 읽는다(4c358d5).
  `test_arxiv_id_from_url_requires_real_arxiv_host`의 매개변수 사례들이 통과한다.
- 재발 방지: 평가의 `no_fabricated_links`가 제품과 같은 추출 함수를 쓰는지 `test_link_extraction_uses_product_rules`가 확인한다.

### [재처리 멱등성 보강, 해시는 청크 저장 뒤에 기록하고 부분 실패 잡은 재시도](phase-3/2026-09-25_05_재처리-멱등성-보강,-해시는-청크-저장-뒤에-기록하고-부분-실패-잡은-재시도.md)

phase 3 · 2026-09-25 · pipeline

**트러블슈팅** — 청크 저장이 실패한 논문이 재시도에서 unchanged로 건너뛰어짐

- 증상: 본문 저장 뒤 청크 저장에서 실패한 논문을 재시도하면, 본문 해시가 같다는 이유로 교체를 건너뛰어 청크가 없는 상태로
  남을 수 있었다.
- 원인: `content_hash`를 본문 저장 시점, 즉 청크 저장 전에 기록했고, unchanged 판정이 저장된 청크 유무를 보지 않았다.
- 확인 방법: 2차 리뷰에서 지적됐고, 청크 저장에서 예외를 내는 저장소로 재현했다(단위
  `test_chunk_write_failure_does_not_mark_content_unchanged_on_retry`, 통합
  `test_chunk_write_failure_leaves_hash_unset_so_retry_rewrites_chunks`).
- 해결: 해시를 청크 저장 뒤 `update_paper_fulltext_content_hash`로 기록하고 unchanged 판정에 저장된 청크 수를 더했다(515e211).
  3차 리뷰 지적으로 낮은 순위 보호도 같은 조건을 쓰게 했다(4c358d5). 위 두 테스트와
  `test_lower_ranked_source_is_saved_when_existing_fulltext_has_no_chunks`가 통과한다.
- 재발 방지: 통합 테스트가 실제 PostgreSQL에서 청크 쓰기 실패 뒤 해시가 비어 있는지 확인한다.

사례 2 — 일부 논문이 실패한 날짜 잡이 완료로 닫힘

- 증상: 날짜 잡에서 일부 논문이 실패해도 잡이 `done`으로 끝나, 실패한 논문이 다시 처리되지 않았다.
- 원인: phase-0에서 논문 단위로 실패를 격리한 뒤, 잡 결과가 부분 실패여도 큐 소비자가 `complete_prepare_job`을 불렀다.
- 확인 방법: 1차 리뷰에서 지적됐고, 한 논문만 실패하는 가짜 파서로 큐 소비 결과를 확인했다
  (`test_consume_queue_retries_job_on_partial_failure`).
- 해결: 실패가 하나라도 있으면 `fail_prepare_job`으로 backoff 재시도하고, 완료는 실패가 없을 때만 한다(f229c59).
  `test_consume_queue_completes_job_only_without_paper_failures`, `test_papers_prepared_in_a_partially_failed_job_are_embedded`가
  통과한다.
- 재발 방지: 부분 실패·전부 실패·날짜 수준 예외·선점 상실 경우를 `test_prepare_pipeline.py`의 `test_consume_queue_*` 테스트가
  각각 고정한다.

### [인프라와 프론트엔드 정리, nginx 스트림 경로 수정](phase-3/2026-09-25_03_인프라와-프론트엔드-정리,-nginx-스트림-경로-수정.md)

phase 3 · 2026-09-25 · infra, frontend

**트러블슈팅** — 상세 챗 스트림 요청이 index.html을 받음

- 증상: nginx를 거치면 상세 챗 스트림 요청 `/papers/<id>/chat/stream/`이 Django 응답 대신 React `index.html`을 받았다.
- 원인: nginx에 이 경로를 프록시하는 location이 없어 마지막 SPA 폴백(`try_files $uri $uri/ /index.html`)에 걸렸다. phase-2에서
  상세 챗 스트림 경로를 더할 때 nginx 설정을 같이 고치지 않았다.
- 확인 방법: `docker/nginx/nginx.conf`의 location 목록과 Django URL 목록을 대조해 스트림 경로가 빠진 것을 확인했다.
- 해결: 두 SSE 경로를 정규식 location `^/papers/(assistant|[^/]+/chat)/stream/$` 하나로 묶고 `proxy_buffering off`,
  `gzip off`, `proxy_read_timeout 300s`로 프록시했다(dd3b013).
- 재발 방지: CLAUDE.md에 "새 API 경로는 nginx location과 `frontend/vite.config.ts` 프록시에 같이 넣는다"를 적었다. 이 규칙을
  검사하는 자동 테스트는 없다.

## Phase 2

### [에이전트 가드레일, 인용 사후 검증, 도구 호출 턴 스트림 버퍼](phase-2/2026-09-25_03_에이전트-가드레일,-인용-사후-검증,-도구-호출-턴-스트림-버퍼.md)

phase 2 · 2026-09-25 · agent

**트러블슈팅** — 도구 호출 턴의 서두 텍스트가 답변 앞에 새어 나옴

- 증상: 에이전트가 도구를 부르는 턴에서 모델이 먼저 생성한 텍스트가 사용자 화면으로 스트리밍된 뒤, 도구 결과를 반영한
  답변이 이어졌다.
- 원인: `stream_mode=["messages", "updates"]`로 받은 토큰을 즉시 내보냈는데, 그 메시지에 도구 호출이 붙는지는 텍스트 뒤에
  `tool_call_chunks`가 와야 알 수 있다.
- 확인 방법: 1차 리뷰에서 지적됐고, 같은 메시지에 텍스트와 도구 호출이 차례로 오는 가짜 스트림으로 재현했다
  (`test_agent_drops_text_streamed_before_tool_calls_in_the_same_message`).
- 해결: 메시지 단위 120자 버퍼를 두고 도구 호출이 확인되면 버렸다(f229c59). 2차 리뷰에서 줄바꿈이 오면 버퍼를 비우던 규칙
  때문에 줄바꿈이 든 서두가 샌다는 지적을 받아 그 규칙을 없앴다(515e211).
  `test_agent_newline_preamble_before_tool_call_never_leaks`, `test_agent_newline_does_not_release_buffer_before_completion`,
  `test_agent_streams_long_answer_live_after_buffer_threshold`가 통과한다.
- 재발 방지: 짧은 서두·줄바꿈 서두·업데이트로만 알려진 도구 호출 경우를 각각 테스트로 고정했고, CLAUDE.md에 버퍼가 막지
  못하는 경우(120자를 넘는 서두)를 적었다.

## Phase 0

### [프론트엔드 입력·리다이렉트·스트림 오류 처리](phase-0/2026-09-25_09_프론트엔드-입력·리다이렉트·스트림-오류-처리.md)

phase 0 · 2026-09-25 · frontend

**트러블슈팅** — 로그인 뒤 이동 경로로 외부 사이트 이동

- 증상: 로그인 페이지의 `next` 파라미터로 외부 도메인으로 이동할 수 있었다. 첫 수정 뒤에도 `/a/..//evil.com` 입력이
  검사를 통과했다.
- 원인: 기준선은 `next`를 검사하지 않았다. 첫 수정(dc0981d)은 파싱한 URL의 origin만 비교했는데, `/a/..//evil.com`은 같은
  origin으로 파싱되면서 경로가 `//evil.com`으로 정규화되고, 이 경로로 이동하면 프로토콜 상대 URL이 되어 외부로 나간다.
- 확인 방법: 1차 리뷰(산타 방식)에서 이 입력이 지적됐고, 정규화된 pathname이 `//evil.com`이 되는 것을 확인했다.
- 해결: origin 비교에 더해 정규화된 pathname이 `//`로 시작하거나 백슬래시를 포함하는 경우를 퍼센트 디코딩 뒤에도
  거부했다(f229c59). 프론트엔드 테스트 러너가 없어 CI frontend 잡의 타입 검사·빌드와 리뷰 재확인으로 검증했다.
- 재발 방지: 로그인과 `next` 파라미터 자체를 phase-4에서 없앴다(e8b57f9). 현재 코드에는 이 리다이렉트 경로가 없다.

### [커밋된 Tailscale 키 폐기와 서버 compose 노출 축소](phase-0/2026-09-25_08_커밋된-Tailscale-키-폐기와-서버-compose-노출-축소.md)

phase 0 · 2026-09-25 · infra

**트러블슈팅** — 저장소 문서에 실제 Tailscale 인증 키가 커밋되어 있음

- 증상: `docs/management/TEAM_SETUP.md`에 Tailscale 인증 키와 서버 IP가 평문으로 들어 있었다.
- 원인: 팀 설정 절차 문서에 실제 값을 붙여 넣은 채 커밋했고(c7b2c34), 시크릿 스캔 장치가 없었다.
- 확인 방법: 감사 중 문서에서 발견했고, gitleaks 기본 규칙으로는 이 형식이 잡히지 않는 것을 확인해 사용자 규칙을 만들었다.
- 해결: 키를 폐기하고 문서 값을 플레이스홀더로 바꿨다(dc0981d). 사용자 규칙과 폐기 키 예외를 두어 security 잡이 녹색이다.
- 재발 방지: CI security 잡(전체 히스토리 스캔)과 pre-commit gitleaks 훅이 `tskey-` 형식을 잡는다.

### [논문 단위 실패 격리와 보강 필드 보존, 임베딩 backlog 독립 실행](phase-0/2026-09-25_07_논문-단위-실패-격리와-보강-필드-보존,-임베딩-backlog-독립-실행.md)

phase 0 · 2026-09-25 · pipeline

**트러블슈팅** — 논문 한 건의 예외로 날짜 잡 전체가 실패

- 증상: 한 날짜의 논문 중 하나만 파싱이나 저장에서 예외를 내도 그 날짜의 나머지 논문이 처리되지 않고 잡이 실패했다.
- 원인: `run_prepare_papers`의 논문 루프에 논문 단위 예외 처리가 없어 첫 예외가 날짜 잡까지 전파됐다.
- 확인 방법: 루프 구조를 읽어 확인했고, 한 논문에서만 예외를 내는 가짜 파서로 재현하는 테스트를 만들었다.
- 해결: 논문 단위로 예외를 잡아 결과 집계에 실패로 세고 다음 논문으로 넘어간다(dc0981d).
  `test_run_prepare_papers_isolates_per_paper_failures`가 통과한다. 실패가 섞인 잡의 재시도는 f229c59에서 더했다.
- 재발 방지: 전부 실패(`test_run_prepare_papers_all_failed`), 빈 날짜(`test_run_prepare_papers_empty_date_is_success`),
  일부 실패(`test_consume_queue_retries_job_on_partial_failure`)를 각각 테스트로 고정했다.

### [리포지토리 생성자에서 DDL을 빼고 스키마 생성을 명시 호출로](phase-0/2026-09-25_06_리포지토리-생성자에서-DDL을-빼고-스키마-생성을-명시-호출로.md)

phase 0 · 2026-09-25 · pipeline, backend

**트러블슈팅** — 에이전트 도구 호출마다 DDL과 DROP TABLE 실행

- 증상: 에이전트 도구를 부를 때마다 PostgreSQL에 DDL 약 15개가 실행되고 `paper_ai_summaries` 테이블이 삭제됐다.
- 원인: 스키마 생성 코드가 `PaperRepository.__init__`에 있었고, 도구가 호출마다 리포지토리를 새로 만들었다.
- 확인 방법: 기준선 `paper_repository.py`의 생성자에서 CREATE·ALTER·DROP 문을 세고, 도구 코드의 생성 위치를 확인했다.
- 해결: 생성자에서 DDL을 제거하고 `ensure_schema()`와 `scripts/migrate_schema.py`로 옮겼다(dc0981d).
  `test_constructor_runs_no_ddl`, `test_ensure_schema_is_explicit_and_never_drops_tables`가 통과한다.
- 재발 방지: `test_constructor_runs_no_ddl`은 `psycopg2.connect`를 호출 즉시 실패하게 바꾼 상태에서 생성자를 부른다.
  CLAUDE.md에 "요청 경로에서 DDL 금지"를 적었다.

### [SSE 요청 검증을 스트림 시작 전에 하고 분석 엔드포인트를 POST로](phase-0/2026-09-25_05_SSE-요청-검증을-스트림-시작-전에-하고-분석-엔드포인트를-POST로.md)

phase 0 · 2026-09-25 · backend

**트러블슈팅** — 잘못된 요청이 HTTP 200으로 응답됨

- 증상: 빈 메시지나 인증 없는 요청에도 SSE 엔드포인트가 HTTP 200을 돌려주고 본문에 `error` 이벤트만 실었다.
- 원인: 검증 코드가 `StreamingHttpResponse`에 넘긴 제너레이터 안에 있어, 첫 이벤트를 만들 때 이미 200 헤더가 확정됐다.
- 확인 방법: 뷰의 호출 순서를 따라가 검증이 제너레이터 안에 있는 것을 확인했다. `test_api_views.py`는 `RequestFactory`로
  빈 메시지를 보내 응답 상태 코드를 확인한다.
- 해결: 검증을 스트림 생성 전으로 옮겼다(dc0981d). `test_empty_message_returns_400`, `test_invalid_body_returns_400`이
  통과한다.
- 재발 방지: SSE 뷰마다 4xx·503 경로와 GET 거부를 `test_api_views.py`가 고정한다.

### [에이전트 검색 도구의 필드 계약과 설명 복구](phase-0/2026-09-25_04_에이전트-검색-도구의-필드-계약과-설명-복구.md)

phase 0 · 2026-09-25 · agent

**트러블슈팅** — 에이전트가 보는 모든 검색 결과가 "제목 없음"

- 증상: 에이전트 답변의 근거 목록에서 모든 논문 제목이 "제목 없음"으로 나왔다.
- 원인: retriever 결과 키는 `paper_title`인데 도구 포매터가 `title`을 읽었다.
- 확인 방법: retriever 반환 dict와 포매터가 읽는 키를 대조했다. `test_formatter_uses_retriever_paper_title`가 retriever와 같은
  키 이름의 픽스처로 이 경우를 재현한다.
- 해결: 포매터가 `paper_title`을 우선 읽고 `title`로 폴백한다(dc0981d). 같은 커밋에서 docstring과 줄바꿈 리터럴도 고쳤다.
  `test_agent_tools.py`의 포매터 테스트 9개가 통과한다.
- 재발 방지: 포매터 테스트가 retriever 반환 형태의 픽스처를 쓰고, CLAUDE.md의 Architectural Contracts에 Retrieval 결과
  shape를 적어 두었다.

### [참고문헌 섹션 판정을 제목 전체 일치 규칙 하나로](phase-0/2026-09-25_03_참고문헌-섹션-판정을-제목-전체-일치-규칙-하나로.md)

phase 0 · 2026-09-25 · pipeline, retrieval

**트러블슈팅** — 제목에 Preference가 든 본문 섹션이 참고문헌으로 분류됨

- 증상: "Direct Preference Optimization", "Reference Model" 섹션의 청크가 `content_role=references`로 저장되어 임베딩에서
  빠지고 lexical 결과에서 걸러졌다.
- 원인: 섹션 판정이 부분 문자열 검사(`'reference' in title`)였고, "Preference"도 "reference"를 포함한다.
- 확인 방법: 기준선 판정식 `'reference' in title`에 두 제목을 대입하면 참이 된다. 같은 두 제목을
  `tests/unit/test_pdf_parser.py`의 본문 픽스처에 넣어 분류 결과를 테스트로 확인한다.
- 해결: 제목 전체 일치 규칙 `is_references_section_title`로 바꾸고 retriever와 SQL이 공유하게 했다(dc0981d). 기존 청크는
  `scripts/backfill_content_roles.py --apply`로 재분류한다. `test_infer_content_role_uses_word_boundaries`와
  `test_sql_regex_agrees_with_parser_rule`이 통과한다.
- 재발 방지: 파서 규칙과 SQL 정규식이 같은 예시에서 같은 답을 내는지 `test_sql_regex_agrees_with_parser_rule`이 매 CI에서
  확인한다.

### [초록 폴백이 기존 본문과 임베딩을 지우지 않게 한다](phase-0/2026-09-25_02_초록-폴백이-기존-본문과-임베딩을-지우지-않게-한다.md)

phase 0 · 2026-09-25 · pipeline

**트러블슈팅** — 다운로드 실패 한 번으로 본문과 임베딩이 사라짐

- 증상: PDF 다운로드가 일시적으로 실패한 논문을 다시 prepare하면 본문이 초록으로 바뀌고 그 논문의 청크 임베딩이
  사라졌다.
- 원인: 초록 폴백 결과가 정상 파싱 결과와 같은 저장 경로를 탔고, 청크 교체의 `DELETE FROM paper_chunks`가
  `paper_embeddings`로 cascade됐다.
- 확인 방법: `src/pipeline/prepare_papers.py`의 저장 분기가 source를 구분하지 않는 것과 스키마의 cascade 제약을 코드에서
  확인했다.
- 해결: 파싱된 본문이 있으면 폴백을 저장하지 않게 했고(dc0981d), source 순위·`content_hash`·`--force`로 일반화했다
  (f229c59). `test_lower_ranked_source_does_not_overwrite_existing_fulltext`와
  `test_re_prepare_is_idempotent_and_never_downgrades_without_force`가 통과한다.
- 재발 방지: 단위 테스트는 CI python 잡, 통합 테스트는 pgvector 서비스를 띄우는 integration 잡에서 돈다. 낮은 순위로
  덮어쓰려면 `--force`를 명시해야 한다.

### [PDF 파서 믹스인 참조 복구](phase-0/2026-09-25_01_PDF-파서-믹스인-참조-복구.md)

phase 0 · 2026-09-25 · pipeline

**트러블슈팅** — 모든 PDF 파싱 경로가 예외로 끝남

- 증상: prepare 단계에서 어떤 논문이든 본문 파싱이 `NameError` 또는 `AttributeError`로 실패했다.
- 원인: a35b1d5 리팩터링에서 파서를 믹스인으로 나누면서 `FulltextParser` import, `_looks_like_numbered_heading` 메서드,
  `layout_parser.py`의 `Counter` import가 빠졌다.
- 확인 방법: 기준선 코드에 `tests/unit/test_pdf_parser.py`를 적용해 파서 테스트 14개가 모두 실패하는 것을 확인했다.
- 해결: 누락된 참조 세 가지를 복구했다(dc0981d). 같은 테스트 14개가 통과하고, `test_numbered_heading_helper_restored`가
  복구한 메서드를 직접 호출한다.
- 재발 방지: CI python 잡이 매 push에 `ruff check`(F 규칙이 정의되지 않은 이름을 잡는다)와 `pytest tests/unit`을 돈다.
