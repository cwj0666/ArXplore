---
title: 평가 하네스 arxplore_eval 이식과 lexical 기준선 재현
date: 2026-09-29
area: [eval, tooling]
decision: `eval/*.py`와 `scripts/eval_*.py`를 제품 private 코드에 묶이지 않는 `arxplore_eval` 패키지와 `arxplore-eval` 명령으로 옮기고 연구용 융합은 research_fusion에 둔다. 같은 가짜 벡터로 옛 하네스와 10방식 × 111질의가 같았고, 골든 코퍼스에서 lexical 3방식이 20260929-005408과 111/111 같다. vector·hybrid 재현은 G1 실제 질의 임베딩 표를 기다린다
---

**결정과 근거** — 옛 하네스는 retriever private 메서드 7개(01번 항목)와 모듈 monkeypatch에 묶여 있었다. 새 하네스는 제품의
공개 인터페이스만 쓴다(3261077).
- 검색 러너는 10방식을 공개 단계 함수로 조립한다: 채널별 fetch·rerank·filter, `fuse_hybrid_hits`, `apply_paper_diversity`,
  `Retriever.hybrid_inputs`, `build_contexts`(04번 항목).
- 답변 수집은 제품 `ChatService`를 쓰고, 발췌문은 모듈 monkeypatch 대신 검색 관찰 훅 `context.observe_retrieval`로 받는다.
  입력 검사는 `api.services.chat.validate_chat_input`을 다시 쓴다. 채팅 서비스 조립도 제품 공개 함수 `build_chat_service`로
  바꿔(4299ea0, f25ebad) 제품이 조립 방식을 바꾸면 수집도 따라간다.
- `arxplore_eval.research_fusion`은 옛 가중 RRF 규칙, 표준 RRF, 고정·confidence 가중, 이론 정규화, 품질 가중을 재생용으로
  둔다. 제품에는 convex 융합만 남긴다.
- 명령(734a0ae): `arxplore-eval retrieval|generation|fusion gate|sweep|convex|latency|build-queries|dump-candidates`가
  `scripts/eval_*.py`를 대체한다. 옛 옵션과 종료 코드(실패한 실행·gate 1, 전제 조건 미충족 2), `eval/results`·`eval/cache`의
  출력 파일 이름을 지켰고, `--openai-base-url`로 모든 OpenAI 클라이언트를 가짜 서버에 붙인다. `fusion gate`는 캐시를 제품
  융합으로도 재생해 research_fusion의 제품 설정과 다르면 실패한다. `explain`(6fa2ff2)이 `scripts/explain_retrieval.py`를
  대체한다.
- 패키지는 `arxplore`와 같은 wheel에 넣고 `arxplore-eval` 스크립트를 등록했다(24aaf12).

검증(3261077, 64d5433)
- LEGACY 융합 골든 fixture가 research_fusion에서 60/60(sha256 같음) 통과한다. research_fusion의 제품 설정은 합성 사례와
  캐시된 111질의 모두에서 `arxplore.retrieval.fusion`과 같은 순위를 낸다.
- 골든 코퍼스에서 옛 하네스와 같은 가짜 질의 벡터로 나란히 돌리면 10방식 × 111질의 = 1,110행의 순위와 지표가 같다.
- 대본을 따르는 가짜 LLM으로 답변 235건을 모으면 필드마다 같다. `answers_20260926-103836`의 행동 점수와 RAGAS 요청 계획
  430건도 바이트까지 같다. `build_chat_service`로 바꾼 뒤에도 235건이 지연 시간 말고는 같았다(f25ebad).
- fusion gate는 `candidates_20260928-040013`을 111/111로 재생한다.
- lexical 재현(`eval/results/20260929-133658`): 평가 당시와 같은 pgdata16 사본인 골든 DB를 읽기 전용으로 두고 lexical,
  lexical_nodiv, lexical_nofilter를 돌렸다. 세 방식 모두 111/111 질의에서 청크 id, arxiv id, role과 논문·청크·noise 지표가
  `20260929-005408`과 같다. 예를 들어 lexical 전체는 hit@1 0.514, hit@10 0.649, MRR@10 0.552로 같다. 다른 것은 지연뿐이다
  (lexical p50/p95 186/420ms 대 148/373ms). DB 컨테이너를 다른 테스트 실행과 함께 썼다.
- 01번 항목의 재발 방지로 약속한 검사를 넣었다. `queries.jsonl`이 `queries.generated.jsonl` 뒤에 `queries.cases.jsonl`을 이어
  붙인 것과 바이트까지 같은지 본다.

리포트 기호(−, ×, α)는 ASCII(-, x, alpha)로 바꿨다. v2의 ruff 설정이 켜는 RUF 규칙(RUF001)이 세 기호를 ASCII와 헷갈리는
문자로 잡는다. 옛 `eval/*.py`와 `scripts/eval_*.py`에는 이 기호가 든 줄이 54줄 있었고, `arxplore_eval`과 그 테스트에는
없다.

**트레이드오프** — 하네스를 공개 인터페이스에 묶어 제품 내부를 고쳐도 eval이 깨지지 않는다. 대신 공개 이름이 늘고, 그
시그니처를 바꾸면 eval도 함께 고쳐야 한다.
- 평가 코드를 제품 저장소에 두기로 한 결정(01번 항목)에 따라 제품과 같은 wheel에 넣었다(24aaf12). eval 의존성(ragas 0.4.3,
  tiktoken)은 dependency group이지만, ragas → instructor의 `jiter<0.15` 제약 때문에 lock 전체의 openai가 3.3.0에 묶인다
  (eval 그룹 없이는 3.20.0, 2118d75).
- scikit-network(ragas가 씀)는 cp314 휠이 없어 eval 그룹을 설치하려면 C++ 컴파일러로 빌드해야 한다.
- 지연 수치는 DB 컨테이너를 공유한 실행이라 옛 결과와 비교하지 않았다.

**eval 영향** — 새 측정 지점은 다음과 같다.
- 결과 파일 `eval/results/20260929-133658.{md,csv}`, `20260929-133658_summary.csv`. 머리말에 "구현: arxplore_eval (v2 공개
  인터페이스)"가 붙는다.
- `v2/tests/integration/eval/test_eval_golden_lexical_pg.py::test_lexical_methods_reproduce_the_recorded_evaluation`
  (`GOLDEN_DATABASE_URL`이 있을 때 lexical 3방식을 `20260929-005408.csv`와 비교)
- `v2/tests/unit/eval/test_eval_research_fusion.py`(LEGACY 골든 `test_legacy_rules_reproduce_the_recorded_output`),
  `test_eval_candidate_cache.py`(커밋된 후보 캐시의 gate), `test_eval_query_sets.py::test_queries_jsonl_is_generated_then_cases_byte_for_byte`
- `v2/tests/integration/eval/test_eval_cli_pg.py`: 작은 pgvector 코퍼스를 심고 가짜 OpenAI 서버로 모든 명령을 돌린다.

**알려진 한계** — vector·hybrid를 기록된 실행과 비교하려면 G1의 실제 질의 임베딩 표가 필요하다. 기록된 실행은 실제
OpenAI 벡터를 썼고, 표는 아직 없다.
- 실제 RAGAS 채점은 C++ 컴파일러(build-essential)를 설치한 뒤에야 돈다.
- 나란히 비교(1,110행, 답변 235건, RAGAS 요청 430건)는 일회성 실행이라 저장소 테스트로 남지 않았다. 자동으로 남은 검사는
  lexical 골든 테스트, LEGACY 융합 골든, 커밋된 후보 캐시 gate, 질의셋 결합 검사다.
- 수정 후 코퍼스(F01~F04 전환)의 검색 지표와 생성 평가 재수집은 아직 없다.
- `20260929-133658` 머리말의 커밋은 `fcde8cb (dirty)`다. 하네스를 커밋(3261077)하기 전의 작업 트리에서 돌렸으므로, 같은
  코드는 3261077 이후 커밋에 있다.

**트러블슈팅** — 해당 없음
