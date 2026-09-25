---
title: 에이전트 검색 도구를 hybrid RRF로, 키가 없으면 lexical
date: 2026-09-25
area: [retrieval, agent]
decision: 에이전트 검색 도구가 질의 임베딩 키가 있으면 lexical·vector를 RRF(k=60)로 합친 hybrid 검색을 쓰고, 키가 없거나 임베딩 호출이 실패하면 lexical로 내려가며, RETRIEVAL_MODE로 강제할 수 있다
---

**결정과 근거** — 기준선 에이전트 도구는 PostgreSQL 전문 검색(lexical)만 썼다. 설정이 `english`라 한국어 질문은 거의 맞지
않는다. `src/core/agent/retrieval.py`의 `retrieve_contexts`가 `RETRIEVAL_MODE=hybrid`(기본)이고 질의 임베딩 키가 있으면
lexical과 vector(pgvector 코사인, text-embedding-3-large 1536차원) 결과를 reciprocal rank fusion(k=60)과 content-role 재정렬로
합친다. 키가 없거나 임베딩 호출이 `OpenAIError`로 실패하면 lexical로 내려간다. 에이전트 도구와 상세 챗이 같은 함수를 쓴다
(985ce4f).

**트레이드오프** — hybrid는 질의마다 임베딩 API 호출이 하나 더 들고, lexical과 vector 두 질의를 모두 기다리므로 지연이 가장 길다
(최종 실측 p50 483ms, lexical 202ms, vector 173ms). 표준 RRF 대신 lexical 신뢰도에 따라 가중치를 옮기고 두 채널 동시 hit에
보너스를 주는 규칙을 썼는데, 이 규칙이 표준 RRF보다 나은지는 `hybrid_plainrrf` ablation으로 따로 잰다.

**eval 영향** — `tests/unit/test_agent_tools.py`의 `test_search_tool_uses_hybrid_when_embedding_key_available`,
`test_search_tool_uses_lexical_without_embedding_key`, `test_search_tool_honors_lexical_retrieval_mode`,
`test_no_key_means_unavailable_and_retrieval_degrades_to_lexical`, `test_detail_chat.py`의
`test_embedding_api_error_degrades_to_lexical`. 검색 평가에서 lexical·vector·hybrid 세 경로와 ablation을 같은 질의셋으로
비교한다.

**알려진 한계** — 로컬 코퍼스 최종 실행에서 hybrid는 vector를 넘지 못했다(hit@1 0.704 대 0.704, hit@10 0.817 대 0.826,
MRR@10 0.737 대 0.741). `hybrid_plainrrf`는 hit@1 0.600으로 낮지만 hit@10 0.835로 가장 높다. 가중치 규칙이 상위 1위를 올리는
대신 상위 10위 안의 재현을 줄이고 있다. 결과 해석은 phase-4 검색 결과 항목에 있다.

**트러블슈팅** — 해당 없음
