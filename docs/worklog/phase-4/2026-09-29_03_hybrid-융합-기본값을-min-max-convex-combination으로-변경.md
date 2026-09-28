---
title: hybrid 융합 기본값을 min-max convex combination으로 변경
date: 2026-09-29
area: [retrieval, eval, docs]
decision: 사전 등록 규칙 (1)이 가리킨 CC pick(min-max 정규화, lexical 가중치 0.35, 부분 일치 행 제거)을 사용자가 채택해 제품 hybrid 융합 기본값으로 바꾸고, 예전 가중 RRF 규칙은 LEGACY_RULES_FUSION으로 남긴다. live 재측정은 MRR@10 0.805 / hit@1 0.766으로 오프라인 재생과 111개 질의 모두 같다. vector 단독 대비 이득은 유의하지 않다
---

**결정과 근거** — `2026-09-29_02`의 판정에 따라 사용자가 규칙 (1)을 채택했다. 규칙 (1)은 C를 CC pick `CC_mm_a0.35`로 바꾸라고
가리켰다. 09-28_02에서는 같은 규칙 (1)이 표준 RRF를 가리켰지만 사용자가 C를 유지했다. 이번에는 규칙대로 교체한다. 근거는 세 가지다.
pick은 상수가 1개(C는 25개)다. C와 순위가 갈린 3개 질의에서 모두 이긴다. 표준 RRF보다 구간상 낫다(규칙 (4), [+0.005, +0.040]).
규칙 (3)이 말하는 대로 vector 단독보다 유의하게 낫지는 않다(ΔMRR +0.016, [−0.001, +0.038]). 기본 검색 방식은 hybrid로 유지한다.
유지 이유는 전과 같이 임베딩 키가 없을 때의 lexical 폴백을 한 경로로 관리하는 것이다.

바꾼 것(d73e136):
- `DEFAULT_HYBRID_FUSION`은 `HybridFusionConfig()`와 같다. `weighting="convex"`, `convex_alpha=0.35`, `score_normalization="minmax"`,
  `drop_partial_lexical=True`다. 필드 기본값을 제품 기본값에 맞춰, 인자 없는 생성자가 예전 규칙을 돌려주는 함정을 없앴다. RRF
  상수 필드의 기본값은 예전 규칙의 값 그대로다.
- 예전 규칙은 `LEGACY_RULES_FUSION = HybridFusionConfig(weighting="rules")`다. sweep의 C·CF·REF 가족, golden fixture
  (`hybrid_fusion_golden.json`, 리팩터링 전 출력) 테스트, 규칙 분기 테스트가 이 설정으로 돈다. sweep의 설정 이름 `C_current`는
  예전 리포트와 비교할 수 있게 그대로 뒀다.
- eval에 ablation `hybrid_rules`(예전 규칙을 제품 입력에 적용)를 더했다. 같은 실행에서 예전 규칙을 다시 재기 위해서다.
- 재생 게이트는 캐시 머리말의 `fusion_default`(기록 시점 제품 설정, `recorded_fusion_config`)로 재생한다. 기본값이 바뀌어도
  09-28 캐시(기록 시점 `weighting="rules"`)의 게이트는 111/111로 그대로 통과한다. 머리말에 없는 새 필드는 지금 기본값을 쓰는데,
  `rules` 모드에서는 convex 필드를 읽지 않으므로 결과가 같다.
- `PaperRetriever`에서 예전 규칙의 가중치·품질 가중을 테스트에 노출하던 래퍼 3개를 지웠다. 테스트는 모듈 함수에
  `LEGACY_RULES_FUSION`을 넘긴다. 에이전트 검색 도구 설명(LLM에 보내는 문자열)의 "RRF로 결합"은 "점수를 결합"으로 바꿨다.
  생성 평가는 다시 돌리지 않았다.
- 새 테스트: 기본값이 pick 설정과 같은지, 예전 상수 전체 보존, 제품 병합 경로가 기본값을 쓰는지, lang-abbr-dpo 유형 합성 입력에서
  기본값은 vector 1위를 지키고 예전 규칙은 두 채널 중간의 다른 논문 청크를 1위로 올리는지, 머리말 설정으로 게이트하는지.

오프라인 게이트: 09-28 캐시에서 새 기본값 재생, `CC_mm_a0.35` 재생, 제품 `_merge_hybrid_candidates` 기본 호출의 순위가 111개 질의
모두 같다(설정도 `==`로 같다). MRR@10 0.8051, hit@1 0.7658이다.

live 재측정: 로컬 PostgreSQL(재구축 코퍼스 덤프 복원, 논문 275, 청크 16,010, 임베딩 13,328)에서 09-28 기준선과 같은 방식·ablation에
`hybrid_rules`를 더해 돌렸다(`eval/results/20260929-005408.{md,csv}`·`_summary.csv`, 커밋 d73e136). 질의셋은 09-28_02의 매핑 표
(청크 합성 질의 11개의 chunk id)로 저장소의 `eval/queries.jsonl`에서 다시 만들었다. sha256 fe0cce78a89b로 기준선과 같다. 매핑한
파일은 이번에도 저장소에 넣지 않았다.

| 방식 (전체 111개) | hit@1 | hit@5 | hit@10 | MRR@10 | recall@10 | 청크 hit@5 |
|---|---|---|---|---|---|---|
| hybrid (새 기본) | 0.739 → **0.766** | 0.865 | 0.883 | 0.792 → **0.805** | 0.836 → 0.834 | 0.636 |
| hybrid_nodiv | 0.739 → 0.766 | 0.865 | 0.883 | 0.790 → 0.805 | 0.834 | 0.636 |
| hybrid_rules (예전 규칙, 새 ablation) | 0.739 | 0.865 | 0.883 | 0.792 | 0.836 | 0.636 |
| vector | 0.748 | 0.856 | 0.883 | 0.789 | 0.836 | 0.636 |
| lexical | 0.514 | 0.586 | 0.649 | 0.552 | 0.612 | 0.273 |
| hybrid_plainrrf | 0.586 | 0.784 | 0.883 | 0.677 | 0.841 | 0.727 |

화살표 왼쪽이 20260928-034856, 오른쪽이 이번 값이다. 화살표가 없는 줄은 두 실행이 같다. lexical_nodiv, vector_nodiv, lexical_nofilter,
vector_norerank도 모든 부분집합에서 기준선과 같다. `hybrid_rules`는 기준선 hybrid와 모든 부분집합·지표에서 같다. 따라서 DB와
질의셋이 기준선을 그대로 재현하고, 바뀐 것은 융합뿐이다. 부분집합별로 보면 en MRR@10 0.828 → 0.856(hit@1 0.774 → 0.830), manual
0.559 → 0.582, known_item 0.944 → 0.953이고, ko(0.758)와 llm_synth(0.909)는 같다.

live와 오프라인 비교: live hybrid 0.805 / 0.766은 오프라인 pick과 같다. 질의별 논문 지표(hit@1/5/10, MRR@10, recall@10)도 111개
모두 같다. 순위가 갈린 질의도 오프라인과 같다. 예전 규칙과는 3개(ki-2609.27334-en, lang-abbr-dpo, qf-long-paragraph-en, 모두 2위 →
1위), vector와는 4개다(ki-2609.06251-en, ki-2609.20804-en, lang-en은 hybrid 우세, mp-first-proposed-en은 vector 우세). 청크 순위까지
보면 2개 질의(ki-2609.10715-ko, ki-2609.18323-en)가 다르다. 둘 다 vector 채널 8~10위의 거의 같은 점수(예: 코사인 차 0.00002)
사이의 순서라 정답 논문 순위와 무관하다. 질의 임베딩 호출마다 벡터가 조금씩 달라서 생긴다. 09-28에도 기준선 실행과 캐시 기록
실행이 같은 두 질의에서 같은 식으로 갈렸다.

지연: hybrid는 전과 같이 두 채널을 병렬로 실행하고 융합 계산만 바뀌었으므로 지연 A/B(`latency_20260928-151847`)는 다시 재지 않고
README 값을 그대로 쓴다. 이번 실행의 지연 열(hybrid p50 263ms)은 교대 측정이 아니라 비교에 쓰지 않는다. 기준선의 지연 열(568ms)은
다른 장비·순차 코드 값이다.

문서: README(검색 계층 hybrid 줄, Evaluation 표와 설명, 지연 문단, 로드맵 "hybrid 대 vector 단독"), `ARCHITECTURE.md` hybrid 줄,
`AGENTS.md` 검색 경로 문장, `CLAUDE.md` Retrieval Hybrid 항목, `eval/README.md`(ablation 표, 게이트 설명, C 가족 설명)를 고쳤다.
README Evaluation 표는 첫 재구축 코퍼스의 `20260926-103105`에서 이번 실행으로 바꿨다. 청크 수가 16,011 → 16,010이다.

**트레이드오프** — 상수 25개의 근거 없는 규칙이 α 하나로 줄었다. 대신 그 α는 같은 111개 질의로 골랐다. 이득의 근거는 3개 질의뿐이고
(가족 CV 0.799 ± 0.037은 C 0.792와 SE 안), vector 단독과는 구분되지 않는다. min-max는 질의마다 채널 상위 30개의 최솟값에 기대므로,
후보 수(`hybrid_branch_limit`)나 채널 점수 보정(SQL 가중, rerank)을 바꾸면 α를 다시 확인해야 한다. 예전 규칙을 지우지 않고 이름 붙인
설정으로 남겨, 재생 게이트와 ablation으로 언제든 다시 잴 수 있게 했다. 게이트를 머리말 설정으로 바꾼 것은, 기본값이 바뀔 때마다
기존 캐시를 다시 기록하지 않기 위해서다.

**eval 영향** — 새 기준선 `eval/results/20260929-005408.{md,csv}`·`_summary.csv`. 새 ablation `hybrid_rules`(`--ablations rules` 또는
`all`). 앞으로 기록하는 캐시의 `fusion_default`는 convex 설정이다. 재생 게이트는 머리말 설정을 쓴다. 생성 평가(RAGAS)와 지연
A/B는 다시 재지 않았다.

**알려진 한계** — α는 같은 질의셋으로 골랐고, 새 질의셋으로 확인하지 않았다. 코퍼스를 한 번 재구축한 것 하나라 코퍼스 간 분산은
모른다. 상세 챗(논문 범위 검색)은 같은 융합을 쓰지만 논문 범위 검색 품질은 따로 재지 않았다. 에이전트 도구 설명 문구를 바꾼 뒤의
생성 평가도 재지 않았다.

**트러블슈팅** — 해당 없음
