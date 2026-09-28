---
title: convex combination 융합 비교 사전 등록
date: 2026-09-29
area: [retrieval, eval]
decision: 채널 점수를 정규화해 α·lexical + (1−α)·vector로 합치는 convex combination(CC)을 현재 가중 규칙 C·표준 RRF·vector 단독과 같은 캐시 재생·클러스터 CV·1-SE·클러스터 paired bootstrap으로 비교하기로 하고, 측정 전에 정규화 3종·α 격자·결측 점수 규칙·비교 쌍·판정 규칙을 이 항목에 고정한다
---

**결정과 근거** — 2026-09-28 재검증(`2026-09-28_02`)은 순위 기반 융합(RRF 가족)만 비교했다. 결과는 C(현재 가중 규칙) 0.792,
표준 RRF 0.785, vector 단독 0.789(논문 MRR@10)로 구분되지 않았고, 성능을 좌우한 것은 부분 일치 lexical 행 제거였다. 이번에는
점수 수준 융합을 잰다. 질문은 "채널 점수를 정규화해 볼록 결합하면 RRF보다 각 채널의 강점을 더 잘 살리는가"다. 예로
lang-abbr-dpo는 두 채널 모두 정답 논문을 1위로 올렸는데 hybrid 융합 뒤 2위가 됐다(기준선 `20260928-034856.csv`).

방법 출처는 Bruch, Gai, Ingber, "An Analysis of Fusion Functions for Hybrid Retrieval", ACM Transactions on Information Systems
42(1), 2023, doi:10.1145/3596512 (arXiv:2210.11934)다. 서지는 Crossref, 주장은 arXiv 본문으로 확인했다. 초록의 주장은 네 가지다.
RRF는 매개변수에 민감하다. CC 학습은 점수 정규화 선택에 대체로 무관하다. CC는 in-domain·out-of-domain 모두에서 RRF보다 낫다.
CC는 표본 효율적이라 유일한 매개변수를 적은 학습 예로 맞출 수 있다. 논문의 식은 `α·φ_Sem + (1−α)·φ_Lex`로 α가 **semantic** 쪽에
붙는다. 권장 범위 α ∈ [0.6, 0.8]은 이 문서 표기(α가 lexical 쪽)로는 0.2~0.4다. 논문은 MS MARCO·BEIR 8개에서 BM25 + 
all-MiniLM-L6-v2, 채널당 1,000개 후보, NDCG@1000·Recall@1000으로 쟀다. 이번 설정(채널당 30개, 논문 MRR@10, 111개 질의)과
다르므로 논문의 결론이 여기서도 성립한다고 가정하지 않는다.

측정 전에 캐시 입력(`eval/cache/candidates_20260928-040013.jsonl.gz`, sha256 d7cfdbc24a82)의 점수 분포만 봤다. CC 결과는 아직 하나도
계산하지 않았다. 본 것은 다음과 같다. lexical strict 행 242개(점수 0.851~11.02), 부분 일치 행 2,214개(0.013~1.038), vector
후보 3,330개(질의당 30개, 0.161~1.056)다. 부분 일치 행을 빼면 **111개 중 78개 질의에서 lexical 채널이 비고**, 7개는 1개만 남는다.
이 78개에서는 C와 α < 1인 모든 CC가 vector 순위와 같다. 따라서 융합 방식 간 차이는 나머지 33개 질의에서만 생길 수 있다.

(1) 점수가 실제로 무엇인가. 정규화 하한은 이 정의에서 정한다.
- lexical `score` = SQL `fts_score + ilike_bonus + content_role_adjustment + section_boost + structural_adjustment`
  (`build_lexical_candidates_query`) + retriever의 section-intent 보너스(0~0.14, `_rerank_lexical_candidates`)다.
  strict 행의 `fts_score`는 `ts_rank_cd(websearch) + 0.65·ts_rank_cd(plainto) + STRICT_MATCH_BONUS(1.0)`이고, 부분 일치 행은
  `coverage × ts_rank_cd(any_term, 32)` ∈ [0, 1)이다. ilike 보너스는 strict에만 0~0.45가 붙는다. 조정 항은 content_role −0.24~0
  (toc는 `screened`에서 이미 빠져 −0.28은 나오지 않는다), section_boost −0.08~+0.16, structural −0.12~0이다. SQL은
  `score > 0.01`인 행만 남기고 그 뒤에 붙는 보너스는 0 이상이다. 그래서 **전달되는 모든 lexical 점수는 0.01보다 크고**, 0은 그
  하한이다(논문의 BM25 infimum 0과 같은 자리). strict 행만 보면 infimum은 1.0 − 0.24 − 0.08 − 0.12 = **0.56**이다(ts_rank_cd와
  ilike ≥ 0, 세 조정 항의 최솟값). ts_rank_cd는 정규화하지 않은 값이라 상한은 없다.
- vector `score` = 코사인 유사도(`1 − (embedding <=> query)`) + content_role 조정(−0.22~0, toc 제외) + section_boost(−0.1~+0.12)
  (`build_vector_search_query`) + retriever rerank(`_rerank_vector_candidates`)다. rerank는 질의 토큰 겹침 +0~0.12, 부록 −0.08,
  결론 −0.03, 참고문헌·감사 −0.18, front_matter·table_like −0.02, 참고문헌 같은 본문 −0.14, section-intent +0~0.14다. 코사인의
  infimum은 −1이다(논문의 예와 같음). 조정 항까지 더한 엄밀한 infimum은 −1.77이지만 쓰지 않는다. 논문의 요점은 데이터 의존 최솟값을
  고정 상수로 바꾸는 것이다. 하한이 상수이면 정규화는 채널 안에서 단조 affine이라 값이 하한 아래로 가도 순위 정의에 문제가 없다.
  채널 점수가 이미 코사인에 가산 보정을 더한 값이라는 점은 한계로 남긴다(보정 전 코사인은 캐시에 없다).

(2) 설정. 새 융합 모드 `weighting="convex"`를 `HybridFusionConfig`에 더하고 같은 재생 경로(`fuse_hybrid_candidates` →
`apply_paper_diversity`)로 돌린다. **제품 기본값 `DEFAULT_HYBRID_FUSION`은 바꾸지 않는다.**
- 융합 점수: `s(d) = α·φ_lex(d) + (1 − α)·φ_vec(d)`. α는 **lexical** 가중치다(논문과 반대 표기).
- 정규화 φ는 채널마다 따로, 그 채널이 융합에 넘긴 후보 목록(부분 일치 행 제거 뒤) 위에서 계산한다. 논문은 두 채널 합집합에서
  결측 점수를 실제로 계산한 뒤 정규화한다. 캐시에는 채널당 상위 30개만 있어 그렇게 할 수 없다.
  - `mm`(per-query min-max): `(s − min) / (max − min)`.
  - `tmm`(theoretical min-max, 하한 lexical 0 / vector −1): `(s − L) / (max − L)`.
  - `tmms`(theoretical min-max, 하한 lexical 0.56 / vector −1): strict 행만의 lexical infimum을 쓴다. 부분 일치 행 제거가 켜진
    주 설정에서는 융합되는 lexical 행이 모두 strict이므로 이것이 그 집합의 이론 하한이다.
- 결측 점수 규칙: 한 채널에만 나온 후보의 다른 채널 정규화 점수는 **0**이다. `mm`에서는 그 채널 최하위 후보와 같은 값이고,
  `tmm`·`tmms`에서는 이론 하한이다. 논문처럼 실제 점수를 계산할 수 없어서 정한 규칙이다.
- 퇴화 규칙: `max − min`(mm) 또는 `max − L`(tmm·tmms)이 0 이하이면(후보 1개, 전부 동점, 하한 이하 최댓값) 그 채널의 후보는 모두
  **1.0**이다. 채널의 최상위 후보에게 tmm이 주는 값과 같다. 빈 채널은 기여가 없다(모든 후보가 결측 0). 주 설정에서 tmm·tmms의
  퇴화는 입력 범위상 일어나지 않고, mm의 퇴화는 lexical 1개인 질의 7개에서 일어난다. 결과 리포트에 퇴화 발생 수를 적는다.
- 동점·후처리: 정렬 키는 기존 융합과 같게 (융합 점수, 일치 채널 수, chunk_id) 내림차순이다. 그 뒤 기존과 같은 `apply_paper_diversity`
  (k=10, 논문당 2개)를 적용한다. RRF 상수, 교차 보너스, 품질 가중, 방법 가중은 CC에서 쓰지 않는다.
- α 격자: {0.00, 0.05, …, 0.95} 20개. α = 1.0은 뺀다. lexical이 빈 78개 질의에서 모든 융합 점수가 0이 되어 순위가 chunk_id 순이
  되기 때문이다(lexical 단독은 R0로 따로 있다). α = 0.00은 넣는다. vector 순위에 lexical 전용 후보를 맨 뒤에 붙인 것이라 vector
  단독과 거의 같아야 하는 점검값이다.
- 가족: **CC**(주 설정, 부분 일치 행 제거 켬, C와 같음) = 3 정규화 × 20 α = 60개, 선택 대상. **CCK**(보조, 부분 일치 행 제거 끔)
  = `mm`·`tmm` × 20 α = 40개, 보고만 하고 선택·판정에 쓰지 않는다. `tmms`는 부분 일치 행(0.56 미만 가능)이 섞이는 CCK에서 하한이
  성립하지 않으므로 뺀다. 비교 기준은 09-28과 같은 R0(vector·lexical 단독), F0(표준 RRF k 격자), F1, F2, C, CF, REF 전체를 같은
  실행에서 다시 재생해 가족 CV 표에 함께 싣는다.
- 상수 수(`count_parameters`): CC 설정은 모두 1(α)이다. 정규화 종류와 이론 하한은 점수 정의에서 정한 값이라 세지 않는다.

(3) 선택과 비교. 주 지표는 전체 111개 질의의 논문 MRR@10이다. hit@1/5/10, recall@10, 청크 hit@5와 source·lang 부분집합은 함께
보고한다. CV, 1-SE, bootstrap은 `2026-09-28_01`과 같다. 클러스터 키는 id 끝 `-ko`/`-en`을 뗀 값이고 74개 클러스터다. CV는 클러스터
층화 5-fold × 10회 반복(seed 20260928)이고 fold 배정은 모든 가족이 공유한다. pick은 **CC 가족 안에서** 1-SE 규칙으로 고른다. 전체
데이터 MRR 최고 CC 설정의 CV SE만큼 낮춘 기준 이상인 설정 중 상수가 가장 적은 설정이다. CC는 모두 상수 1개라 이 규칙은 사실상
"전체 데이터 최고 CC 설정"이 된다. 동률은 이름 순이다(`CC_mm_…` < `CC_tmm_…` < `CC_tmms_…`, 같은 정규화 안에서는 α가 작은 쪽, 즉
vector에 가까운 쪽). 기준 이상 목록과 동률 설정을 리포트에 남긴다. paired bootstrap은 클러스터 복원 추출 10,000회(seed 20260928,
manual은 20260929)다. 비교 쌍은 pick − C, pick − F0(k=60), pick − vector 단독이고, 참고로 C − F0(k=60), C − vector 단독도 낸다.
각 쌍을 전체와 manual 부분집합(43개, 클러스터 33개)에서 ΔMRR@10과 Δhit@1의 95% 백분위 구간으로 낸다.

판정 규칙은 다음과 같다(리포트의 "사전 등록 규칙 판정" 절이 기계적으로 적용한다). 09-28과 같은 문장이고 pick만 CC pick으로 바뀐다.

- 비회귀 조건: manual 부분집합에서 pick − C ΔMRR@10 구간의 상한이 0보다 작으면 교체하지 않는다.
- (1) pick − C 구간이 0을 포함하면 더 단순한 pick(상수 1개 대 25개)으로 교체한다. 구간 전체가 0보다 크면 역시 교체한다.
- (2) C가 유의하게 나으면(구간 상한 < 0) 현재 규칙을 유지한다.
- (3) 최종 설정(pick 또는 C)이 vector 단독보다 유의하게 낫지 않으면(구간 하한 ≤ 0) vector 단독 근거로 보고하고, 기본 검색 방식의
  결정은 사용자에게 넘긴다.
- (4) 논문 주장 점검(제품 판정에는 쓰지 않음): pick − F0(k=60) 구간 하한 > 0이면 "이 설정에서 CC가 RRF보다 낫다"를 재현한 것이고,
  상한 < 0이면 반대이며, 그 밖에는 구분되지 않는다고 적는다. F0(k=60)와 CC pick은 둘 다 상수 1개라 규칙 (1)은 둘 사이에 선호를
  주지 않는다.

제품 설정 변경은 규칙이 가리키는 방향과 상관없이 사용자가 정한다.

함께 보고하는 서술 항목(판정에 쓰지 않음): pick과 C, pick과 vector 단독 사이에 논문 RR@10이 달라지는 질의 수·id·방향. 기준선에서
hybrid와 vector가 갈린 7개 질의(ki-2609.06251-en, ki-2609.20804-en, lang-en, ki-2609.27334-en, lang-abbr-dpo, mp-first-proposed-en,
qf-long-paragraph-en)의 C·vector 단독·pick 순위. lang-abbr-dpo에서 정답 논문이 몇 위에 오는지(pick과 CC 60개 설정 각각). 정규화별
하위 가족(CC_mm·CC_tmm·CC_tmms) CV와 α 곡선. CCK 가족의 전체 데이터 최고값과 CV. 퇴화 정규화 발생 수.

**트레이드오프** — 재생 방식을 그대로 쓰므로 새 DB·임베딩 호출이 없고 09-28 결과와 같은 입력·fold·표본으로 비교된다. 대신 논문과
다른 점이 셋 있다. 결측 점수는 실제 점수가 아니라 0으로 채운다. 정규화 범위가 합집합이 아니라 채널 상위 30개다. 점수가 순수 코사인·
BM25가 아니라 가산 보정이 섞인 값이다. z-score 정규화는 넣지 않았다. 논문이 min-max와 z-score 해의 대응을 보였고, lexical 후보가
1개인 질의(7개)에서 분산이 0이라 정의되지 않기 때문이다. α 격자를 0.05 간격으로 둔 것은 논문 권장 구간(lexical 0.2~0.4)을 여러
점으로 보려는 선택이다. 1-SE 규칙은 CC 안에서 단순성 차이가 없어 선택 편향을 줄여 주지 못한다. 그래서 CV 값으로만 가족 간 성능을
말하고, pick − X 구간은 선택 편향만큼 낙관적이라고 리포트에 적는다.

**eval 영향** — 새 측정 지점은 `eval/results/fusion_<timestamp>.{md,csv}`(CC 가족 포함 전체 설정, CC pick 기준 판정, 질의별 차이,
정규화별 하위 가족 CV)다. 입력 캐시는 09-28 것을 그대로 쓴다. 게이트(기본 설정 재생 = live hybrid, 111/111)를 먼저 통과해야 결과를
쓴다. 테스트는 정규화·결합 계산(후보 1개, 전부 동점, 한 채널이 빔, 한 채널에만 있는 후보, 이론 하한, 퇴화)과 CC 설정 검증,
`count_parameters`, sweep 명세에 더한다.

**알려진 한계** — 질의 111개의 단일 질의셋이고 manual 43개라 부분집합 구간이 넓다. 78개 질의에서 lexical 채널이 비어 있어, 어떤
융합이든 차이는 33개 질의에서만 나온다. 결측 0 규칙은 "상위 30개에 없으면 최하위와 같거나 그보다 낮다"는 가정이다. 채널 안
per-paper 상한, rerank, 부분 일치 행 제거 때문에 실제로는 그보다 높은 점수를 가진 청크가 빠져 있을 수 있다. 이 항목은 측정 전에
쓴 것이고, 결과와 채택 여부는 별도 항목에 기록한다.

**트러블슈팅** — 해당 없음
