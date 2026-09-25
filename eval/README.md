# 오프라인 검색 평가

`PaperRetriever`의 lexical / vector / hybrid 경로와 일부 ablation을 같은 질의셋으로 비교하는 하니스입니다.
retriever 코드는 수정하지 않고, 공개 함수(`search_paper_contexts*`)와 하위 단계를 그대로 호출합니다.

| 파일 | 역할 |
| --- | --- |
| `eval/metrics.py` | hit@k, MRR@k, recall@k, percentile (순수 함수) |
| `eval/dataset.py` | 질의셋 JSONL 스키마·검증·로더, 질의 생성 프롬프트 |
| `eval/runner.py` | 방식 레지스트리(기본 + ablation), 질의 실행, 집계. retriever는 주입받음 |
| `eval/report.py` | 질의별·집계 CSV, Markdown 표 |
| `eval/queries.sample.jsonl` | 형식 예시 6건. arxiv_id·chunk_id는 **가짜 자리표시자**(`0000.0000x`, `90000x`) |
| `scripts/eval_build_queries.py` | DB 표본 + LLM으로 `eval/queries.jsonl` 생성 (기본 dry-run) |
| `scripts/eval_retrieval.py` | 평가 실행 → `eval/results/<timestamp>.{csv,md}`, `<timestamp>_summary.csv` |

단위 테스트(`tests/unit/test_eval_metrics.py`, `tests/unit/test_eval_runner.py`)는 가짜 retriever로 전 과정을 돌리며 DB·API 키가 필요 없습니다.

## 전제 조건 (측정 전에 반드시)

측정은 아래가 끝난 DB에서만 의미가 있습니다. 하나라도 빠지면 결과를 README·포트폴리오에 쓰지 않습니다.

1. **Phase 0 파서 복구가 반영된 prepare-worker로 재처리**: `scripts/requeue_failed_prepare_jobs.py`로 failed 잡을 재등록하고 처리 완료까지 대기. `paper_fulltexts.source`에서 `layout_pdf`/`pdf` 비중을 확인합니다(평가 리포트의 "본문 source" 줄에 기록됨).
2. **`content_role` 백필**: `scripts/backfill_content_roles.py` dry-run → 승인 후 `--apply`. "Reference Model", "Direct Preference Optimization" 같은 섹션이 body로 돌아와야 합니다.
3. **임베딩 backlog 소진**: vector/hybrid를 잴 때 references가 아닌 청크 대부분에 임베딩이 있어야 합니다. 리포트의 `chunks` 대비 `embeddings` 수로 확인합니다.
4. **질의셋은 1~3 이후에 생성**: chunk_synth 질의의 정답은 `chunk_id`인데, 재청킹·백필로 청크가 다시 만들어지면 id가 바뀝니다. `eval_retrieval.py`는 정답 id가 DB에 없으면 실행을 거부합니다.

## 실행

```bash
# 0) 형식 확인 (DB 불필요)
pytest tests/unit/test_eval_metrics.py tests/unit/test_eval_runner.py

# 1) 질의셋 표본 확인 — DB만 필요, LLM 미호출
python scripts/eval_build_queries.py --papers 20 --chunks 10 --seed 42

# 2) 생성 — DB + OPENAI_API_KEY. known_item 20편 × ko/en = 최대 40개 + chunk_synth 최대 10개
python scripts/eval_build_queries.py --generate --papers 20 --chunks 10 --seed 42

# 3) 사람이 검토: eval/queries.jsonl에서 정답이 모호하거나 너무 쉬운(제목을 그대로 쓴) 질의를 지우고,
#    직접 고친 항목은 source를 "manual"로 바꾼다. 필요하면 manual 질의를 손으로 추가한다.

# 4) 평가 — lexical만 (API 키 불필요)
python scripts/eval_retrieval.py --methods lexical

# 5) 평가 — 3방식 + 모든 ablation (DB + OPENAI_API_KEY; 질의 임베딩 비용 발생)
python scripts/eval_retrieval.py --ablations all

# 일부만 빠르게
python scripts/eval_retrieval.py --limit 5 --methods lexical,hybrid
```

주요 옵션: `--k 10`, `--adjacency-window 1`(제품 경로와 동일한 문맥 창, 지연에 포함), `--lang ko|en`,
`--warmup 1`(방식별로 첫 질의를 한 번 버림), `--keep-going`(질의 단위 예외를 기록하고 계속; 기본은 즉시 중단).

DB에 연결할 수 없거나, 정답 id가 DB에 없거나, vector 계열인데 키·임베딩이 없으면 **아무 파일도 쓰지 않고 종료 코드 2**로 끝납니다.
`--keep-going`으로 실패한 질의가 있으면 결과 파일은 쓰되 종료 코드 1이고, 실패 건은 집계에서 빠지며 `errors` 열에 개수가 남습니다.

## 질의셋 형식 (`eval/queries.jsonl`)

한 줄에 JSON 하나:

```json
{"id": "ki-2405.01234-ko", "query": "...", "lang": "ko", "relevant_arxiv_ids": ["2405.01234"], "source": "known_item", "notes": "basis=abstract; title=..."}
{"id": "cs-48213", "query": "...", "lang": "en", "relevant_arxiv_ids": ["2405.01234"], "relevant_chunk_ids": [48213], "source": "llm_synth", "notes": "chunk_index=12; section=Experiments"}
```

- `lang`: `ko` | `en`. `source`: `known_item`(초록·key findings 바꿔 말하기) | `llm_synth`(청크 하나로만 답할 수 있는 질문) | `manual`.
- `relevant_chunk_ids`는 선택. 있으면 청크 단위 지표도 계산합니다.
- 생성기는 입력과 5단어 이상 연속으로 겹치는 질의를 버립니다(`--max-shared-words`).

## 지표와 해석

- **논문 단위**: 상위 k개 hit의 `arxiv_id` 순서로 계산. 같은 논문 청크가 여러 자리를 차지하면 그 자리는 소모되지만 recall은 한 번만 셉니다.
- **청크 단위**: 상위 k개 hit의 `chunk_id`(문맥 창의 이웃 청크는 제외)로 계산. `relevant_chunk_ids`가 있는 질의만 대상.
- hit@1/5/10, MRR@10, recall@10은 질의 평균. 정답이 비어 있는 질의는 해당 지표에서 제외합니다.
- **noise@10**: 상위 10개 hit 중 `content_role ∈ {references, toc, front_matter}`인 비율.
- **지연 p50/p95**: 방식 호출 1회의 wall-clock(ms). 문맥 창 조회 DB 왕복, vector 계열은 OpenAI 임베딩 API 왕복까지 포함합니다. 네트워크(Tailscale 경유 DB 등) 영향이 크므로 같은 환경에서 잰 값끼리만 비교합니다.
- 부분집합: `all`, `ko`, `en`, `known_item`, `llm_synth`, `manual`.

해석할 때 주의:

- 질의 40개면 hit@5 1건 차이가 0.025입니다. 방식 간 차이가 몇 건 수준이면 "우위"라고 쓰지 않습니다(CSV로 질의별 승패를 확인).
- known_item 질의는 초록을 바꿔 말한 것이고, lexical은 모든 청크의 tsvector에 제목·초록을 포함하므로 lexical에 유리할 수 있습니다. 본문 검색 능력은 `llm_synth`·청크 단위 표로 봅니다.
- lexical은 FTS 설정이 `english`라 `ko` 질의에서 낮게 나오는 것이 예상된 결과입니다. `ko`/`en` 행을 나눠 보고합니다.
- LLM이 만든 질의는 생성 모델의 표현 습관을 따릅니다. 가능하면 manual 질의를 일부 섞고, 부분집합별 결과를 같이 봅니다.

## Ablation (retriever 무수정)

| 이름 | 방식 | 무엇을 끄나 |
| --- | --- | --- |
| `nodiv` | `lexical_nodiv`, `vector_nodiv`, `hybrid_nodiv` | `_apply_paper_diversity`(논문당 2청크 우선). hybrid는 입력 두 목록과 병합 결과 모두에서 끔 |
| `nofilter` | `lexical_nofilter` | `_filter_lexical_candidates`(references/front_matter/참고문헌형·목차형 텍스트 제거) |
| `norerank` | `vector_norerank` | `_rerank_vector_candidates`(섹션 prior, 토큰 겹침, 참고문헌형 감점) |
| `plainrrf` | `hybrid_plainrrf` | hybrid의 적응 가중치·lexical 품질 가중·교차 보너스 → 표준 RRF(k=60)로 대체. 입력은 기본 lexical/vector 결과, diversity는 유지 |

ablation은 `PaperRetriever`의 하위 단계(저장소 조회 → 정규화 → rerank → filter → diversity → 문맥 창)를 같은 순서·같은 후보 수로 다시 조합해 구현합니다.
retriever를 고치지 않고는 끌 수 없는 것:

- lexical/vector **SQL 안의** `content_role`·섹션 가중(`paper_repository.list_chunk_candidates_by_query`, `vector_repository.search_paper_chunks`). `vector_norerank`도 SQL 감점은 남아 있습니다.
- lexical SQL의 `ts_rank_cd` / ILIKE 보너스 개별 기여, 후보 수(`max(k*3, 10)`) 변경.
- hybrid 적응 가중치와 품질 가중을 **따로** 끄기(`plainrrf`는 둘을 함께 끕니다).
- `toc` 청크 제외(두 SQL 모두 WHERE 절에서 제외).

## 결과 반영 체크리스트

측정을 실제로 돌린 뒤에만 진행합니다. 추정치나 부분 실행 결과는 쓰지 않습니다.

- [ ] 전제 조건 1~4 충족을 리포트 머리말(코퍼스 규모, 본문 source 분포, 임베딩 수, 커밋)로 확인
- [ ] `errors` 열이 모두 0 (아니면 원인 해결 후 재실행)
- [ ] `eval/results/<timestamp>.md`, `.csv`, `_summary.csv`와 사용한 `eval/queries.jsonl`을 함께 커밋 (리포트의 sha256과 파일 일치)
- [ ] README "Evaluation" 표의 "측정 예정"을 리포트의 "README 붙여넣기용" 표로 교체하고, 질의 수(ko/en, 출처별)·측정일·커밋·결과 파일 경로를 표 아래에 적기
- [ ] ko/en 차이, known_item vs llm_synth 차이, noise@10을 한두 문장으로 요약. 차이가 몇 건 수준이면 "차이 없음"으로 서술
- [ ] "hybrid 우위" 같은 문장은 표가 뒷받침할 때만 README·포트폴리오 덱(5p)에 쓰기
- [ ] ablation 결과로 diversity·필터·가중치의 효과를 확인한 경우에만 덱에서 해당 설계를 "효과 있음"으로 서술
