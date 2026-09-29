# G1 검색 골든 v2 변형: 원본과 다른 점

변형 파일은 `g1_retrieval.v2.json.gz`이고 `GOLDEN_IMPL=v2`일 때만 쓰인다. 원본 `g1_retrieval.json.gz`는 옛 구현 기준으로 그대로 둔다.
- 변형 파일은 F06 커밋(2ad7bb8)에서 처음 만들었다.
- 원본을 실제 질의 임베딩으로 다시 캡처한 뒤(60b131d, `query_vectors.source = stored`) 변형도 같은 벡터로 다시 캡처했다.
- 두 번 캡처한 결과가 바이트 단위로 같았다. 555번의 임베딩 요청은 모두 저장된 표에서 응답했다.

다시 만드는 명령(저장소 루트): `GOLDEN_IMPL=v2 v2/.venv/bin/python -m scripts.goldens.capture_g1_retrieval capture --variant v2 --vectors stored`

이식 동등성
- 04e19a7 시점에 v2는 당시 원본 골든(결정적 가짜 질의 벡터)과 351/351이 같았다. 351은 평가 111, 제품 142, 상세 챗 97에 개수 검사 1을 더한 수다.
- 실제 임베딩으로 다시 캡처한 원본(60b131d)과도 비교했다. 현재 v2에서 F06만 되돌리면(`retrieval/fusion.py`를 507c0dd 판으로, lexical 행 초록을 빈 문자열로) 351/351이 같다. 따라서 아래 차이는 모두 F06에서 온다.
- 변형과의 비교는 `GOLDEN_EXACT_SEARCH=1`(인덱스 스캔 끔)에서도 351/351이다.

## F05: 참고문헌 제목 판정을 Python·SQL 한 규칙으로

G1 차이는 0건이다. 가짜 벡터로 잡은 원본과 실제 벡터로 잡은 원본 모두 그렇다. 판정이 바뀌는 제목이 코퍼스에 없어서, 질의 벡터와 관계없이 점수가 같다.

- 기준 규칙은 청크 역할을 정하는 `arxplore.parsing.roles.is_references_section_title`이다. 검색의 Python 쪽(lexical 필터, vector rerank 감점)은 이미 이 함수를 쓴다.
- 바뀐 것은 lexical SQL의 structural_adjustment(-0.08) 조건이다.
  - 옛 조건은 `section_title ~* '<정규식>'`이었다. 대소문자를 무시해서 구두점 없는 소문자 로마 숫자(`iv references`)와 로마 숫자 글자로만 된 단어(`Civil References`, `cl Bibliography`)도 번호로 보고 참고문헌 제목으로 판정했다.
  - 새 조건은 `section_title ~ '<정규식>'`이다.
    - 번호 접두어는 대소문자를 구분한다. 대문자 로마 숫자는 구두점이 없어도 되고, 소문자가 섞이면 `.` 또는 `)`가 있어야 한다.
    - 제목 단어만 `[Rr][Ee]…` 꼴로 대소문자를 무시한다.
  - 정규식은 `arxplore.parsing.roles.references_section_title_sql_regex()` 하나다. a02125a부터 roles에서 정의하고 lexical은 import한다. Python `re`와 PostgreSQL에서 같은 뜻으로 읽히고, `is_references_section_title`도 이 정규식으로 판정한다.
- 확인
  - 골든 DB의 서로 다른 섹션 제목 4,855개와 합성 제목을 합친 379,877개에서 Python 함수, 새 정규식(Python `re`), 새 정규식(PostgreSQL `~`)의 판정이 모두 같았다.
  - 같은 제목 중 옛 SQL 판정이 Python 함수와 다른 것은 6,930개였고 모두 합성 제목이다.
  - 골든 DB 제목 중 SQL 판정이 바뀌는 것은 0개다. 그래서 점수와 순위가 바뀌는 청크가 없다.
- 남는 차이는 유니코드 경계뿐이다.
  - Python과 PostgreSQL의 `\s`·`\d` 분류가 다른 문자(줄바꿈 없는 공백 U+00A0, 비ASCII 숫자 등)를 쓴 제목은 판정이 갈릴 수 있다. 코퍼스에는 없다.
  - 대소문자 접기 차이(İ, ı, ſ)는 roles가 같은 정규식으로 판정하면서 없어졌다.

## F06: hybrid 병합 항목의 필드 누락

옛 동작
- lexical 행은 SQL이 초록을 가져오면서도 결과에 싣지 않아 `paper_abstract`가 늘 빈 문자열이었다.
- hybrid 병합 항목은 처음 본 채널 행의 사본이다.
  - lexical이 먼저 돌므로 두 채널에 모두 나온 청크는 lexical 행을 따른다. 그래서 초록이 비고 vector 점수(`raw_similarity_score`, `rerank_adjustment`)가 빠졌다.
  - vector에만 나온 청크만 vector 점수와 초록을 가졌다.

새 동작
- lexical 행이 초록을 싣는다.
- 병합 항목은 채널별 필드를 모두 가진다. lexical에 나왔으면 lexical 점수 내역, vector에 나왔으면 vector 점수 내역을 갖고, 초록은 비지 않은 쪽을 쓴다.
- 점수, 순위, snippet, 문맥은 바뀌지 않는다.

G1 차이: 351개 항목 중 51개(평가 15, 제품 15, 상세 챗 21)
- 모두 hybrid 결과의 키 정보다. 평가·제품은 `hybrid`, 상세 챗은 `hybrid.contexts`다.
- hit 목록(chunk_id, arxiv_id, 점수, 섹션, 역할, 문맥 해시)과 상세 챗 출처는 원본과 같다.

| 필드 | 평가 | 제품 | 상세 챗 | 바뀐 내용 |
| --- | --- | --- | --- | --- |
| `uniform_keys` | 15 | 15 | 16 | False → True. 두 채널에 모두 나온 hit도 vector 키를 가지면서 모든 hit의 키가 같아졌다 |
| `keys` | 0 | 0 | 5 | 17개 → 19개. `raw_similarity_score`, `rerank_adjustment`가 생겼다. 이전에는 hit이 모두 lexical에서 먼저 나와 이 키가 한 번도 없었다 |

`uniform_keys`가 여전히 False인 항목은 평가 8, 제품 11, 상세 챗 1개다(원본은 23, 26, 17). lexical에만 나온 hit은 vector 점수가 없어서 다른 hit과 키가 다르다.

hybrid hit의 채널 분포(F06 뒤 결과 기준):

| 그룹 | hit | lexical만 | 두 채널 | vector만 |
| --- | --- | --- | --- | --- |
| 평가(k=10) | 1,110 | 16 | 27 | 1,067 |
| 제품(limit 5) | 695 | 14 | 28 | 653 |
| 상세 챗(논문 범위, limit 5) | 485 | 2 | 57 | 426 |

- 두 채널에 모두 나온 hit 112개가 vector 점수를 새로 갖는다.
- 초록은 G1에 기록되지 않는다.
  - F06 전에는 lexical에서 먼저 나온 hit 144개(평가 43, 제품 42, 상세 챗 59)의 초록이 비어 있었다.
  - F06 뒤에는 세 그룹의 hybrid hit 2,290개가 모두 초록을 가진다(골든 DB에 초록이 빈 논문은 없다).
- 가짜 벡터로 잡았던 이전 변형에서는 전역 hybrid 결과에 두 채널에 모두 나온 hit이 없어서 차이가 상세 챗 18개뿐이었다. 실제 벡터에서는 평가·제품에서도 두 채널이 겹친다.
