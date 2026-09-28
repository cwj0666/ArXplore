# G1 검색 골든 v2 변형: 원본과 다른 점

변형 파일은 `g1_retrieval.v2.json.gz`이고 `GOLDEN_IMPL=v2`일 때만 쓰인다. 원본 `g1_retrieval.json.gz`는 옛 구현 기준으로 그대로 둔다. 변형 파일은 F06 커밋에서 처음 만들었다.

다시 만드는 명령(저장소 루트): `GOLDEN_IMPL=v2 v2/.venv/bin/python -m scripts.goldens.capture_g1_retrieval capture --variant v2 --vectors deterministic`

이식 동등성: 04e19a7 시점에 v2는 원본 골든과 351/351(평가 111, 제품 142, 상세 챗 97, 개수 검사 1)이 같았다. `GOLDEN_EXACT_SEARCH=1`(인덱스 스캔 끔)에서도 351/351이다.

## F05: 참고문헌 제목 판정을 Python·SQL 한 규칙으로

G1 차이는 0건이다. 그래서 F05 커밋에는 변형 파일이 없다.

- 기준 규칙은 청크 역할을 정하는 `arxplore.parsing.roles.is_references_section_title`이다. 검색의 Python 쪽(lexical 필터, vector rerank 감점)은 이미 이 함수를 쓴다.
- 바뀐 것은 lexical SQL의 structural_adjustment(-0.08) 조건이다.
  - 옛 조건은 `section_title ~* '<정규식>'`이었다. 대소문자를 무시해서 구두점 없는 소문자 로마 숫자(`iv references`)와 로마 숫자 글자로만 된 단어(`Civil References`, `cl Bibliography`)도 번호로 보고 참고문헌 제목으로 판정했다.
  - 새 조건은 `section_title ~ '<정규식>'`이다. 번호 접두어는 대소문자를 구분하고(대문자 로마 숫자는 구두점 선택, 소문자가 섞이면 `.` 또는 `)` 필수), 제목 단어만 `[Rr][Ee]…` 꼴로 대소문자를 무시한다. 정규식은 `lexical.REFERENCES_SECTION_TITLE_SQL_REGEX` 하나이고 Python `re`와 PostgreSQL에서 같은 뜻으로 읽힌다.
- 확인
  - 골든 DB의 서로 다른 섹션 제목 4,855개와 합성 제목을 합친 379,877개에서 Python 함수, 새 정규식(Python `re`), 새 정규식(PostgreSQL `~`)의 판정이 모두 같았다.
  - 같은 제목 중 옛 SQL 판정이 Python 함수와 다른 것은 6,930개였고 모두 합성 제목이다.
  - 골든 DB 제목 중 SQL 판정이 바뀌는 것은 0개다. 그래서 점수와 순위가 바뀌는 청크가 없다.
- 남는 차이는 유니코드 경계뿐이다. Python의 `str.split()`·`re.IGNORECASE`와 PostgreSQL의 `\s`·`\d` 분류가 다른 문자(줄바꿈 없는 공백 U+00A0, 비ASCII 숫자, 점 없는 i 등)를 쓴 제목은 판정이 갈릴 수 있다. 코퍼스에는 없다.

## F06: hybrid 병합 항목의 필드 누락

옛 동작
- lexical 행은 SQL이 초록을 가져오면서도 결과에 싣지 않아 `paper_abstract`가 늘 빈 문자열이었다.
- hybrid 병합 항목은 처음 본 채널 행의 사본이다. lexical이 먼저 돌므로 두 채널에 모두 나온 청크는 lexical 행을 따라 초록이 비고 vector 점수(`raw_similarity_score`, `rerank_adjustment`)가 빠졌다. vector에만 나온 청크만 vector 점수와 초록을 가졌다.

새 동작
- lexical 행이 초록을 싣는다.
- 병합 항목은 채널별 필드를 모두 가진다. lexical에 나왔으면 lexical 점수 내역, vector에 나왔으면 vector 점수 내역을 갖고, 초록은 비지 않은 쪽을 쓴다.
- 점수, 순위, snippet, 문맥은 바뀌지 않는다.

G1 차이: 351개 항목 중 상세 챗 18개. 모두 hybrid 모드의 `contexts` 키 정보다.

| 필드 | 항목 수 | 바뀐 내용 |
| --- | --- | --- |
| `keys` | 6 | 17개 → 19개. `raw_similarity_score`, `rerank_adjustment`가 생겼다. 이전에는 hit이 모두 lexical에서 먼저 나와 이 키가 한 번도 없었다 |
| `uniform_keys` | 12 | False → True. 두 채널에 모두 나온 hit도 vector 키를 가지면서 모든 hit의 키가 같아졌다 |

- 평가·제품 그룹은 바뀌지 않았다. G1의 질의 벡터가 결정적 가짜 벡터라서 전역 hybrid 상위 결과에는 두 채널에 모두 나온 hit이 없다(평가 1,110개 hit 중 lexical만 38, vector만 1,072, 제품 695개 중 lexical만 27, vector만 668). 그래서 키 집합과 균일 여부가 그대로다.
- 상세 챗 hybrid hit 485개 중 52개가 두 채널에 모두 나왔고, 이 hit들이 vector 점수를 새로 갖는다.
- 초록은 G1에 기록되지 않는다. F06 뒤에는 세 그룹의 hybrid hit 2,290개가 모두 초록을 가진다(골든 DB에 초록이 빈 논문은 없다). F06 전에는 lexical에서 먼저 나온 hit의 초록이 모두 비어 있었다.
