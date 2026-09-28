# G1 검색 골든 v2 변형: 원본과 다른 점

변형 파일은 `g1_retrieval.v2.json.gz`이고 `GOLDEN_IMPL=v2`일 때만 쓰인다. 변형 파일이 없으면 v2도 원본 `g1_retrieval.json.gz`와 비교한다. 원본은 옛 구현 기준으로 그대로 둔다.

다시 만드는 명령(저장소 루트): `GOLDEN_IMPL=v2 v2/.venv/bin/python -m scripts.goldens.capture_g1_retrieval capture --variant v2 --vectors deterministic`

이식 동등성: 04e19a7 시점에 v2는 원본 골든과 351/351(평가 111, 제품 142, 상세 챗 97, 개수 검사 1)이 같았다. `GOLDEN_EXACT_SEARCH=1`(인덱스 스캔 끔)에서도 351/351이다.

## F05: 참고문헌 제목 판정을 Python·SQL 한 규칙으로

G1 차이는 0건이다. 이 커밋에서는 변형 파일을 만들지 않았다.

- 기준 규칙은 청크 역할을 정하는 `arxplore.parsing.roles.is_references_section_title`이다. 검색의 Python 쪽(lexical 필터, vector rerank 감점)은 이미 이 함수를 쓴다.
- 바뀐 것은 lexical SQL의 structural_adjustment(-0.08) 조건이다.
  - 옛 조건은 `section_title ~* '<정규식>'`이었다. 대소문자를 무시해서 구두점 없는 소문자 로마 숫자(`iv references`)와 로마 숫자 글자로만 된 단어(`Civil References`, `cl Bibliography`)도 번호로 보고 참고문헌 제목으로 판정했다.
  - 새 조건은 `section_title ~ '<정규식>'`이다. 번호 접두어는 대소문자를 구분하고(대문자 로마 숫자는 구두점 선택, 소문자가 섞이면 `.` 또는 `)` 필수), 제목 단어만 `[Rr][Ee]…` 꼴로 대소문자를 무시한다. 정규식은 `lexical.REFERENCES_SECTION_TITLE_SQL_REGEX` 하나이고 Python `re`와 PostgreSQL에서 같은 뜻으로 읽힌다.
- 확인
  - 골든 DB의 서로 다른 섹션 제목 4,855개와 합성 제목을 합친 379,877개에서 Python 함수, 새 정규식(Python `re`), 새 정규식(PostgreSQL `~`)의 판정이 모두 같았다.
  - 같은 제목 중 옛 SQL 판정이 Python 함수와 다른 것은 6,930개였고 모두 합성 제목이다.
  - 골든 DB 제목 중 SQL 판정이 바뀌는 것은 0개다. 그래서 점수와 순위가 바뀌는 청크가 없다.
- 남는 차이는 유니코드 경계뿐이다. Python의 `str.split()`·`re.IGNORECASE`와 PostgreSQL의 `\s`·`\d` 분류가 다른 문자(줄바꿈 없는 공백 U+00A0, 비ASCII 숫자, 점 없는 i 등)를 쓴 제목은 판정이 갈릴 수 있다. 코퍼스에는 없다.
