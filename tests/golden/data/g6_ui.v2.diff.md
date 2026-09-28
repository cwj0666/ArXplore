# G6 화면 골든 v2 변형: 원본과 다른 점

변형 파일은 `g6_ui.v2/<화면>.{txt,aria.yml,png}`이고 v2 프론트(`v2/frontend`) 캡처와 비교할 때만 쓰인다. 원본 `g6_ui/`는 옛 프론트 기준으로 그대로 둔다. 변형이 없는 화면은 원본과 비교한다.

다시 만드는 명령(저장소 루트, 먼저 `cd v2/frontend && npm ci && npm run build`)
- 비교: `.venv/bin/python -m tests.golden.ui.g6_v2` 또는 `.venv/bin/python -m pytest tests/golden/ui/test_g6_ui_v2.py`
- 변형 저장: `.venv/bin/python -m tests.golden.ui.g6_v2 --out <폴더> --write-variant <화면...>`. 원본과 바이트가 다른 파일만 저장하고, 비교할 때는 파일마다 변형이 있으면 변형을, 없으면 원본을 쓴다.

캡처 방식
- v2 빌드를 `vite preview`로 띄우고 API는 옛 Django 백엔드로 프록시한다. DB는 `arxplore_tpl`에서 복사한 `v2_test_ui_g6`, LLM은 가짜 OpenAI 서버다.
- 캡처 스크립트는 `--impl v2`로 돈다. 클래스 이름에 기대던 선택자 두 개(요약 모델 대화상자의 생성 버튼, PDF 패널 표시 대기)만 역할·id 기반으로 바꾼다.
- 비교 기준: innerText와 ARIA 스냅샷은 바이트가 같아야 하고, PNG는 달라진 픽셀 수가 0이어야 한다.

이식 동등성: ae59715(동등 이식) 시점에 v2는 11개 화면 모두 원본과 innerText, ARIA 스냅샷, PNG 바이트가 같았다.

## F19: 검색 0건 문구

- 화면: `list_no_results`
- 서버 응답의 `query`(서버가 trim한 검색어)가 있는데 결과가 0건이면 "검색 결과가 없습니다."를 보인다. 검색어 없이 0건이면 예전처럼 "수집된 논문이 없습니다."다.
- innerText와 ARIA: `수집된 논문이 없습니다.` → `검색 결과가 없습니다.` 한 줄만 바뀐다.
- PNG: 문구 영역 1,674픽셀(x 640~799, y 341~354)만 다르고 나머지는 같다.

## F29 접근성: 카드 링크, 챗 패널 포커스, 라벨

화면 4개의 ARIA 스냅샷만 바뀌고 innerText와 PNG는 원본과 같다(픽셀 차이 0). 그래서 변형은 `.aria.yml`만 둔다.

- `list_default`, `list_search`: 논문 카드
  - 원본은 `article role=link` 카드 안에 PDF 제목 링크가 들어 있는 중첩 링크였다.
  - 이제 카드는 역할 없는 `article`이고, 그 안에 PDF 링크와 상세 링크가 형제로 있다. 상세 링크는 `/papers/<id>/`로 가는 실제 `<a>`이고 이름은 전과 같은 "{제목} 상세 보기"다. 초록·날짜·추천 수가 이 링크 안에 있다.
  - 상세 링크의 `::before`가 카드 전체를 덮어 카드 아무 곳이나 누르면 상세로 간다. PDF 링크는 그 위에 있어 따로 눌린다.
  - 스냅샷: `link "... 상세 보기"` 아래 PDF 링크가 있던 구조가 `article` 아래 `link "{잘린 제목}"`(PDF)과 `link "... 상세 보기"`(`/url` 포함)로 바뀐다. 카드 순서, 텍스트, `::before`·`::after` 문구는 같다.
- `pdf_split_view`: PDF 분할창 닫기 버튼에 `aria-label="분할창 닫기"`를 붙였다. `button "×"` → `button "분할창 닫기": ×`
- `assistant_answer`: 어시스턴트 입력창에 `aria-label="AI 어시스턴트에게 질문하기"`를 붙였다. 전에는 placeholder가 이름이었다. `textbox "연구 주제, 트렌드, 논문을 질문해 보세요."` → `textbox "AI 어시스턴트에게 질문하기"`와 `/placeholder` 줄
- 챗 패널: 열면 입력창으로, Esc나 ×로 닫으면 여는 버튼으로 포커스가 간다. G6 캡처는 스크린샷 전에 포커스를 풀기 때문에 화면 차이가 없다.

G6 밖에서 옛 앱과 나란히 비교한 결과
- 초록 위 호버 오버레이, 제목 호버 툴팁, 카드 키보드 포커스 링은 픽셀까지 같다. 포커스 링은 상세 링크가 포커스를 받을 때 `:has()`로 카드에 그린다.
- 탭 순서가 바뀐다. 전에는 카드 → 그 안의 PDF 링크 순서였고, 이제 PDF 링크 → 상세 링크 순서다.
- 카드 이동은 이제 브라우저 기본 링크 동작이다. Ctrl/Cmd 클릭과 가운데 클릭이 새 탭으로 열리고, Space 키로는 이동하지 않는다(Enter만 이동).
