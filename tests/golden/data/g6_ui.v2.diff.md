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
