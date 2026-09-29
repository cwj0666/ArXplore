from collections.abc import Mapping
from pathlib import Path
from typing import Any

from tests.fakes.openai_server import default_chat_handler
from tests.golden import g3_pypdf, g5_http
from tests.golden.store import DATA_DIR, GOLDEN_ROOT

GOLDEN_DIR = DATA_DIR / "g6_ui"
UI_DIR = GOLDEN_ROOT / "ui"
CAPTURE_SCRIPT = UI_DIR / "capture.mjs"
DETAIL_PAPER = "2609.27980"
MISSING_PAPER = "2609.99999"
PAPER_CHAT_REPLY = (
    "이 논문은 Whisper 인코더의 층을 줄이고 레이블 없이 성능을 회복하는 방법을 제안합니다 [1].\n\n"
    "- 인코더 층을 여섯 개 줄여도 인식 품질을 유지합니다 [2].\n"
    "- 회복 단계는 레이블이 없는 음성만 사용합니다 [1, 3]."
)
SUMMARY_REPLY = "\n".join(
    [
        "## 문제 정의",
        "음성 인식 모델의 인코더가 커서 추론 비용이 높다는 문제를 다룹니다.",
        "## 접근 방법",
        "인코더 층을 줄인 뒤 레이블 없는 데이터로 성능을 회복합니다.",
        "## 실험 및 결과",
        "여러 벤치마크에서 원래 모델과 비슷한 오류율을 보고합니다.",
        "## 한계",
        "특정 언어와 도메인에서만 검증했습니다.",
        "## 핵심 가치",
        "적은 비용으로 음성 인식 모델을 가볍게 만드는 방법을 보여 줍니다.",
    ]
)


def ui_reply(request: Mapping[str, Any]) -> Any:
    if request.get("tools"):
        return g5_http.reply_search_then_link(request)
    system = g5_http.system_text(request)
    if "연구 보조자" in system:
        return PAPER_CHAT_REPLY
    if "핵심 포인트" in system:
        return g5_http.KEY_FINDINGS_REPLY
    if "브리핑" in system:
        return g5_http.OVERVIEW_REPLY
    if "깊이 있게 분석" in system:
        return SUMMARY_REPLY
    return default_chat_handler(request)


def detail_pdf() -> Path:
    entry = next(entry for entry in g3_pypdf.load_manifest() if entry["arxiv_id"] == DETAIL_PAPER)
    path = g3_pypdf.pdf_path(entry)
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing; run python -m scripts.goldens.capture_g3_pypdf fetch")
    return path
