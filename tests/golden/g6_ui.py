import json
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from tests.fakes.openai_server import FakeOpenAIServer, default_chat_handler
from tests.golden import environment, g3_pypdf, g5_http, g5_runner
from tests.golden.store import DATA_DIR, GOLDEN_ROOT

GOLDEN_DIR = DATA_DIR / "g6_ui"
UI_DIR = GOLDEN_ROOT / "ui"
CAPTURE_SCRIPT = UI_DIR / "capture.mjs"
REPO_ROOT = GOLDEN_ROOT.parents[1]
FRONTEND_DIST = REPO_ROOT / "frontend" / "dist"
DETAIL_PAPER = "2609.27980"
MISSING_PAPER = "2609.99999"
SERVER_START_SECONDS = 90
CAPTURE_TIMEOUT_SECONDS = 1200
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


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def wait_until_ready(url: str, process: subprocess.Popen[bytes], log_path: Path) -> None:
    deadline = time.monotonic() + SERVER_START_SECONDS
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Django exited early:\n{log_path.read_text(errors='replace')[-3000:]}")
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return
        except OSError:
            time.sleep(0.5)
    raise RuntimeError(f"Django did not answer {url}:\n{log_path.read_text(errors='replace')[-3000:]}")


def detail_pdf() -> Path:
    entry = next(entry for entry in g3_pypdf.load_manifest() if entry["arxiv_id"] == DETAIL_PAPER)
    path = g3_pypdf.pdf_path(entry)
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing; run python -m scripts.goldens.capture_g3_pypdf fetch")
    return path


def capture_ui(out_dir: Path, *, database_url: str) -> dict[str, Any]:
    if not (FRONTEND_DIST / "index.html").is_file():
        raise FileNotFoundError(f"{FRONTEND_DIST} is missing; build the frontend first (npm ci && npm run build)")
    pdf = detail_pdf()
    out_dir.mkdir(parents=True, exist_ok=True)
    with (
        tempfile.TemporaryDirectory(prefix="golden-g6-") as workdir,
        environment.database_copy(database_url, "golden_g6_ui") as copy_url,
        FakeOpenAIServer(chat_handler=ui_reply) as server,
    ):
        port = free_port()
        values = {
            **g5_runner.runner_environ(),
            **g5_runner.bundle_app_environ(server, copy_url),
            **environment.LEGACY_DJANGO_ENV,
            "RATE_LIMIT_ENABLED": "false",
        }
        log_path = Path(workdir) / "django.log"
        with log_path.open("wb") as log:
            django = subprocess.Popen(
                [sys.executable, str(REPO_ROOT / "backend" / "manage.py"), "runserver", f"127.0.0.1:{port}"]
                + ["--insecure", "--noreload"],
                cwd=workdir,
                env=values,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            try:
                base_url = f"http://127.0.0.1:{port}"
                wait_until_ready(f"{base_url}/bootstrap.json", django, log_path)
                subprocess.run(
                    ["node", str(CAPTURE_SCRIPT), "--base-url", base_url, "--out", str(out_dir), "--pdf", str(pdf)]
                    + ["--paper", DETAIL_PAPER, "--missing-paper", MISSING_PAPER],
                    cwd=UI_DIR,
                    check=True,
                    timeout=CAPTURE_TIMEOUT_SECONDS,
                )
            finally:
                django.terminate()
                django.wait(timeout=30)
        errors = server.errors
    if errors:
        raise RuntimeError(f"fake server errors during the UI capture: {errors[:3]}")
    return json.loads((out_dir / "captures.json").read_text(encoding="utf-8"))


def compare_dirs(expected: Path, actual: Path) -> list[str]:
    problems: list[str] = []
    expected_files = sorted(path.name for path in expected.iterdir() if path.is_file())
    actual_files = sorted(path.name for path in actual.iterdir() if path.is_file())
    if expected_files != actual_files:
        problems.append(f"file sets differ: {sorted(set(expected_files) ^ set(actual_files))}")
    for name in sorted(set(expected_files) & set(actual_files)):
        if (expected / name).read_bytes() != (actual / name).read_bytes():
            problems.append(f"{name} differs")
    return problems
