import json
import shutil

import pytest

from tests.golden import g6_ui
from tests.golden.store import sha256_bytes

MANIFEST = g6_ui.GOLDEN_DIR / "captures.json"
CAPTURES = json.loads(MANIFEST.read_text(encoding="utf-8"))["captures"] if MANIFEST.is_file() else []
NAMES = [capture["name"] for capture in CAPTURES]


@pytest.fixture(scope="module")
def ui_prerequisites(golden_database_url: str) -> str:
    missing = [
        label
        for label, ready in (
            ("node", shutil.which("node") is not None),
            ("tests/golden/ui/node_modules (npm ci)", (g6_ui.UI_DIR / "node_modules" / "playwright").is_dir()),
            ("frontend/dist (npm ci && npm run build)", (g6_ui.FRONTEND_DIST / "index.html").is_file()),
        )
        if not ready
    ]
    if missing:
        pytest.skip(f"UI golden prerequisites missing: {', '.join(missing)}")
    try:
        g6_ui.detail_pdf()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))
    return golden_database_url


def test_manifest_lists_every_screen_with_matching_files():
    assert NAMES == [
        "list_default",
        "list_search",
        "list_no_results",
        "not_found_route",
        "detail_not_found",
        "detail_analysis",
        "summary_model_dialog",
        "summary_result",
        "pdf_split_view",
        "detail_chat_answer",
        "assistant_answer",
    ]
    for capture in CAPTURES:
        stored = {
            "png_sha256": (g6_ui.GOLDEN_DIR / f"{capture['name']}.png").read_bytes(),
            "text_sha256": (g6_ui.GOLDEN_DIR / f"{capture['name']}.txt").read_bytes(),
            "aria_sha256": (g6_ui.GOLDEN_DIR / f"{capture['name']}.aria.yml").read_bytes(),
        }
        assert {key: sha256_bytes(data) for key, data in stored.items()} == {key: capture[key] for key in stored}


def test_fresh_capture_matches_golden(ui_prerequisites, tmp_path):
    g6_ui.capture_ui(tmp_path, database_url=ui_prerequisites)

    problems = g6_ui.compare_dirs(g6_ui.GOLDEN_DIR, tmp_path)
    assert not problems, "\n".join(problems)
