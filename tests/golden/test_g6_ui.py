import json

from tests.golden import g6_ui
from tests.golden.store import sha256_bytes

MANIFEST = g6_ui.GOLDEN_DIR / "captures.json"
CAPTURES = json.loads(MANIFEST.read_text(encoding="utf-8"))["captures"] if MANIFEST.is_file() else []
NAMES = [capture["name"] for capture in CAPTURES]


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
