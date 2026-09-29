import shutil
from pathlib import Path

import pytest

from tests.golden import g6_ui
from tests.golden.ui import g6_v2


@pytest.fixture(scope="module")
def v2_results(golden_database_url: str, tmp_path_factory: pytest.TempPathFactory) -> dict[str, g6_v2.ScreenResult]:
    missing = [
        label
        for label, ready in (
            ("node", shutil.which("node") is not None),
            ("tests/golden/ui/node_modules (npm ci)", (g6_ui.UI_DIR / "node_modules" / "playwright").is_dir()),
            ("v2/frontend/dist (npm ci && npm run build)", (g6_v2.V2_DIST / "index.html").is_file()),
            (f"{g6_v2.APP_COMMAND.name} in the project environment", g6_v2.APP_COMMAND.is_file()),
        )
        if not ready
    ]
    if missing:
        pytest.skip(f"v2 UI golden prerequisites missing: {', '.join(missing)}")
    try:
        g6_ui.detail_pdf()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))
    out: Path = tmp_path_factory.mktemp("g6-v2")
    g6_v2.capture_v2(out)
    return {result.name: result for result in g6_v2.compare(out)}


@pytest.mark.parametrize("name", g6_v2.screen_names())
def test_v2_screen_matches_golden(v2_results: dict[str, g6_v2.ScreenResult], name: str) -> None:
    result = v2_results[name]
    assert result.text_equal, f"{name}: innerText differs"
    assert result.aria_equal, f"{name}: ARIA snapshot differs"
    assert result.pixels == 0, f"{name}: {result.pixels} pixels differ ({result.detail})"
