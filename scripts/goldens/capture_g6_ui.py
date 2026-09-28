import argparse
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path

from tests.golden import environment, g6_ui


def capture(out: Path) -> int:
    started = time.perf_counter()
    manifest = g6_ui.capture_ui(out, database_url=environment.database_url())
    size = sum(path.stat().st_size for path in out.iterdir() if path.is_file())
    print(
        f"captured {len(manifest['captures'])} screens in {time.perf_counter() - started:.0f}s -> {out} ({size} bytes)"
    )
    return 0


def compare(expected: Path) -> int:
    with tempfile.TemporaryDirectory(prefix="golden-g6-compare-") as workdir:
        actual = Path(workdir)
        g6_ui.capture_ui(actual, database_url=environment.database_url())
        problems = g6_ui.compare_dirs(expected, actual)
    for problem in problems:
        print(problem)
    print("identical" if not problems else f"{len(problems)} difference(s)")
    return 1 if problems else 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m scripts.goldens.capture_g6_ui")
    parser.add_argument("--out", type=Path, default=g6_ui.GOLDEN_DIR)
    parser.add_argument("--compare", action="store_true", help="capture into a temporary directory and compare")
    args = parser.parse_args(argv)
    return compare(args.out) if args.compare else capture(args.out)


if __name__ == "__main__":
    raise SystemExit(main())
