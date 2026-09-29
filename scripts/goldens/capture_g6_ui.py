import argparse
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path

from tests.golden.ui import g6_v2


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m scripts.goldens.capture_g6_ui")
    parser.add_argument("--out", type=Path, help="keep the capture in this directory")
    parser.add_argument("--reuse", action="store_true", help="compare an existing capture in --out without capturing")
    parser.add_argument("--write-variant", nargs="+", metavar="SCREEN", default=[], help="store screens as v2 variants")
    args = parser.parse_args(argv)
    with tempfile.TemporaryDirectory(prefix="v2-g6-capture-") as scratch:
        out_dir = args.out or Path(scratch)
        if not args.reuse:
            started = time.perf_counter()
            g6_v2.capture_v2(out_dir)
            print(f"captured in {time.perf_counter() - started:.0f}s -> {out_dir}")
        if args.write_variant:
            unknown = sorted(set(args.write_variant) - set(g6_v2.screen_names()))
            if unknown:
                parser.error(f"unknown screens: {unknown}")
            g6_v2.write_variant(out_dir, args.write_variant)
        results = g6_v2.compare(out_dir)
    for result in results:
        print(result.describe())
    mismatched = [result for result in results if not result.identical]
    print("identical" if not mismatched else f"{len(mismatched)} screen(s) differ")
    return 1 if mismatched else 0


if __name__ == "__main__":
    raise SystemExit(main())
