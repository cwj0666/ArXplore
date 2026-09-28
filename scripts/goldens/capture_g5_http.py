import argparse
import time
from collections.abc import Sequence
from pathlib import Path

from tests.golden import environment, g5_http, g5_runner
from tests.golden.store import write_golden


def capture(out: Path, bundles: Sequence[str]) -> int:
    started = time.perf_counter()
    names = list(bundles) or sorted(g5_http.BUNDLES)
    results = g5_runner.run_bundles(names, database_url=environment.database_url())
    golden = {
        "format": g5_http.FORMAT,
        "captured_with": next(iter(results.values()))["captured_with"],
        "bundles": {name: result["scenarios"] for name, result in results.items()},
    }
    written = write_golden(out, golden)
    scenarios = sum(len(result["scenarios"]) for result in results.values())
    responses = sum(
        len(scenario["responses"]) for result in results.values() for scenario in result["scenarios"].values()
    )
    print(
        f"captured {len(names)} bundles, {scenarios} scenarios, {responses} responses"
        f" in {time.perf_counter() - started:.0f}s -> {out} ({len(written)} bytes)"
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m scripts.goldens.capture_g5_http")
    parser.add_argument("--out", type=Path, default=g5_http.GOLDEN_PATH)
    parser.add_argument("--bundle", action="append", default=[], choices=sorted(g5_http.BUNDLES))
    args = parser.parse_args(argv)
    return capture(args.out, args.bundle)


if __name__ == "__main__":
    raise SystemExit(main())
