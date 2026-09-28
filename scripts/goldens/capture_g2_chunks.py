import argparse
import time
from collections.abc import Sequence
from pathlib import Path

from tests.golden import environment, g2_chunks
from tests.golden.adapters import load_parser_adapter
from tests.golden.store import variant_path, write_golden


def capture(out: Path) -> int:
    url = environment.database_url()
    started = time.perf_counter()
    with environment.loopback_only():
        inputs = g2_chunks.load_inputs(url)
        database_chunks = g2_chunks.load_database_chunks(url)
        golden = g2_chunks.build_golden(load_parser_adapter(), inputs)
    papers = golden["papers"]
    same_papers = [arxiv_id for arxiv_id, entry in papers.items() if entry["chunks"] == database_chunks.get(arxiv_id)]
    total_chunks = sum(len(chunks) for chunks in database_chunks.values())
    same_chunks = sum(
        sum(1 for left, right in zip(entry["chunks"], database_chunks.get(arxiv_id, []), strict=False) if left == right)
        for arxiv_id, entry in papers.items()
    )
    written = write_golden(out, golden)
    print(
        f"papers {len(same_papers)}/{len(papers)} and chunks {same_chunks}/{total_chunks} match paper_chunks;"
        f" {time.perf_counter() - started:.1f}s -> {out} ({len(written)} bytes)"
    )
    return 0 if len(same_papers) == len(papers) and same_chunks == total_chunks else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m scripts.goldens.capture_g2_chunks")
    parser.add_argument("--out", type=Path, default=g2_chunks.GOLDEN_PATH)
    parser.add_argument("--variant", help="write <name>.<variant>.json.gz next to the original golden")
    args = parser.parse_args(argv)
    return capture(variant_path(args.out, args.variant) if args.variant else args.out)


if __name__ == "__main__":
    raise SystemExit(main())
