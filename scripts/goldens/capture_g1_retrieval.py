import argparse
import csv
import gzip
import os
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from tests.fakes.openai_server import EmbeddingVectors, FakeOpenAIServer, embedding_key, write_embedding_table
from tests.golden import environment, g1_retrieval
from tests.golden.adapters import load_retrieval_adapter
from tests.golden.store import read_golden, sha256_bytes, variant_path, write_golden

REAL_API_BASE = "https://api.openai.com/v1"
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIMENSIONS = 1536
EMBEDDING_BATCH = 64
EVAL_CSV = g1_retrieval.REPO_ROOT / "eval" / "results" / "20260929-005408.csv"


def _apply_environment(server: FakeOpenAIServer, *, exact: bool) -> None:
    options = environment.READ_ONLY_OPTION
    if exact:
        options = f"{options} {environment.EXACT_SEARCH_OPTION}"
    os.environ.update(
        environment.harness_environ(
            openai_environ=server.environ(), database_url=environment.database_url(), pg_options=options
        )
    )


def _progress(group: str, key: str) -> None:
    print(f"\r{group:<10} {key[:40]:<40}", end="", file=sys.stderr, flush=True)


def _run(server: FakeOpenAIServer, sets: dict[str, Any], *, exact: bool) -> dict[str, Any]:
    _apply_environment(server, exact=exact)
    adapter = load_retrieval_adapter()
    try:
        return g1_retrieval.build_golden(adapter, server, sets, progress=_progress)
    finally:
        adapter.close()
        print(file=sys.stderr)


def table_digest() -> str:
    return sha256_bytes(gzip.decompress(g1_retrieval.EMBEDDING_TABLE_PATH.read_bytes()))


def embed() -> int:
    from dotenv import dotenv_values

    sets = g1_retrieval.build_query_sets()
    recorder = EmbeddingVectors()
    with environment.loopback_only(), FakeOpenAIServer(embedding_handler=recorder) as server:
        _run(server, sets, exact=False)
        if server.errors:
            raise SystemExit(f"fake server errors during the dry run: {server.errors[:3]}")
    texts = list(dict.fromkeys(recorder.inputs))
    if not all(isinstance(text, str) and text for text in texts):
        raise SystemExit("the dry run sent a non-string or empty embedding input")
    print(f"dry run sent {len(recorder.inputs)} embedding inputs, {len(texts)} unique")

    import openai

    api_key = str(dotenv_values(g1_retrieval.REPO_ROOT / ".env").get("OPENAI_API_KEY") or "").strip()
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is missing from .env")
    client = openai.OpenAI(api_key=api_key, base_url=REAL_API_BASE, max_retries=2)
    vectors: dict[str, list[float]] = {}
    calls = 0
    tokens = 0
    for start in range(0, len(texts), EMBEDDING_BATCH):
        batch = texts[start : start + EMBEDDING_BATCH]
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch, dimensions=EMBEDDING_DIMENSIONS)
        calls += 1
        tokens += response.usage.total_tokens
        ordered = sorted(response.data, key=lambda item: item.index)
        if len(ordered) != len(batch):
            raise SystemExit(f"batch {calls} returned {len(ordered)} vectors for {len(batch)} inputs")
        for text, item in zip(batch, ordered, strict=True):
            vectors[embedding_key(text)] = list(item.embedding)
    write_embedding_table(
        g1_retrieval.EMBEDDING_TABLE_PATH, vectors, model=EMBEDDING_MODEL, dimensions=EMBEDDING_DIMENSIONS
    )
    print(
        f"embedded {len(texts)} strings in {calls} API calls ({tokens} tokens) -> {g1_retrieval.EMBEDDING_TABLE_PATH}"
    )
    return 0


def capture(out: Path, *, exact: bool, vectors_source: str) -> int:
    sets = g1_retrieval.build_query_sets()
    if vectors_source == "auto":
        vectors_source = "stored" if g1_retrieval.EMBEDDING_TABLE_PATH.is_file() else "deterministic"
    query_vectors = {"source": vectors_source}
    if vectors_source == "stored":
        query_vectors["table_sha256"] = table_digest()
    vectors = g1_retrieval.embedding_handler(query_vectors)
    started = time.perf_counter()
    with environment.loopback_only(), FakeOpenAIServer(embedding_handler=vectors) as server:
        golden = _run(server, sets, exact=exact)
        errors = server.errors
    if errors:
        raise SystemExit(f"fake server errors during capture: {errors[:3]}")
    golden["query_vectors"] = query_vectors
    written = write_golden(out, golden)
    counts = {group: len(golden[group]) for group in g1_retrieval.GROUPS}
    print(
        f"captured {counts} with {vectors_source} query vectors ({vectors.table_hits} stored hits)"
        f" in {time.perf_counter() - started:.0f}s -> {out} ({len(written)} bytes)"
    )
    return 0


def compare_eval_csv(golden_path: Path) -> int:
    golden = read_golden(golden_path)
    totals: dict[str, list[int]] = {}
    mismatches: dict[str, list[str]] = {}
    with EVAL_CSV.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            method = row["method"]
            if method not in g1_retrieval.EVAL_METHODS:
                continue
            entry = golden["eval"].get(row["query_id"])
            counts = totals.setdefault(method, [0, 0, 0, 0])
            counts[0] += 1
            if entry is None:
                mismatches.setdefault(method, []).append(f"{row['query_id']}: not in golden")
                continue
            hits = entry["results"][method]["hits"]
            chunk_ids = [str(hit["chunk_id"]) for hit in hits]
            expected_ids = row["retrieved_chunk_ids"].split()
            if chunk_ids == expected_ids:
                counts[1] += 1
            else:
                mismatches.setdefault(method, []).append(row["query_id"])
            if set(chunk_ids) == set(expected_ids):
                counts[2] += 1
            if [hit["arxiv_id"] for hit in hits] == row["retrieved_arxiv_ids"].split():
                counts[3] += 1
    for method in g1_retrieval.EVAL_METHODS:
        total, exact_order, same_set, same_papers = totals.get(method, [0, 0, 0, 0])
        print(
            f"{method:<8} queries={total} same_chunk_order={exact_order} ({exact_order / max(1, total):.1%})"
            f" same_chunk_set={same_set} same_paper_order={same_papers}"
        )
        if mismatches.get(method):
            print(f"         differing: {', '.join(mismatches[method][:12])}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m scripts.goldens.capture_g1_retrieval")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("embed", help="dry-run every G1 path, then embed the exact inputs once with the real API")
    capture_parser = commands.add_parser("capture", help="run every G1 path against the stored query embeddings")
    capture_parser.add_argument("--out", type=Path, default=g1_retrieval.GOLDEN_PATH)
    capture_parser.add_argument("--variant", help="write <name>.<variant>.json.gz next to the original golden")
    capture_parser.add_argument("--exact", action="store_true", help="disable index scans (exact vector search)")
    capture_parser.add_argument(
        "--vectors",
        choices=("auto", "stored", "deterministic"),
        default="auto",
        help="stored = real query embeddings from the table, deterministic = fake sha256 vectors",
    )
    compare_parser = commands.add_parser("compare-csv", help="compare eval rankings with the 20260929-005408 run")
    compare_parser.add_argument("--golden", type=Path, default=g1_retrieval.GOLDEN_PATH)
    args = parser.parse_args(argv)
    if args.command == "embed":
        return embed()
    if args.command == "capture":
        out = variant_path(args.out, args.variant) if args.variant else args.out
        return capture(out, exact=args.exact, vectors_source=args.vectors)
    return compare_eval_csv(args.golden)


if __name__ == "__main__":
    raise SystemExit(main())
