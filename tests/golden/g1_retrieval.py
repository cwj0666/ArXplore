import json
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from typing import Any

from arxplore_eval.dataset import load_queries
from tests.fakes.openai_server import EmbeddingVectors, FakeHTTPError, FakeOpenAIServer, load_embedding_table
from tests.golden.adapters import RetrievalAdapter
from tests.golden.store import DATA_DIR, GOLDEN_ROOT, sha256_text

FORMAT = 1
GOLDEN_PATH = DATA_DIR / "g1_retrieval.json.gz"
EMBEDDING_TABLE_PATH = DATA_DIR / "g1_query_embeddings.json.gz"
REPO_ROOT = GOLDEN_ROOT.parents[1]
QUERIES_PATH = REPO_ROOT / "eval" / "queries.jsonl"
ANSWERS_PATH = REPO_ROOT / "eval" / "results" / "answers_20260926-103836.jsonl"
EVAL_METHODS = ("lexical", "vector", "hybrid")
MODES = ("hybrid", "lexical", "embedding_failure")
EVAL_K = 10
ADJACENCY_WINDOW = 1
PRODUCT_LIMIT = 5
GROUPS = ("eval", "product", "paper_chat")


def _fail_embeddings(request: Mapping[str, Any]) -> list[list[float]]:
    raise FakeHTTPError(401, "simulated embedding failure", error_type="invalid_api_key")


@contextmanager
def embedding_mode(server: FakeOpenAIServer, mode: str) -> Iterator[str]:
    if mode != "embedding_failure":
        yield "lexical" if mode == "lexical" else "hybrid"
        return
    handler = server.embedding_handler
    server.embedding_handler = _fail_embeddings
    try:
        yield "hybrid"
    finally:
        server.embedding_handler = handler


def hit_record(hit: Mapping[str, Any]) -> dict[str, Any]:
    score = hit.get("score")
    record = {
        "chunk_id": hit.get("chunk_id"),
        "arxiv_id": hit.get("arxiv_id"),
        "score": None if score is None else repr(float(score)),
        "section_title": hit.get("section_title"),
        "content_role": hit.get("content_role"),
    }
    if "context_text" in hit:
        record["context_sha256"] = sha256_text(str(hit["context_text"]))
    return record


def hits_record(hits: list[Mapping[str, Any]]) -> dict[str, Any]:
    key_sets = [sorted(hit) for hit in hits]
    return {
        "hits": [hit_record(hit) for hit in hits],
        "keys": sorted({key for keys in key_sets for key in keys}),
        "uniform_keys": all(keys == key_sets[0] for keys in key_sets),
    }


def source_record(source: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "arxiv_id": source.get("arxiv_id"),
        "title": source.get("title"),
        "url": source.get("url"),
        "section_title": source.get("section_title"),
        "chunk_id": source.get("chunk_id"),
        "text_sha256": sha256_text(str(source.get("text") or "")),
    }


def run_eval_entry(adapter: RetrievalAdapter, server: FakeOpenAIServer, entry: Mapping[str, Any]) -> dict[str, Any]:
    return {
        method: hits_record(adapter.eval_search(method, entry["query"], k=EVAL_K, adjacency_window=ADJACENCY_WINDOW))
        for method in EVAL_METHODS
    }


def run_product_entry(adapter: RetrievalAdapter, server: FakeOpenAIServer, entry: Mapping[str, Any]) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for mode in MODES:
        with embedding_mode(server, mode) as retrieval_mode:
            contexts, used = adapter.retrieve_contexts(
                entry["query"], limit=PRODUCT_LIMIT, arxiv_id=None, retrieval_mode=retrieval_mode
            )
        results[mode] = {"retrieval_mode": used, **hits_record(contexts)}
    return results


def run_paper_chat_entry(
    adapter: RetrievalAdapter, server: FakeOpenAIServer, entry: Mapping[str, Any]
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for mode in MODES:
        with embedding_mode(server, mode) as retrieval_mode:
            contexts, used = adapter.retrieve_contexts(
                entry["query"], limit=PRODUCT_LIMIT, arxiv_id=entry["arxiv_id"], retrieval_mode=retrieval_mode
            )
            chat = adapter.paper_chat_sources(entry["arxiv_id"], entry["query"], retrieval_mode=retrieval_mode)
        sources, sources_mode = chat if chat is not None else ([], None)
        results[mode] = {
            "contexts": {"retrieval_mode": used, **hits_record(contexts)},
            "sources": {"retrieval_mode": sources_mode, "items": [source_record(source) for source in sources]},
        }
    return results


RUNNERS: dict[str, Callable[[RetrievalAdapter, FakeOpenAIServer, Mapping[str, Any]], dict[str, Any]]] = {
    "eval": run_eval_entry,
    "product": run_product_entry,
    "paper_chat": run_paper_chat_entry,
}


def pair_key(arxiv_id: str, question: str) -> str:
    return f"{arxiv_id}:{sha256_text(question)[:16]}"


def build_query_sets() -> dict[str, dict[str, dict[str, Any]]]:
    queries = load_queries(QUERIES_PATH)
    sets: dict[str, dict[str, dict[str, Any]]] = {
        "eval": {query.id: {"query": query.query} for query in queries if query.retrieval_eligible},
        "product": {query.id: {"query": query.query} for query in queries},
        "paper_chat": {},
    }

    def add_pair(arxiv_id: str, question: str, source_id: str) -> None:
        entry = sets["paper_chat"].setdefault(
            pair_key(arxiv_id, question), {"arxiv_id": arxiv_id, "query": question, "source_ids": []}
        )
        if source_id not in entry["source_ids"]:
            entry["source_ids"].append(source_id)

    for query in queries:
        if query.mode == "paper_chat" and query.paper_chat_arxiv_id:
            add_pair(query.paper_chat_arxiv_id, query.query, query.id)
    for line in ANSWERS_PATH.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("mode") == "paper_chat" and row.get("arxiv_id"):
            add_pair(row["arxiv_id"], row["query"], row["id"])
    for entry in sets["paper_chat"].values():
        entry["source_ids"].sort()
    return sets


def build_golden(
    adapter: RetrievalAdapter,
    server: FakeOpenAIServer,
    sets: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    progress: Callable[[str, str], None] | None = None,
) -> dict[str, Any]:
    golden: dict[str, Any] = {
        "format": FORMAT,
        "captured_with": adapter.describe(),
        "settings": {
            "eval_methods": list(EVAL_METHODS),
            "eval_k": EVAL_K,
            "adjacency_window": ADJACENCY_WINDOW,
            "modes": list(MODES),
            "product_limit": PRODUCT_LIMIT,
        },
    }
    for group in GROUPS:
        entries: dict[str, Any] = {}
        for key, entry in sets[group].items():
            if progress is not None:
                progress(group, key)
            entries[key] = {**entry, "results": RUNNERS[group](adapter, server, entry)}
        golden[group] = entries
    return golden


def golden_entry_ids(golden: Mapping[str, Any]) -> list[tuple[str, str]]:
    return [(group, key) for group in GROUPS for key in sorted(golden.get(group, {}))]


def embedding_handler(query_vectors: Mapping[str, Any]) -> EmbeddingVectors:
    if query_vectors.get("source") == "stored":
        return EmbeddingVectors(load_embedding_table(EMBEDDING_TABLE_PATH), strict=True)
    return EmbeddingVectors()
