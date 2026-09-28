from collections import defaultdict
from typing import Any

from tests.golden import environment
from tests.golden.adapters import ParserAdapter
from tests.golden.g3_pypdf import chunk_record
from tests.golden.store import DATA_DIR, canonical_json, sha256_bytes

FORMAT = 1
GOLDEN_PATH = DATA_DIR / "g2_chunks.json.gz"
INPUT_SQL = """
    SELECT f.arxiv_id, f.text, f.sections, p.abstract
    FROM paper_fulltexts f
    JOIN papers p ON p.arxiv_id = f.arxiv_id
    ORDER BY f.arxiv_id
"""
DATABASE_CHUNKS_SQL = """
    SELECT arxiv_id, chunk_index, md5(chunk_text), section_title, token_count, metadata
    FROM paper_chunks
    ORDER BY arxiv_id, chunk_index
"""
CHUNK_TEXT_SQL = "SELECT chunk_text FROM paper_chunks WHERE arxiv_id = %s AND chunk_index = %s"


def load_inputs(url: str) -> dict[str, tuple[str, list[dict[str, Any]]]]:
    return {
        arxiv_id: (text or abstract or "", list(sections or []))
        for arxiv_id, text, sections, abstract in environment.query_rows(url, INPUT_SQL)
    }


def load_database_chunks(url: str) -> dict[str, list[dict[str, Any]]]:
    chunks: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for arxiv_id, chunk_index, text_md5, section_title, token_count, metadata in environment.query_rows(
        url, DATABASE_CHUNKS_SQL
    ):
        chunks[arxiv_id].append(
            {
                "chunk_index": chunk_index,
                "chunk_text_md5": text_md5,
                "section_title": section_title,
                "token_count": token_count,
                "metadata": metadata,
            }
        )
    return dict(chunks)


def database_chunk_text(url: str, arxiv_id: str, chunk_index: int) -> str | None:
    rows = environment.query_rows(url, CHUNK_TEXT_SQL, (arxiv_id, chunk_index))
    return rows[0][0] if rows else None


def input_digest(text: str, sections: list[dict[str, Any]]) -> str:
    return sha256_bytes(canonical_json({"text": text, "sections": sections}))


def chunk_records(adapter: ParserAdapter, text: str, sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [chunk_record(chunk) for chunk in adapter.build_chunks(text, sections)]


def build_golden(adapter: ParserAdapter, inputs: dict[str, tuple[str, list[dict[str, Any]]]]) -> dict[str, Any]:
    return {
        "format": FORMAT,
        "input": "build_chunks(paper_fulltexts.text or papers.abstract, sections=paper_fulltexts.sections)",
        "papers": {
            arxiv_id: {"input_sha256": input_digest(text, sections), "chunks": chunk_records(adapter, text, sections)}
            for arxiv_id, (text, sections) in inputs.items()
        },
    }
