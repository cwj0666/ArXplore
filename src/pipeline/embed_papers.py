from __future__ import annotations

from typing import Any

from src.integrations import EmbeddingClient, VectorRepository

from .tracing import build_pipeline_trace_config


def _normalize_chunk_limit(max_chunks: int | str | None) -> int:
    """입력값을 안전한 정수 limit으로 정규화한다."""
    return 200 if max_chunks in (None, "") else max(1, int(str(max_chunks)))


def run_embed_papers(
    *,
    runtime: str = "airflow",
    user: str | None = None,
    max_chunks: int | str | None = 200,
    arxiv_id: str | None = None,
    embedding_client: EmbeddingClient | None = None,
    vector_repository: VectorRepository | None = None,
) -> dict[str, Any]:
    """아직 임베딩이 없는 논문 청크를 선택해 pgvector에 저장한다."""
    normalized_limit = _normalize_chunk_limit(max_chunks)
    embedding_client = embedding_client or EmbeddingClient()
    vector_repository = vector_repository or VectorRepository()

    candidates = vector_repository.list_chunks_missing_embeddings(limit=normalized_limit, arxiv_id=arxiv_id)
    if not candidates:
        return {
            "stage": "embed_papers",
            "status": "no_op",
            "selected_chunk_count": 0,
            "embedded_chunk_count": 0,
            "model_name": embedding_client.model_name,
            "sample_embedded": [],
            "trace_config": build_pipeline_trace_config(stage="embed_papers", runtime=runtime, user=user),
        }

    embeddings = embedding_client.embed_texts([chunk["chunk_text"] for chunk in candidates])
    rows = [
        {
            "chunk_id": chunk["chunk_id"],
            "embedding": embedding,
            "model_name": embedding_client.model_name,
        }
        for chunk, embedding in zip(candidates, embeddings, strict=True)
    ]
    vector_repository.upsert_paper_embeddings(rows)

    trace_config = build_pipeline_trace_config(
        stage="embed_papers",
        runtime=runtime,
        user=user,
        extra_metadata={
            "selected_chunk_count": len(candidates),
            "embedded_chunk_count": len(rows),
            "arxiv_id": arxiv_id,
            "model_name": embedding_client.model_name,
        },
    )

    return {
        "stage": "embed_papers",
        "status": "success",
        "selected_chunk_count": len(candidates),
        "embedded_chunk_count": len(rows),
        "model_name": embedding_client.model_name,
        "sample_embedded": [
            {
                "chunk_id": chunk["chunk_id"],
                "arxiv_id": chunk["arxiv_id"],
                "chunk_index": chunk["chunk_index"],
                "section_title": chunk["section_title"],
            }
            for chunk in candidates[:5]
        ],
        "trace_config": trace_config,
    }
