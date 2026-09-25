"""PostgreSQL 스키마(테이블·인덱스)를 멱등하게 생성한다.

리포지토리 생성자는 DDL을 실행하지 않으므로, 새 DB를 쓰거나 스키마가 바뀐 뒤에는 이 스크립트를 1회 실행한다.

    python scripts/migrate_schema.py

기존 DB에서 처음 실행하면 `papers`/`paper_chunks`에 tsvector 생성 컬럼을 추가하면서 테이블을 다시 쓰고
GIN·HNSW 인덱스를 만든다. 이 동안 두 테이블에 쓰기 잠금이 걸리므로 prepare-worker를 멈춘 상태에서 실행한다.
HNSW 인덱스는 pgvector 0.5.0 이상에서만 만들어지고, 없으면 경고만 남기고 계속한다.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.integrations.db import close_all_pools  # noqa: E402
from src.integrations.paper_repository import PaperRepository  # noqa: E402
from src.integrations.prepare_job_repository import PrepareJobRepository  # noqa: E402

SEARCH_INDEXES = (
    "idx_papers_title_abstract_vector",
    "idx_paper_chunks_chunk_vector",
    "paper_embeddings_embedding_hnsw",
)


def report_search_indexes(repository: PaperRepository) -> dict[str, bool]:
    with repository._connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            "SELECT indexname FROM pg_indexes WHERE schemaname = current_schema() AND indexname = ANY(%s)",
            (list(SEARCH_INDEXES),),
        )
        present = {row[0] for row in cursor.fetchall()}
    return {name: name in present for name in SEARCH_INDEXES}


def main() -> int:
    try:
        paper_repository = PaperRepository()
        paper_repository.ensure_schema()
        print("papers / paper_fulltexts / paper_chunks / paper_embeddings / paper_ai_* schema ensured")
        for name, present in report_search_indexes(paper_repository).items():
            print(f"  {name}: {'present' if present else 'MISSING'}")
        PrepareJobRepository().ensure_schema()
        print("prepare_jobs schema ensured")
    finally:
        close_all_pools()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
