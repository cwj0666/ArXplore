"""제목 부분문자열('reference') 규칙으로 references로 잘못 분류된 청크를 body로 되돌린다. 기본은 dry-run.

    python scripts/backfill_content_roles.py           # 영향 논문·청크·임베딩 수 출력
    python scripts/backfill_content_roles.py --apply   # metadata.content_role을 body로 갱신

대상: content_role='references' 이면서 섹션 제목이 옛 규칙('reference' 포함 또는 'bibliography'로 시작)에는
걸리지만 새 규칙(pdf_parser.section_roles.is_references_section_title)에는 걸리지 않는 청크.
제목과 무관하게 본문 휴리스틱으로 references가 된 청크는 건드리지 않는다.
청크·임베딩은 삭제하지 않으며, 갱신된 청크는 prepare-worker의 임베딩 backlog가 이어서 처리한다.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from collections.abc import Callable, Iterable
from contextlib import contextmanager
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.integrations.pdf_parser.section_roles import is_references_section_title  # noqa: E402

CANDIDATE_SQL = """
    SELECT c.id, c.arxiv_id, c.section_title, (e.chunk_id IS NOT NULL) AS has_embedding
    FROM paper_chunks c
    LEFT JOIN paper_embeddings e ON e.chunk_id = c.id
    WHERE c.metadata->>'content_role' = 'references'
      AND (c.section_title ILIKE %s OR c.section_title ILIKE %s)
    ORDER BY c.arxiv_id ASC, c.chunk_index ASC
"""
CANDIDATE_PARAMS = ("%reference%", "bibliography%")

UPDATE_SQL = """
    UPDATE paper_chunks
    SET
        metadata = metadata || jsonb_build_object('content_role', 'body', 'content_role_backfilled_from', 'references'),
        updated_at = NOW()
    WHERE id = ANY(%s)
      AND metadata->>'content_role' = 'references'
    RETURNING id
"""

NEEDS_EMBEDDING_SQL = """
    SELECT COUNT(*)
    FROM paper_chunks c
    LEFT JOIN paper_embeddings e ON e.chunk_id = c.id
    WHERE c.id = ANY(%s)
      AND e.chunk_id IS NULL
"""


def matched_old_title_rule(section_title: str | None) -> bool:
    lowered = str(section_title or "").lower()
    return "reference" in lowered or lowered.startswith("bibliography")


def should_reclassify(section_title: str | None) -> bool:
    return matched_old_title_rule(section_title) and not is_references_section_title(str(section_title or ""))


def select_targets(rows: Iterable[tuple[Any, ...]]) -> list[dict[str, Any]]:
    return [
        {"chunk_id": int(row[0]), "arxiv_id": row[1], "section_title": row[2], "has_embedding": bool(row[3])}
        for row in rows
        if should_reclassify(row[2])
    ]


def summarize_targets(targets: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "papers": len({target["arxiv_id"] for target in targets}),
        "chunks": len(targets),
        "embeddings": sum(1 for target in targets if target["has_embedding"]),
        "titles": Counter(str(target["section_title"] or "") for target in targets),
    }


def fetch_targets(cursor) -> list[dict[str, Any]]:
    cursor.execute(CANDIDATE_SQL, CANDIDATE_PARAMS)
    return select_targets(cursor.fetchall())


def apply_reclassification(cursor, chunk_ids: list[int]) -> dict[str, int]:
    if not chunk_ids:
        return {"updated_chunks": 0, "needs_embedding": 0}
    cursor.execute(UPDATE_SQL, (chunk_ids,))
    updated_ids = [int(row[0]) for row in cursor.fetchall()]
    if not updated_ids:
        return {"updated_chunks": 0, "needs_embedding": 0}
    cursor.execute(NEEDS_EMBEDDING_SQL, (updated_ids,))
    row = cursor.fetchone()
    return {"updated_chunks": len(updated_ids), "needs_embedding": int(row[0] or 0) if row else 0}


@contextmanager
def _default_connection():
    import psycopg2

    from src.shared import build_postgres_connection_params, get_settings

    connection = psycopg2.connect(**build_postgres_connection_params(get_settings()))
    try:
        yield connection
    finally:
        connection.close()


def main(argv: list[str] | None = None, *, connection_factory: Callable | None = None) -> int:
    parser = argparse.ArgumentParser(description="잘못 references로 분류된 청크의 content_role을 body로 되돌린다 (기본 dry-run).")
    parser.add_argument("--apply", action="store_true", help="지정하면 실제로 metadata를 갱신한다.")
    parser.add_argument("--show-titles", type=int, default=30, help="출력할 섹션 제목 상위 개수.")
    args = parser.parse_args(argv)

    connection_factory = connection_factory or _default_connection
    with connection_factory() as connection:
        try:
            with connection.cursor() as cursor:
                targets = fetch_targets(cursor)
                summary = summarize_targets(targets)
                label = "[APPLY]" if args.apply else "[DRY-RUN]"
                print(
                    f"{label} 영향 논문 {summary['papers']}편, 청크 {summary['chunks']}개, "
                    f"기존 임베딩 {summary['embeddings']}개"
                )
                for title, count in summary["titles"].most_common(max(0, args.show_titles)):
                    print(f"  {count:6d}  {title}")

                if not args.apply:
                    connection.rollback()
                    if targets:
                        print("실제로 갱신하려면 --apply를 붙여 다시 실행한다.")
                    return 0

                applied = apply_reclassification(cursor, [target["chunk_id"] for target in targets])
            connection.commit()
        except Exception:
            connection.rollback()
            raise

    print(
        f"[APPLY] content_role=body로 갱신한 청크 {applied['updated_chunks']}개, "
        f"임베딩이 필요한 청크 {applied['needs_embedding']}개 (prepare-worker 임베딩 backlog가 처리)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
