"""PostgreSQL 스키마(테이블·인덱스)를 멱등하게 생성한다.

리포지토리 생성자는 DDL을 실행하지 않으므로, 새 DB를 쓰거나 스키마가 바뀐 뒤에는 이 스크립트를 1회 실행한다.

    python scripts/migrate_schema.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.integrations.paper_repository import PaperRepository  # noqa: E402
from src.integrations.prepare_job_repository import PrepareJobRepository  # noqa: E402


def main() -> int:
    PaperRepository().ensure_schema()
    print("papers / paper_fulltexts / paper_chunks / paper_embeddings / paper_ai_* schema ensured")
    PrepareJobRepository().ensure_schema()
    print("prepare_jobs schema ensured")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
