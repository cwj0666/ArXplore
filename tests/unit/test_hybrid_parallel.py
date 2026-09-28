"""hybrid 검색의 lexical/vector 채널 병렬 실행이 순차 실행과 같은 결과·예외를 내는지 확인한다(DB 불필요)."""

from __future__ import annotations

import threading
from contextvars import ContextVar
from types import SimpleNamespace

import psycopg2
import psycopg2.extensions
import pytest
from openai import OpenAIError

from src.core.agent import retrieval
from src.integrations import db
from src.integrations.paper_repository import STRICT_MATCH_BONUS, PaperRepository
from src.integrations.paper_retriever import PaperRetriever
from src.integrations.vector_repository import VectorRepository

BARRIER_TIMEOUT = 5.0


def _lexical_row(chunk_id: int, arxiv_id: str, score: float, section: str = "Method") -> dict:
    return {
        "chunk_id": chunk_id,
        "arxiv_id": arxiv_id,
        "paper_title": f"Paper {arxiv_id}",
        "chunk_text": f"policy loss body {chunk_id}",
        "chunk_index": chunk_id % 7,
        "section_title": section,
        "content_role": "body",
        "score": STRICT_MATCH_BONUS + score,
        "similarity_score": STRICT_MATCH_BONUS + score,
        "retrieval_method": "lexical",
        "score_breakdown": {"strict_match": True, "coverage": 1.0},
        "snippet": "policy loss",
    }


def _vector_row(chunk_id: int, arxiv_id: str, score: float, section: str = "Introduction") -> dict:
    return {
        "chunk_id": chunk_id,
        "arxiv_id": arxiv_id,
        "paper_title": f"Paper {arxiv_id}",
        "paper_abstract": "",
        "chunk_text": f"vector body {chunk_id}",
        "chunk_index": chunk_id % 5,
        "section_title": section,
        "content_role": "body",
        "score": score,
        "similarity_score": score,
        "raw_similarity_score": score,
        "retrieval_method": "vector",
    }


LEXICAL_ROWS = [_lexical_row(i, f"2401.{i % 6:05d}", 1.0 - i * 0.03) for i in range(1, 25)]
VECTOR_ROWS = [_vector_row(i, f"2401.{i % 9:05d}", 0.9 - (i - 10) * 0.02) for i in range(10, 40)]


class FakeRepository:
    def __init__(self, rows=LEXICAL_ROWS, *, error: BaseException | None = None, barrier=None):
        self.rows = rows
        self.error = error
        self.barrier = barrier
        self.calls: list[dict] = []
        self.threads: list[int] = []

    def list_chunk_candidates_by_query(self, query, *, limit, arxiv_id=None):
        self.calls.append({"query": query, "limit": limit, "arxiv_id": arxiv_id})
        self.threads.append(threading.get_ident())
        if self.barrier is not None:
            self.barrier.wait()
        if self.error is not None:
            raise self.error
        rows = [row for row in self.rows if arxiv_id is None or row["arxiv_id"] == arxiv_id]
        return [dict(row) for row in rows[:limit]]

    def list_chunk_windows(self, centers, *, window):
        return [
            [{"chunk_id": index, "arxiv_id": arxiv_id, "chunk_index": index, "chunk_text": "ctx"}]
            for arxiv_id, index in centers
        ]


class FakeEmbeddingClient:
    def __init__(self, *, error: BaseException | None = None, barrier=None, gate: threading.Event | None = None):
        self.error = error
        self.barrier = barrier
        self.gate = gate
        self.threads: list[int] = []
        self.calls = 0

    def is_available(self):
        return True

    def embed_texts(self, texts):
        self.calls += 1
        self.threads.append(threading.get_ident())
        if self.barrier is not None:
            self.barrier.wait()
        if self.gate is not None:
            self.gate.wait(BARRIER_TIMEOUT)
        if self.error is not None:
            raise self.error
        return [[float(len(text))] for text in texts]


class FakeVectorRepository:
    def __init__(self, rows=VECTOR_ROWS, *, error: BaseException | None = None):
        self.rows = rows
        self.error = error
        self.calls: list[dict] = []

    def search_paper_chunks(self, embedding, *, limit, arxiv_id=None):
        self.calls.append({"embedding": embedding, "limit": limit, "arxiv_id": arxiv_id})
        if self.error is not None:
            raise self.error
        rows = [row for row in self.rows if arxiv_id is None or row["arxiv_id"] == arxiv_id]
        return [dict(row) for row in rows[:limit]]


def _pair(**kwargs):
    """같은 가짜 저장소 설정으로 (순차, 병렬) retriever를 만든다."""

    def build(parallel: bool) -> PaperRetriever:
        return PaperRetriever(
            repository=kwargs.get("repository_factory", FakeRepository)(),
            embedding_client=kwargs.get("embedding_factory", FakeEmbeddingClient)(),
            vector_repository=kwargs.get("vector_factory", FakeVectorRepository)(),
            parallel_channels=parallel,
        )

    return build(False), build(True)


def test_parallel_is_the_default():
    retriever = PaperRetriever(repository=object(), embedding_client=object(), vector_repository=object())
    assert retriever.parallel_channels is True


@pytest.mark.parametrize(
    "kwargs",
    [
        {"limit": 5},
        {"limit": 10},
        {"limit": 1},
        {"limit": 5, "arxiv_id": "2401.00003"},
        {"limit": 5, "lexical_limit": 4, "vector_limit": 7},
    ],
)
def test_parallel_matches_sequential(kwargs):
    sequential, parallel = _pair()

    assert parallel.hybrid_fusion_inputs("  policy\x00 loss ", **kwargs) == sequential.hybrid_fusion_inputs(
        "  policy\x00 loss ", **kwargs
    )
    assert parallel.search_paper_chunks_by_hybrid("policy loss", **kwargs) == sequential.search_paper_chunks_by_hybrid(
        "policy loss", **kwargs
    )
    assert parallel.search_paper_contexts_by_hybrid(
        "policy loss", adjacency_window=1, **kwargs
    ) == sequential.search_paper_contexts_by_hybrid("policy loss", adjacency_window=1, **kwargs)
    assert parallel.repository.calls == sequential.repository.calls
    assert parallel.vector_repository.calls == sequential.vector_repository.calls


def test_limits_reach_each_channel():
    _, parallel = _pair()
    parallel.hybrid_fusion_inputs("policy loss", limit=5, arxiv_id="2401.00001", lexical_limit=4, vector_limit=7)

    assert parallel.repository.calls == [{"query": "policy loss", "limit": 4, "arxiv_id": "2401.00001"}]
    assert parallel.vector_repository.calls[0]["limit"] == 7
    assert parallel.vector_repository.calls[0]["arxiv_id"] == "2401.00001"


@pytest.mark.parametrize("parallel_channels", [False, True])
def test_empty_query_touches_nothing(parallel_channels):
    retriever = PaperRetriever(
        repository=FakeRepository(),
        embedding_client=FakeEmbeddingClient(),
        vector_repository=FakeVectorRepository(),
        parallel_channels=parallel_channels,
    )

    assert retriever.hybrid_fusion_inputs("\x00 \t", limit=5) == ("", [], [])
    assert retriever.search_paper_contexts_by_hybrid("\x00", limit=5) == []
    assert retriever.repository.calls == []
    assert retriever.embedding_client.calls == 0


def test_channels_run_concurrently_on_different_threads():
    # 두 채널이 같은 barrier를 통과해야 끝난다. 순차 실행이면 첫 채널이 timeout으로 깨진다.
    barrier = threading.Barrier(2, timeout=BARRIER_TIMEOUT)
    retriever = PaperRetriever(
        repository=FakeRepository(barrier=barrier),
        embedding_client=FakeEmbeddingClient(barrier=barrier),
        vector_repository=FakeVectorRepository(),
    )

    _, lexical, vector = retriever.hybrid_fusion_inputs("policy loss", limit=5)

    assert lexical and vector
    assert retriever.repository.threads == [threading.get_ident()]
    assert retriever.embedding_client.threads[0] != threading.get_ident()


def test_sequential_switch_does_not_overlap():
    barrier = threading.Barrier(2, timeout=0.2)
    retriever = PaperRetriever(
        repository=FakeRepository(barrier=barrier),
        embedding_client=FakeEmbeddingClient(),
        vector_repository=FakeVectorRepository(),
        parallel_channels=False,
    )

    with pytest.raises(threading.BrokenBarrierError):
        retriever.hybrid_fusion_inputs("policy loss", limit=5)
    assert retriever.embedding_client.calls == 0


_REQUEST_SCOPED: ContextVar[str] = ContextVar("request_scoped", default="unset")


def test_worker_thread_sees_request_context():
    seen: list[str] = []

    class RecordingEmbeddingClient(FakeEmbeddingClient):
        def embed_texts(self, texts):
            seen.append(_REQUEST_SCOPED.get())
            return super().embed_texts(texts)

    retriever = PaperRetriever(
        repository=FakeRepository(),
        embedding_client=RecordingEmbeddingClient(),
        vector_repository=FakeVectorRepository(),
    )
    token = _REQUEST_SCOPED.set("request-key")
    try:
        retriever.hybrid_fusion_inputs("policy loss", limit=5)
    finally:
        _REQUEST_SCOPED.reset(token)

    assert seen == ["request-key"]


EMBEDDING_ERROR = OpenAIError("embedding failed")
LEXICAL_SQL_ERROR = psycopg2.errors.QueryCanceled("canceling statement due to statement timeout")
VECTOR_SQL_ERROR = psycopg2.OperationalError("server closed the connection unexpectedly")


@pytest.mark.parametrize(
    ("factories", "expected"),
    [
        ({"embedding_factory": lambda: FakeEmbeddingClient(error=EMBEDDING_ERROR)}, EMBEDDING_ERROR),
        ({"repository_factory": lambda: FakeRepository(error=LEXICAL_SQL_ERROR)}, LEXICAL_SQL_ERROR),
        ({"vector_factory": lambda: FakeVectorRepository(error=VECTOR_SQL_ERROR)}, VECTOR_SQL_ERROR),
        (
            {
                "repository_factory": lambda: FakeRepository(error=LEXICAL_SQL_ERROR),
                "embedding_factory": lambda: FakeEmbeddingClient(error=EMBEDDING_ERROR),
            },
            LEXICAL_SQL_ERROR,
        ),
    ],
    ids=["embedding-openai-error", "lexical-sql-error", "vector-sql-error", "both-fail-lexical-wins"],
)
def test_exceptions_propagate_like_sequential(factories, expected):
    for retriever in _pair(**factories):
        with pytest.raises(type(expected)) as caught:
            retriever.search_paper_contexts_by_hybrid("policy loss", limit=5)
        assert caught.value is expected


def test_lexical_failure_does_not_wait_for_vector():
    # vector가 gate에서 멈춰 있어도 lexical 예외는 바로 올라온다.
    gate = threading.Event()
    retriever = PaperRetriever(
        repository=FakeRepository(error=LEXICAL_SQL_ERROR),
        embedding_client=FakeEmbeddingClient(gate=gate),
        vector_repository=FakeVectorRepository(),
    )
    try:
        with pytest.raises(psycopg2.errors.QueryCanceled):
            retriever.hybrid_fusion_inputs("policy loss", limit=5)
    finally:
        gate.set()


class _FakeCursor:
    def __init__(self, connection):
        self.connection = connection

    def execute(self, sql, params=None):
        self.connection.pool_log.append((threading.get_ident(), self.connection.serial))
        if "set_config" in sql:
            return
        with self.connection.lock:
            self.connection.active += 1
            assert self.connection.active == 1, "a pooled connection was used by two threads at once"
        try:
            self.connection.barrier.wait()
        finally:
            with self.connection.lock:
                self.connection.active -= 1

    def fetchall(self):
        return []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeConnection:
    def __init__(self, serial, barrier, pool_log):
        self.serial = serial
        self.barrier = barrier
        self.pool_log = pool_log
        self.lock = threading.Lock()
        self.active = 0
        self.closed = 0
        self.autocommit = False
        self.info = SimpleNamespace(transaction_status=psycopg2.extensions.TRANSACTION_STATUS_IDLE)

    def cursor(self):
        return _FakeCursor(self)

    def commit(self):
        pass

    def rollback(self):
        pass

    def close(self):
        self.closed = 1


def test_each_channel_borrows_its_own_pooled_connection(monkeypatch):
    # 실제 PaperRepository/VectorRepository와 db 풀을 쓰고 psycopg2.connect만 가짜로 바꾼다.
    # 두 SQL이 barrier에서 동시에 실행 중이어야 통과하므로, 동시에 두 연결이 서로 다른 스레드에 빌려져 있음을 확인한다.
    barrier = threading.Barrier(2, timeout=BARRIER_TIMEOUT)
    pool_log: list[tuple[int, int]] = []
    created: list[_FakeConnection] = []
    lock = threading.Lock()

    def fake_connect(**params):
        with lock:
            connection = _FakeConnection(len(created), barrier, pool_log)
            created.append(connection)
        return connection

    monkeypatch.setattr(db.psycopg2, "connect", fake_connect)
    monkeypatch.setenv("POSTGRES_POOL_MAX", "2")
    db.close_all_pools()
    params = {"dbname": "arxplore", "host": "fake-hybrid-parallel", "port": 5432}
    settings = SimpleNamespace(openai_embedding_model="text-embedding-3-large", vector_min_similarity=0.0)
    repository = PaperRepository(settings=settings)
    vector_repository = VectorRepository(settings=settings)
    repository._build_postgres_connection_params = lambda: params
    vector_repository._build_postgres_connection_params = lambda: params
    try:
        retriever = PaperRetriever(
            repository=repository, embedding_client=FakeEmbeddingClient(), vector_repository=vector_repository
        )
        assert retriever.hybrid_fusion_inputs("policy loss", limit=5) == ("policy loss", [], [])
    finally:
        db.close_all_pools()

    threads_by_connection: dict[int, set[int]] = {}
    for thread_id, serial in pool_log:
        threads_by_connection.setdefault(serial, set()).add(thread_id)
    assert len(created) == 2
    assert all(len(threads) == 1 for threads in threads_by_connection.values())
    assert len({thread for threads in threads_by_connection.values() for thread in threads}) == 2


class TestRetrieveContextsFallback:
    @pytest.fixture(autouse=True)
    def _hybrid_mode(self, monkeypatch):
        monkeypatch.setattr(retrieval, "get_settings", lambda: SimpleNamespace(retrieval_mode="hybrid"))

    @pytest.mark.parametrize("parallel_channels", [False, True])
    def test_embedding_error_falls_back_to_lexical(self, parallel_channels):
        retriever = PaperRetriever(
            repository=FakeRepository(),
            embedding_client=FakeEmbeddingClient(error=EMBEDDING_ERROR),
            vector_repository=FakeVectorRepository(),
            parallel_channels=parallel_channels,
        )

        contexts, mode = retrieval.retrieve_contexts("policy loss", retriever=retriever, limit=5)

        assert mode == "lexical"
        assert contexts == retriever.search_paper_contexts("policy loss", limit=5)
        assert contexts

    def test_hybrid_mode_and_results_match_sequential(self):
        sequential, parallel = _pair()

        assert retrieval.retrieve_contexts("policy loss", retriever=parallel, limit=5) == retrieval.retrieve_contexts(
            "policy loss", retriever=sequential, limit=5
        )
        assert retrieval.retrieve_contexts("policy loss", retriever=parallel, limit=5)[1] == "hybrid"

    @pytest.mark.parametrize("parallel_channels", [False, True])
    def test_sql_error_is_not_swallowed(self, parallel_channels):
        retriever = PaperRetriever(
            repository=FakeRepository(),
            embedding_client=FakeEmbeddingClient(),
            vector_repository=FakeVectorRepository(error=VECTOR_SQL_ERROR),
            parallel_channels=parallel_channels,
        )

        with pytest.raises(psycopg2.OperationalError):
            retrieval.retrieve_contexts("policy loss", retriever=retriever, limit=5)
