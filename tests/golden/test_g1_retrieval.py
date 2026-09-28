from collections.abc import Iterator
from typing import Any

import pytest

from tests.fakes.openai_server import FakeOpenAIServer
from tests.golden import environment, g1_retrieval
from tests.golden.adapters import AdapterUnavailable, implementation_name, load_retrieval_adapter
from tests.golden.store import diff, format_problems, read_golden

GOLDEN = read_golden(g1_retrieval.GOLDEN_PATH) if g1_retrieval.GOLDEN_PATH.is_file() else {}
ENTRY_IDS = g1_retrieval.golden_entry_ids(GOLDEN)


@pytest.fixture(scope="module")
def g1_harness(golden_database_url: str, golden_pg_options: str) -> Iterator[tuple[Any, FakeOpenAIServer]]:
    vectors = g1_retrieval.embedding_handler(GOLDEN.get("query_vectors", {}))
    with FakeOpenAIServer(embedding_handler=vectors) as server, pytest.MonkeyPatch.context() as patch:
        values = environment.harness_environ(
            openai_environ=server.environ(), database_url=golden_database_url, pg_options=golden_pg_options
        )
        for key, value in values.items():
            patch.setenv(key, value)
        try:
            adapter = load_retrieval_adapter()
        except AdapterUnavailable as exc:
            pytest.skip(f"{implementation_name()}: {exc}")
        try:
            yield adapter, server
        finally:
            adapter.close()


def test_golden_covers_every_group():
    assert GOLDEN.get("format") == g1_retrieval.FORMAT
    assert {group: len(GOLDEN[group]) for group in g1_retrieval.GROUPS} == {
        "eval": 111,
        "product": 142,
        "paper_chat": 97,
    }


@pytest.mark.parametrize(("group", "key"), ENTRY_IDS, ids=[f"{group}:{key}" for group, key in ENTRY_IDS])
def test_retrieval_matches_golden(g1_harness, group, key):
    adapter, server = g1_harness
    entry = GOLDEN[group][key]
    server.reset()
    actual = g1_retrieval.RUNNERS[group](adapter, server, entry)

    assert server.errors == []
    problems = diff(entry["results"], actual)
    assert not problems, format_problems(f"{group} {key}", problems)
