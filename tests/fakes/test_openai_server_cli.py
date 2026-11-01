import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import openai
import pytest

from tests.fakes.openai_server import (
    FAKE_API_KEY,
    deterministic_unit_vector,
    embedding_key,
    write_embedding_table,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
STORED = [0.5, -0.5, 0.25, 0.125]
MODEL = "text-embedding-3-large"


@pytest.fixture
def table(tmp_path: Path) -> Path:
    path = tmp_path / "query_embeddings.json.gz"
    write_embedding_table(path, {embedding_key("stored query"): STORED}, model=MODEL, dimensions=4)
    return path


class RunningCli:
    def __init__(self, *options: str) -> None:
        self.process = subprocess.Popen(
            [sys.executable, "-m", "tests.fakes.openai_server", "--port", "0", *options],
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        assert self.process.stdout is not None
        self.base_url = self.process.stdout.readline().strip()
        self.client = openai.OpenAI(api_key=FAKE_API_KEY, base_url=self.base_url, max_retries=0)

    def stop(self) -> str:
        self.process.terminate()
        _, stderr = self.process.communicate(timeout=10)
        return stderr


@pytest.fixture
def start_cli() -> Iterator[type[RunningCli]]:
    started: list[RunningCli] = []

    class Tracked(RunningCli):
        def __init__(self, *options: str) -> None:
            super().__init__(*options)
            started.append(self)

    yield Tracked
    for cli in started:
        if cli.process.poll() is None:
            cli.stop()


def test_a_table_makes_unknown_inputs_fail_and_logs_the_hash(table: Path, start_cli: type[RunningCli]) -> None:
    cli = start_cli("--embedding-table", str(table))

    stored = cli.client.embeddings.create(model=MODEL, input="stored query", dimensions=4)
    with pytest.raises(openai.BadRequestError) as raised:
        cli.client.embeddings.create(model=MODEL, input=["stored query", "unknown query"], dimensions=4)
    stderr = cli.stop()

    prefix = embedding_key("unknown query")[:12]
    assert stored.data[0].embedding == STORED
    assert raised.value.status_code == 400
    assert prefix in str(raised.value)
    assert f"fake server error 400 POST /v1/embeddings: no stored vector for input {prefix}" in stderr
    assert embedding_key("stored query")[:12] not in stderr


def test_lenient_embeddings_restore_generated_vectors(table: Path, start_cli: type[RunningCli]) -> None:
    cli = start_cli("--embedding-table", str(table), "--lenient-embeddings")

    response = cli.client.embeddings.create(model=MODEL, input=["stored query", "unknown query"], dimensions=4)
    stderr = cli.stop()

    assert response.data[0].embedding == STORED
    assert response.data[1].embedding == deterministic_unit_vector(embedding_key("unknown query"), 4)
    assert "fake server error" not in stderr


def test_without_a_table_every_input_gets_a_generated_vector(start_cli: type[RunningCli]) -> None:
    cli = start_cli()

    response = cli.client.embeddings.create(model=MODEL, input="anything", dimensions=4)
    cli.stop()

    assert response.data[0].embedding == deterministic_unit_vector(embedding_key("anything"), 4)


def test_lenient_embeddings_need_a_table() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "tests.fakes.openai_server", "--lenient-embeddings"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 2
    assert "--lenient-embeddings requires --embedding-table" in result.stderr
