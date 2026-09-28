import argparse
import os
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from tests.fakes.openai_server import FakeOpenAIServer
from tests.golden import environment, g5_http
from tests.golden.adapters import IMPLEMENTATION_ENV, load_http_adapter
from tests.golden.store import GOLDEN_ROOT, read_golden, write_golden

REPO_ROOT = GOLDEN_ROOT.parents[1]
RUNNER_TIMEOUT_SECONDS = 1800
HARNESS_APP_ENV = {
    "DJANGO_SETTINGS_MODULE": "arxplore_web.settings",
    "DJANGO_SECRET_KEY": "golden-harness-secret-key-not-for-production",
    "DJANGO_DEBUG": "",
    "RATE_LIMIT_ENABLED": "true",
    "RATE_LIMIT_LLM_PER_MINUTE": "30",
    "RATE_LIMIT_DETAIL_PER_MINUTE": "60",
    "RATE_LIMIT_IP_HEADER": "X-Real-IP",
    "REDIS_URL": "",
    "OPENAI_MODEL": "gpt-4o",
    "AGENT_RECURSION_LIMIT": "12",
    "AGENT_STREAM_BUFFER_CHARS": "120",
    "VECTOR_MIN_SIMILARITY": "0",
}


def runner_environ() -> dict[str, str]:
    values = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": os.environ.get("HOME", "/tmp"),
        "LANG": "C.UTF-8",
        "PYTHONPATH": f"{REPO_ROOT}{os.pathsep}{REPO_ROOT / 'backend'}",
        "PYTHONHASHSEED": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    if os.environ.get(IMPLEMENTATION_ENV):
        values[IMPLEMENTATION_ENV] = os.environ[IMPLEMENTATION_ENV]
    return values


def run_bundles(names: Sequence[str], *, database_url: str) -> dict[str, Any]:
    results: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix="golden-g5-") as workdir:
        for name in names:
            with environment.database_copy(database_url, f"golden_g5_{name}") as copy_url:
                out = Path(workdir) / f"{name}.json.gz"
                subprocess.run(
                    [sys.executable, "-m", "tests.golden.g5_runner", "--bundle", name, "--database-url", copy_url]
                    + ["--out", str(out)],
                    cwd=workdir,
                    env=runner_environ(),
                    check=True,
                    timeout=RUNNER_TIMEOUT_SECONDS,
                )
                results[name] = read_golden(out)
    return results


def apply_setup_sql(url: str, statements: Sequence[str]) -> None:
    if not statements:
        return
    connection = environment.connect(url)
    try:
        cursor = connection.cursor()
        for statement in statements:
            cursor.execute(statement)
        cursor.close()
    finally:
        connection.close()


def run_child(bundle_name: str, database_url: str, out: Path) -> int:
    if (Path.cwd() / ".env").exists():
        raise SystemExit("the HTTP golden runner must not start next to a .env file")
    bundle = g5_http.BUNDLES[bundle_name]
    if not environment.database_name(database_url).startswith(environment.COPY_PREFIX):
        raise SystemExit("the HTTP golden runner only writes to golden_* database copies")
    apply_setup_sql(database_url, bundle.setup_sql)
    with environment.loopback_only(), FakeOpenAIServer() as server:
        os.environ.update(
            environment.harness_environ(openai_environ=server.environ(), database_url=database_url, pg_options="")
        )
        os.environ.update(HARNESS_APP_ENV)
        adapter = load_http_adapter(frontend_dist=g5_http.SPA_DIST)
        try:
            scenarios = g5_http.run_bundle(adapter, server, bundle)
            described = adapter.describe()
        finally:
            adapter.close()
    write_golden(out, {"captured_with": described, "scenarios": scenarios})
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tests.golden.g5_runner")
    parser.add_argument("--bundle", required=True, choices=sorted(g5_http.BUNDLES))
    parser.add_argument("--database-url", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    return run_child(args.bundle, args.database_url, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
