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
from tests.golden.adapters import IMPLEMENTATION_ENV, AdapterUnavailable, HttpAdapter, load_adapter
from tests.golden.store import GOLDEN_ROOT, read_golden, write_golden

REPO_ROOT = GOLDEN_ROOT.parents[1]
HTTP_ADAPTER_KIND = "http"
BUNDLE_COPY_PREFIX = f"{environment.COPY_PREFIX}g5_"
RUNNER_MODULE = "tests.golden.g5_runner"
RUNNER_TIMEOUT_SECONDS = 1800
ADAPTER_UNAVAILABLE_EXIT_CODE = 3
SHARED_APP_ENV = {
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


def http_adapter_options() -> dict[str, Any]:
    return {"frontend_dist": g5_http.SPA_DIST}


def load_bundle_http_adapter() -> HttpAdapter:
    return load_adapter(HTTP_ADAPTER_KIND, **http_adapter_options())


def bundle_database_copy_name(bundle_name: str) -> str:
    return f"{BUNDLE_COPY_PREFIX}{bundle_name}"


def bundle_app_environ(server: FakeOpenAIServer, database_url: str) -> dict[str, str]:
    return {
        **environment.harness_environ(openai_environ=server.environ(), database_url=database_url, pg_options=""),
        **SHARED_APP_ENV,
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


def bundle_subprocess_command(bundle_name: str, database_url: str, out: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        RUNNER_MODULE,
        "--bundle",
        bundle_name,
        "--database-url",
        database_url,
        "--out",
        str(out),
    ]


def run_bundle_in_subprocess(bundle_name: str, database_url: str, workdir: Path) -> dict[str, Any]:
    with environment.database_copy(database_url, bundle_database_copy_name(bundle_name)) as copy_url:
        out = workdir / f"{bundle_name}.json.gz"
        command = bundle_subprocess_command(bundle_name, copy_url, out)
        completed = subprocess.run(command, cwd=workdir, env=runner_environ(), timeout=RUNNER_TIMEOUT_SECONDS)
        if completed.returncode == ADAPTER_UNAVAILABLE_EXIT_CODE:
            raise AdapterUnavailable(
                f"no {HTTP_ADAPTER_KIND} adapter for {os.environ.get(IMPLEMENTATION_ENV) or 'legacy'}"
            )
        completed.check_returncode()
        return read_golden(out)


def run_bundles(names: Sequence[str], *, database_url: str) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="golden-g5-") as workdir:
        return {name: run_bundle_in_subprocess(name, database_url, Path(workdir)) for name in names}


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


def run_bundle_child(bundle_name: str, database_url: str, out: Path) -> int:
    if (Path.cwd() / ".env").exists():
        raise SystemExit("the HTTP golden runner must not start next to a .env file")
    bundle = g5_http.BUNDLES[bundle_name]
    if not environment.database_name(database_url).startswith(BUNDLE_COPY_PREFIX):
        raise SystemExit(f"the HTTP golden runner only writes to {BUNDLE_COPY_PREFIX}* database copies")
    apply_setup_sql(database_url, bundle.setup_sql)
    with environment.loopback_only(), FakeOpenAIServer() as server:
        os.environ.update(bundle_app_environ(server, database_url))
        try:
            adapter = load_bundle_http_adapter()
        except AdapterUnavailable as exc:
            print(exc, file=sys.stderr)
            return ADAPTER_UNAVAILABLE_EXIT_CODE
        try:
            scenarios = g5_http.run_bundle(adapter, server, bundle)
            described = adapter.describe()
        finally:
            adapter.close()
    write_golden(out, {"captured_with": described, "scenarios": scenarios})
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog=f"python -m {RUNNER_MODULE}")
    parser.add_argument("--bundle", required=True, choices=sorted(g5_http.BUNDLES))
    parser.add_argument("--database-url", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    return run_bundle_child(args.bundle, args.database_url, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
