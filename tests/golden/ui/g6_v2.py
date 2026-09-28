import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from collections.abc import Iterator, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from tests.fakes.openai_server import FakeOpenAIServer
from tests.golden import environment, g5_runner, g6_ui
from tests.golden.store import DATA_DIR

UI_DIR = Path(__file__).resolve().parent
REPO_ROOT = UI_DIR.parents[2]
V2_FRONTEND = REPO_ROOT / "v2" / "frontend"
V2_DIST = V2_FRONTEND / "dist"
VARIANT_DIR = DATA_DIR / "g6_ui.v2"
COPY_PREFIX = "v2_test_ui_"
PORT_RANGE = range(18200, 18300)
SERVER_START_SECONDS = 90
CAPTURE_TIMEOUT_SECONDS = 1200
SCREEN_SUFFIXES = (".txt", ".aria.yml", ".png")


def free_port(taken: set[int]) -> int:
    for port in PORT_RANGE:
        if port in taken:
            continue
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            try:
                probe.bind(("127.0.0.1", port))
            except OSError:
                continue
        taken.add(port)
        return port
    raise RuntimeError(f"no free port in {PORT_RANGE.start}-{PORT_RANGE.stop - 1}")


@contextmanager
def database_copy(name: str) -> Iterator[str]:
    if not name.startswith(COPY_PREFIX) or not name.replace("_", "").isalnum():
        raise ValueError(f"UI database copies must be named {COPY_PREFIX}<alnum>: {name!r}")
    base_url = environment.database_url()
    maintenance_url = environment.with_database(base_url, "postgres")

    def execute(statement: str) -> None:
        connection = environment.connect(maintenance_url)
        try:
            cursor = connection.cursor()
            cursor.execute(statement)
            cursor.close()
        finally:
            connection.close()

    execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
    execute(f'CREATE DATABASE "{name}" TEMPLATE "{environment.TEMPLATE_DATABASE}"')
    try:
        yield environment.with_database(base_url, name)
    finally:
        execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')


def wait_until_ready(url: str, process: subprocess.Popen[bytes], log_path: Path, label: str) -> None:
    deadline = time.monotonic() + SERVER_START_SECONDS
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"{label} exited early:\n{log_path.read_text(errors='replace')[-3000:]}")
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return
        except OSError:
            time.sleep(0.5)
    raise RuntimeError(f"{label} did not answer {url}:\n{log_path.read_text(errors='replace')[-3000:]}")


@contextmanager
def running(command: list[str], *, cwd: Path, env: dict[str, str], log_path: Path) -> Iterator[subprocess.Popen[bytes]]:
    with log_path.open("wb") as log:
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            yield process
        finally:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=30)


def web_environ(django_port: int) -> dict[str, str]:
    return {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": os.environ.get("HOME", "/tmp"),
        "API_PROXY_TARGET": f"http://127.0.0.1:{django_port}",
    }


def capture_v2(out_dir: Path) -> dict[str, object]:
    if not (V2_DIST / "index.html").is_file():
        raise FileNotFoundError(f"{V2_DIST} is missing; run npm ci && npm run build in v2/frontend")
    pdf = g6_ui.detail_pdf()
    out_dir.mkdir(parents=True, exist_ok=True)
    taken: set[int] = set()
    fake_port = free_port(taken)
    django_port = free_port(taken)
    web_port = free_port(taken)
    with ExitStack() as stack:
        workdir = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="v2-g6-")))
        copy_url = stack.enter_context(database_copy(f"{COPY_PREFIX}g6"))
        server = stack.enter_context(FakeOpenAIServer(chat_handler=g6_ui.ui_reply, port=fake_port))
        django_env = {
            **g5_runner.runner_environ(),
            **g5_runner.bundle_app_environ(server, copy_url),
            **environment.LEGACY_DJANGO_ENV,
            "RATE_LIMIT_ENABLED": "false",
        }
        django_log = workdir / "django.log"
        django = stack.enter_context(
            running(
                [sys.executable, str(REPO_ROOT / "backend" / "manage.py"), "runserver", f"127.0.0.1:{django_port}"]
                + ["--insecure", "--noreload"],
                cwd=workdir,
                env=django_env,
                log_path=django_log,
            )
        )
        wait_until_ready(f"http://127.0.0.1:{django_port}/bootstrap.json", django, django_log, "Django")
        web_log = workdir / "web.log"
        web = stack.enter_context(
            running(
                [str(V2_FRONTEND / "node_modules" / ".bin" / "vite"), "preview", "--host", "127.0.0.1"]
                + ["--port", str(web_port), "--strictPort"],
                cwd=V2_FRONTEND,
                env=web_environ(django_port),
                log_path=web_log,
            )
        )
        base_url = f"http://127.0.0.1:{web_port}"
        wait_until_ready(f"{base_url}/", web, web_log, "vite preview")
        wait_until_ready(f"{base_url}/bootstrap.json", web, web_log, "vite preview proxy")
        subprocess.run(
            ["node", str(g6_ui.CAPTURE_SCRIPT), "--base-url", base_url, "--out", str(out_dir), "--pdf", str(pdf)]
            + ["--paper", g6_ui.DETAIL_PAPER, "--missing-paper", g6_ui.MISSING_PAPER, "--impl", "v2"],
            cwd=UI_DIR,
            check=True,
            timeout=CAPTURE_TIMEOUT_SECONDS,
        )
        errors = server.errors
    if errors:
        raise RuntimeError(f"fake server errors during the UI capture: {errors[:3]}")
    return json.loads((out_dir / "captures.json").read_text(encoding="utf-8"))


def screen_names() -> list[str]:
    manifest = json.loads((g6_ui.GOLDEN_DIR / "captures.json").read_text(encoding="utf-8"))
    return [capture["name"] for capture in manifest["captures"]]


def expected_file(name: str, suffix: str) -> Path:
    variant = VARIANT_DIR / f"{name}{suffix}"
    return variant if variant.is_file() else g6_ui.GOLDEN_DIR / f"{name}{suffix}"


def pixel_difference(expected: Path, actual: Path, diff_path: Path | None) -> tuple[int, str]:
    with Image.open(expected) as left_image, Image.open(actual) as right_image:
        left = np.asarray(left_image.convert("RGBA"))
        right = np.asarray(right_image.convert("RGBA"))
    if left.shape != right.shape:
        height = max(left.shape[0], right.shape[0])
        width = max(left.shape[1], right.shape[1])
        return height * width, f"size {left.shape[1]}x{left.shape[0]} -> {right.shape[1]}x{right.shape[0]}"
    changed = np.any(left != right, axis=-1)
    count = int(changed.sum())
    if count and diff_path is not None:
        overlay = right.copy()
        overlay[changed] = (255, 0, 0, 255)
        Image.fromarray(overlay).save(diff_path)
    if not count:
        return 0, ""
    rows = np.flatnonzero(changed.any(axis=1))
    cols = np.flatnonzero(changed.any(axis=0))
    return count, f"box x={cols[0]}..{cols[-1]} y={rows[0]}..{rows[-1]}"


@dataclass(frozen=True)
class ScreenResult:
    name: str
    text_equal: bool
    aria_equal: bool
    pixels: int
    detail: str
    variant: bool

    def describe(self) -> str:
        source = "v2 variant" if self.variant else "original"
        pixel_note = f"{self.pixels} px" + (f" ({self.detail})" if self.detail else "")
        return (
            f"{self.name:22} text={'same' if self.text_equal else 'DIFF'} aria={'same' if self.aria_equal else 'DIFF'}"
            f" png={pixel_note} [{source}]"
        )


def compare(actual_dir: Path, *, write_diffs: bool = True) -> list[ScreenResult]:
    results = []
    for name in screen_names():
        expected_text = expected_file(name, ".txt")
        expected_aria = expected_file(name, ".aria.yml")
        expected_png = expected_file(name, ".png")
        pixels, detail = pixel_difference(
            expected_png, actual_dir / f"{name}.png", actual_dir / f"{name}.diff.png" if write_diffs else None
        )
        results.append(
            ScreenResult(
                name=name,
                text_equal=expected_text.read_bytes() == (actual_dir / f"{name}.txt").read_bytes(),
                aria_equal=expected_aria.read_bytes() == (actual_dir / f"{name}.aria.yml").read_bytes(),
                pixels=pixels,
                detail=detail,
                variant=any(path.parent == VARIANT_DIR for path in (expected_text, expected_aria, expected_png)),
            )
        )
    return results


def write_variant(actual_dir: Path, names: Sequence[str]) -> None:
    VARIANT_DIR.mkdir(parents=True, exist_ok=True)
    for name in names:
        for suffix in SCREEN_SUFFIXES:
            captured = actual_dir / f"{name}{suffix}"
            variant = VARIANT_DIR / f"{name}{suffix}"
            if captured.read_bytes() == (g6_ui.GOLDEN_DIR / f"{name}{suffix}").read_bytes():
                variant.unlink(missing_ok=True)
            else:
                shutil.copyfile(captured, variant)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tests.golden.ui.g6_v2")
    parser.add_argument("--out", type=Path, help="keep the capture in this directory")
    parser.add_argument("--reuse", action="store_true", help="compare an existing capture in --out without capturing")
    parser.add_argument("--write-variant", nargs="+", metavar="SCREEN", default=[], help="store screens as v2 variants")
    args = parser.parse_args(argv)
    with tempfile.TemporaryDirectory(prefix="v2-g6-capture-") as scratch:
        out_dir = args.out or Path(scratch)
        if not args.reuse:
            started = time.perf_counter()
            capture_v2(out_dir)
            print(f"captured in {time.perf_counter() - started:.0f}s -> {out_dir}")
        if args.write_variant:
            unknown = sorted(set(args.write_variant) - set(screen_names()))
            if unknown:
                parser.error(f"unknown screens: {unknown}")
            write_variant(out_dir, args.write_variant)
        results = compare(out_dir)
    for result in results:
        print(result.describe())
    mismatched = [result for result in results if not (result.text_equal and result.aria_equal and not result.pixels)]
    print("identical" if not mismatched else f"{len(mismatched)} screen(s) differ")
    return 1 if mismatched else 0


if __name__ == "__main__":
    raise SystemExit(main())
