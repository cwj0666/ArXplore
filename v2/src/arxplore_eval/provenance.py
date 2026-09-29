import hashlib
import os
import platform
import subprocess
from pathlib import Path

from arxplore_eval.dataset import dataset_digest

EVAL_DIR_ENV = "ARXPLORE_EVAL_DIR"
EVAL_DIR_NAME = "eval"
QUERIES_FILE = "queries.jsonl"
UNKNOWN_REVISION = "unknown"


def find_eval_dir(start: Path | None = None) -> Path:
    configured = os.environ.get(EVAL_DIR_ENV, "").strip()
    if configured:
        return Path(configured).expanduser()
    origin = (start or Path.cwd()).resolve()
    for directory in (origin, *origin.parents):
        candidate = directory / EVAL_DIR_NAME
        if (candidate / QUERIES_FILE).is_file():
            return candidate
    return origin / EVAL_DIR_NAME


def _git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, check=True).stdout.strip()


def git_revision(root: Path) -> str:
    try:
        commit = _git(root, "rev-parse", "--short", "HEAD")
        dirty = _git(root, "status", "--porcelain", "--untracked-files=no")
    except OSError, subprocess.CalledProcessError:
        return UNKNOWN_REVISION
    return f"{commit}{' (dirty)' if dirty else ''}"


def file_digest(path: Path) -> str:
    if path.is_dir():
        return dataset_digest(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def display_path(path: Path, root: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _first_value(path: Path, prefix: str) -> str | None:
    for line in path.read_text().splitlines():
        if line.startswith(prefix):
            return line.split(":", 1)[1].strip()
    return None


def machine_info() -> dict[str, str]:
    cpu = "unknown"
    memory = "unknown"
    try:
        cpu = _first_value(Path("/proc/cpuinfo"), "model name") or cpu
        total = _first_value(Path("/proc/meminfo"), "MemTotal")
        if total:
            memory = f"{int(total.split()[0]) / 1024 / 1024:.1f} GiB"
    except OSError:
        pass
    release = platform.release()
    wsl = "WSL2" if "microsoft" in release.lower() else "아님"
    return {
        "머신": f"{cpu}, 논리 코어 {os.cpu_count()}, RAM {memory}",
        "OS": f"{platform.system()} {release} (WSL: {wsl}), Python {platform.python_version()}",
    }
