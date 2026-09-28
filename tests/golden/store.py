import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Any

GOLDEN_ROOT = Path(__file__).resolve().parent
DATA_DIR = GOLDEN_ROOT / "data"
STRING_CONTEXT_CHARS = 60


def normalize(value: Any) -> Any:
    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"golden data cannot hold non-finite floats: {value!r}")
        return 0.0 if value == 0 else value
    if isinstance(value, dict):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"golden object keys must be strings, got {key!r}")
            normalized[key] = normalize(item)
        return normalized
    if isinstance(value, list | tuple):
        return [normalize(item) for item in value]
    raise TypeError(f"unsupported golden value type: {type(value).__name__}")


def canonical_json(value: Any) -> bytes:
    text = json.dumps(normalize(value), ensure_ascii=True, sort_keys=True, indent=1, allow_nan=False)
    return (text + "\n").encode("ascii")


def write_golden(path: Path, value: Any) -> bytes:
    content = canonical_json(value)
    if path.is_file():
        existing = path.read_bytes()
        if gzip.decompress(existing) == content:
            return existing
    data = gzip.compress(content, compresslevel=9, mtime=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return data


def read_golden(path: Path) -> Any:
    return json.loads(gzip.decompress(path.read_bytes()))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(normalize(value), ensure_ascii=False, indent=2, allow_nan=False)
    path.write_text(text + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()


def md5_text(text: str) -> str:
    return hashlib.md5(text.encode("utf-8", "surrogatepass"), usedforsecurity=False).hexdigest()


def diff(expected: Any, actual: Any, *, limit: int = 30) -> list[str]:
    problems: list[str] = []
    _diff(normalize(expected), normalize(actual), "$", problems, limit)
    return problems


def format_problems(label: str, problems: list[str]) -> str:
    return "\n".join([f"{label}: {len(problems)} difference(s) from golden", *problems])


def _diff(expected: Any, actual: Any, path: str, problems: list[str], limit: int) -> None:
    if len(problems) >= limit:
        return
    if type(expected) is not type(actual):
        problems.append(
            f"{path}: type {type(expected).__name__} -> {type(actual).__name__}"
            f" ({_preview(expected)} -> {_preview(actual)})"
        )
        return
    if isinstance(expected, dict):
        problems.extend(f"{path}.{key}: missing" for key in sorted(expected.keys() - actual.keys()))
        problems.extend(f"{path}.{key}: unexpected" for key in sorted(actual.keys() - expected.keys()))
        for key in sorted(expected.keys() & actual.keys()):
            _diff(expected[key], actual[key], f"{path}.{key}", problems, limit)
        return
    if isinstance(expected, list):
        if len(expected) != len(actual):
            problems.append(f"{path}: length {len(expected)} -> {len(actual)}")
        for index, (left, right) in enumerate(zip(expected, actual, strict=False)):
            _diff(left, right, f"{path}[{index}]", problems, limit)
        return
    if expected != actual:
        problems.append(f"{path}: {_describe_change(expected, actual)}")


def _preview(value: Any) -> str:
    text = repr(value)
    return text if len(text) <= STRING_CONTEXT_CHARS else text[:STRING_CONTEXT_CHARS] + "..."


def _describe_change(expected: Any, actual: Any) -> str:
    if isinstance(expected, str) and isinstance(actual, str) and max(len(expected), len(actual)) > STRING_CONTEXT_CHARS:
        index = next(
            (position for position, (left, right) in enumerate(zip(expected, actual, strict=False)) if left != right),
            min(len(expected), len(actual)),
        )
        start = max(0, index - STRING_CONTEXT_CHARS // 2)
        end = index + STRING_CONTEXT_CHARS
        return (
            f"strings differ at char {index} (lengths {len(expected)} -> {len(actual)}): "
            f"{expected[start:end]!r} -> {actual[start:end]!r}"
        )
    return f"{expected!r} -> {actual!r}"
