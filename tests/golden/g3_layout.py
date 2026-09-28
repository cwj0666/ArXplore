import json
import os
import sys
import time
from typing import Any

from tests.golden.adapters import ParserAdapter
from tests.golden.g3_pypdf import parsed_record
from tests.golden.store import DATA_DIR, sha256_bytes

FORMAT = 1
GOLDEN_DIR = DATA_DIR / "g3_layout"
LAYOUT_URL_ENV = "GOLDEN_LAYOUT_PARSER_URL"
DEFAULT_LAYOUT_URL = "http://127.0.0.1:5060"
REQUEST_OPTIONS = {"fast": "false", "parse_tables_and_math": "false"}
REQUEST_TIMEOUT_SECONDS = 900
REQUEST_ATTEMPTS = 3
RETRY_PAUSE_SECONDS = 3.0
PARSED_KEYS = ("parse", "chunks", "chunk_metrics")


def layout_url() -> str:
    return os.environ.get(LAYOUT_URL_ENV, "").strip() or DEFAULT_LAYOUT_URL


def golden_path(entry: dict[str, Any]) -> Any:
    return GOLDEN_DIR / f"{entry['arxiv_id']}.json.gz"


def request_layout(content: bytes, url: str | None = None) -> bytes:
    import httpx

    for attempt in range(1, REQUEST_ATTEMPTS + 1):
        response = httpx.post(
            f"{(url or layout_url()).rstrip('/')}/",
            files={"file": ("document.pdf", content, "application/pdf")},
            data=REQUEST_OPTIONS,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        if response.status_code < 400 or attempt == REQUEST_ATTEMPTS:
            response.raise_for_status()
            return response.content
        print(f"layout parser answered {response.status_code}; retrying ({attempt})", file=sys.stderr, flush=True)
        time.sleep(RETRY_PAUSE_SECONDS)
    raise AssertionError("unreachable")


def replay_record(adapter: ParserAdapter, payload: list[Any]) -> dict[str, Any] | None:
    parsed = adapter.parse_layout_response(payload)
    return None if parsed is None else parsed_record(adapter, parsed)


def build_record(adapter: ParserAdapter, entry: dict[str, Any], pdf_bytes: bytes, response: bytes) -> dict[str, Any]:
    payload = json.loads(response)
    replayed = replay_record(adapter, payload)
    if replayed is None:
        raise ValueError(f"{entry['arxiv_id']}: the layout response produced no text")
    return {
        "format": FORMAT,
        "arxiv_id": entry["arxiv_id"],
        "version": entry["version"],
        "pdf_sha256": sha256_bytes(pdf_bytes),
        "request": REQUEST_OPTIONS,
        "response_sha256": sha256_bytes(response),
        "response": payload,
        **replayed,
    }
