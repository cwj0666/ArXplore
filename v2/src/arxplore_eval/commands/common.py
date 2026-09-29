from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import NoReturn

import typer
from pydantic import ValidationError

from arxplore.config import AppSettings, ConfigError, load_settings
from arxplore.db.pool import Pool, create_pool_from_settings, open_pool
from arxplore.embedding import EmbeddingClient
from arxplore_eval.provenance import find_eval_dir

EXIT_FAILURES = 1
EXIT_PRECONDITION = 2
POOL_ROLE = "eval"
STAMP_FORMAT = "%Y%m%d-%H%M%S"
TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
RESULTS_DIR = "results"
CACHE_DIR = "cache"
QUERIES_FILE = "queries.jsonl"
PLACEHOLDER_KEY_PREFIX = "change-me"
BASE_URL_ENV = "OPENAI_BASE_URL"
BASE_URL_HELP = "OpenAI 호환 API 주소. 가짜 서버로 돌릴 때 쓴다"


class Lang(StrEnum):
    ko = "ko"
    en = "en"


def echo_err(message: str) -> None:
    typer.echo(message, err=True)


def fail(message: str, code: int = EXIT_PRECONDITION) -> NoReturn:
    echo_err(message)
    raise typer.Exit(code=code)


def eval_dir() -> Path:
    return find_eval_dir()


def default_queries_path() -> Path:
    return eval_dir() / QUERIES_FILE


def default_results_dir() -> Path:
    return eval_dir() / RESULTS_DIR


def default_cache_dir() -> Path:
    return eval_dir() / CACHE_DIR


def repo_root() -> Path:
    return eval_dir().parent


def now() -> datetime:
    return datetime.now()


def stamp_of(moment: datetime) -> str:
    return moment.strftime(STAMP_FORMAT)


def split_names(value: str) -> list[str]:
    return [name.strip() for name in value.split(",") if name.strip()]


def lang_value(lang: Lang | None) -> str | None:
    return None if lang is None else lang.value


def settings_or_exit() -> AppSettings:
    try:
        return load_settings()
    except (ConfigError, ValidationError, ValueError) as exc:
        fail(f"설정을 불러올 수 없습니다(.env 확인): {exc}")


def usable_api_key(settings: AppSettings) -> str | None:
    key = (settings.openai_api_key or "").strip()
    if not key or key.startswith(PLACEHOLDER_KEY_PREFIX):
        return None
    return key


def embedding_client(settings: AppSettings, base_url: str | None) -> EmbeddingClient:
    return EmbeddingClient.from_settings(settings, base_url=base_url)


@asynccontextmanager
async def eval_pool(settings: AppSettings) -> AsyncIterator[Pool]:
    pool = create_pool_from_settings(settings, role=POOL_ROLE)
    async with open_pool(pool):
        yield pool


def count_text(values: Sequence[str]) -> str:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return ", ".join(f"{key} {count}" for key, count in sorted(counts.items()))
